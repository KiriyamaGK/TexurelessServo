import collections
import random

import torch
import math
import numpy as np
import torch.nn as nn
from ultralytics import YOLO

from networks.Network import NetworkBase
from networks.helpers import get_activation_fn
from networks.SpatialSoftmax import SpatialSoftmax
from torchvision import models
from torch.nn import functional as F
import torch.distributions as D
import functools
import operator
from utils.input_process import add_gaussian_spot_to_image,make_scaled_img

class FixableSequential(torch.nn.Sequential):
    def __init__(self, fixed, *args, **kwargs):
        torch.nn.Sequential.__init__(self, *args, **kwargs)
        self.fixed = fixed

    def train(self, mode):
        if self.fixed:
            super().train(False)
        else:
            super().train(mode)

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the configs """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].
        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:
        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the configs"

            # overwrite the attribute
            print("command line overwriting configs attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CfgNode()
        # either model_type or (n_layer, n_head, n_embd) must be given in the configs
        C.model_type = None # 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1 # default 0.1
        C.resid_pdrop = 0.1 # default 0.1
        C.attn_pdrop = 0.1 # default 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t, _ = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = idx  #  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # print('tok_emb.shape, pos_emb.shape: ', tok_emb.shape, pos_emb.shape)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class Transformer(NetworkBase):
    def __init__(self, input_low_dim, output_dim, obs_keys,batch_size,seq_length,training,embedding_size=656,n_layer=4,n_head=4,block_size=10,low_dim_hidden_sizes=None,output_head_sizes=None,activation="relu", output_activation=None,use_gmm=False,
                 encoder=None):
        super().__init__(input_low_dim, output_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.freeze_encoder=encoder['freeze']
        #img
        self.encoder_name = encoder['name']
        assert self.encoder_name in ['ResNet18','YOLO_v11']
        if self.encoder_name == 'YOLO_v11':
            self.pretrained_path=encoder['pretrained_path']
            self.img_enc_num_layers=encoder['params']['num_layers']
            yolo_mdl=YOLO(self.pretrained_path)
            self.img_enc = nn.Sequential(*list(yolo_mdl.model.model)[:self.img_enc_num_layers])
        elif self.encoder_name == 'ResNet18':
            self.img_enc = torch.nn.Sequential(*(list(models.resnet18().children())[:-2]))

        if self.freeze_encoder:
            for param in self.img_enc.parameters():
                param.requires_grad = False

        self.use_siamese = encoder['siamese']
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.ss_num_kp=encoder['params']['SpatialSoftmax']['num_kp']
        self.ss_in_c=encoder['params']['SpatialSoftmax']['in_c']
        self.ss_in_h=encoder['params']['SpatialSoftmax']['in_h']
        self.ss_in_w=encoder['params']['SpatialSoftmax']['in_w']

        self.input_low_dim = input_low_dim
        self.output_dim = output_dim
        self.img_size=encoder['params']['img_size']
        self.crop_size=encoder['params']['crop_size']

        self.obs_keys=obs_keys
        self.low_dim_keys=[]
        for obs_key in obs_keys:
            if "image" not in obs_key and "img" not in obs_key:
                self.low_dim_keys.append(obs_key)

        self.low_dim_hidden_sizes = low_dim_hidden_sizes
        self.output_head_sizes = output_head_sizes
        self.feat_dim = embedding_size
        self.model_config = GPT.get_default_config()
        self.model_config.vocab_size = embedding_size
        self.model_config.n_embd = embedding_size
        self.model_config.n_layer = n_layer
        self.model_config.n_head = n_head
        self.model_config.block_size = block_size
        self.gpt_model = GPT(self.model_config)
        self.use_GMM = use_gmm
        self.is_training = training

        if self.is_training:
            self.use_data_augmentation=encoder['data_augmentation']
            self.use_tcl_loss=encoder['task_consistency_loss']
        else:
            self.use_data_augmentation=False
            self.use_tcl_loss=False
        if self.use_tcl_loss:
            assert self.use_data_augmentation

        self.activation = get_activation_fn(activation)
        self.output_activation = get_activation_fn(output_activation) if output_activation is not None else None

        self.spatial_softmax=SpatialSoftmax(self.ss_in_c,self.ss_in_h,self.ss_in_w,self.ss_num_kp)
        self.ee_ln = nn.Linear(self.ss_num_kp * 2, (self.feat_dim-self.low_dim_hidden_sizes[-1])//2)
        self.grid_source = self.build_grid(self.crop_size, self.crop_size)

        if not self.use_siamese:
            if self.encoder_name=='ResNet18':
                self.img_enc_goal = torch.nn.Sequential(*(list(models.resnet18().children())[:-2]))
            else:
                self.img_enc_goal = nn.Sequential(*list(yolo_mdl.model.model)[:self.img_enc_num_layers])
            self.spatial_softmax_goal = SpatialSoftmax(self.ss_in_c, self.ss_in_h, self.ss_in_w, self.ss_num_kp)
            self.ee_ln_goal = nn.Linear(self.ss_num_kp * 2, (self.feat_dim - self.low_dim_hidden_sizes[-1]) // 2)
            if self.freeze_encoder:
                for param in self.img_enc_goal.parameters():
                    param.requires_grad = False

        #for low_dim
        self.mlp_pos = nn.Sequential(
            nn.Linear(self.input_low_dim, self.low_dim_hidden_sizes[0]),
            self.activation(),
            nn.Linear(self.low_dim_hidden_sizes[0], self.low_dim_hidden_sizes[1]),
            self.activation(),
            nn.Linear(self.low_dim_hidden_sizes[1], self.low_dim_hidden_sizes[2]),
        )
        self.buffer = []

        if self.use_GMM:
            self.gmm_modes = 5
            self.mlp_decoder_mean = nn.Sequential(
                nn.Linear(self.feat_dim, self.output_head_sizes[0]),
                self.activation(),
                nn.Linear(self.output_head_sizes[0], self.output_head_sizes[1]),
                self.activation(),
                nn.Linear(self.output_head_sizes[1], self.output_dim * self.gmm_modes),
            )
            self.mlp_decoder_scale = nn.Sequential(
                nn.Linear(self.feat_dim, self.output_head_sizes[0]),
                self.activation(),
                nn.Linear(self.output_head_sizes[0], self.output_head_sizes[1]),
                self.activation(),
                nn.Linear(self.output_head_sizes[1], self.output_dim * self.gmm_modes),
            )
            self.mlp_decoder_logits = nn.Sequential(
                nn.Linear(self.feat_dim, self.output_head_sizes[0]),
                self.activation(),
                nn.Linear(self.output_head_sizes[0], self.output_head_sizes[1]),
                self.activation(),
                nn.Linear(self.output_head_sizes[1],  self.gmm_modes),
            )
            self.min_std=0.0001
            self.activations = {
                "softplus": F.softplus,
                "exp": torch.exp,
            }
            self.std_activation = "softplus"
            self.low_noise_eval = False
        else:
            self.mlp_output_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.output_head_sizes[0]),
            self.activation(),
            nn.Linear(self.output_head_sizes[0], self.output_head_sizes[1]),
            self.activation(),
            nn.Linear(self.output_head_sizes[1], self.output_dim),
        )


    def random_crop_grid(self, x, grid):
        delta = x.size(2) - grid.size(1)
        grid = grid.repeat(x.size(0), 1, 1, 1).cuda()
        # Add random shifts by x
        grid[:, :, :, 0] = grid[:, :, :, 0] + torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(
            -1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
        # Add random shifts by y
        grid[:, :, :, 1] = grid[:, :, :, 1] + torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(
            -1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
        return grid

    def build_grid(self, source_size, target_size):
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-k, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full.cuda()
    def forward(self, x):
        x_img = x['robot0_eye_in_hand_image'].to(self.device)
        x_img_goal = x['robot0_eye_in_hand_image_goal'].to(self.device)

        # assert x_img.shape[-3:] == (3, self.img_size, self.img_size) #元组而不是列表
        if not  x_img.shape[-3:] == (3, self.img_size, self.img_size):
            print(x_img.shape)
        if len(x_img.shape) == 3:
            b, seq = 1, 1
            x_img = x_img.view(1, 3, self.img_size, self.img_size)  # [b,c,h,w]
            x_img_goal = x_img_goal.view(1, 3, self.img_size, self.img_size)
        elif len(x_img.shape) == 4:
            assert self.batch_size >= x_img.shape[0]
            b, seq = x_img.shape[0], 1
        elif len(x_img.shape) == 5:
            assert self.batch_size >= x_img.shape[0] and self.seq_length == x_img.shape[1]
            b, seq = x_img.shape[0], x_img.shape[1]
        else:
            raise RuntimeError('x_img.shape should between 3 and 5')

        x_low_dim = torch.tensor([])
        assert len(self.low_dim_keys)==1
        for k in x:
            if k in self.low_dim_keys:
                x_low_dim = torch.cat((x_low_dim, x[k].view(b * seq, -1)), dim=-1).contiguous()  # [b*seq,total_len]
        x_low_dim = x_low_dim / 360
        x_low_dim=x_low_dim.to(self.device)

        x_img = x_img.view(b * seq, 3, self.img_size, self.img_size)
        if self.crop_size < self.img_size:
            x_img_grid_shifted = self.random_crop_grid(x_img, self.grid_source)
            x_img = F.grid_sample(x_img, x_img_grid_shifted, align_corners=True)
        if self.use_tcl_loss:
            x_img_aug=x_img.clone()
            for idx in range(x_img.shape[0]):
                x = random.randint(0, min(self.img_size,self.crop_size) - 1)
                y = random.randint(0, min(self.img_size,self.crop_size) - 1)
                x_img_aug[idx]=add_gaussian_spot_to_image(x_img_aug[idx], size=50,sigma=10, position=(x, y),to_device=True)
        else:
            if self.use_data_augmentation:
                for idx in range(x_img.shape[0]):
                    x = random.randint(0, min(self.img_size, self.crop_size) - 1)
                    y = random.randint(0, min(self.img_size, self.crop_size) - 1)
                    x_img[idx] = add_gaussian_spot_to_image(x_img[idx], size=50, sigma=10, position=(x, y),to_device=True)

        x_img = self.img_enc(x_img)
        if self.use_tcl_loss:
            x_img_feat=x_img.clone()
        x_img = self.spatial_softmax(x_img)
        x_img = self.ee_ln(x_img)
        x_img = x_img.view(b * seq, -1).contiguous()

        if self.use_tcl_loss:
            x_img_aug=self.img_enc(x_img_aug)
            if self.use_tcl_loss:
                x_img_aug_feat = x_img_aug.clone()
            x_img_aug = self.spatial_softmax(x_img_aug)
            x_img_aug = self.ee_ln(x_img_aug)
            x_img_aug = x_img_aug.view(b * seq, -1).contiguous()

        x_img_goal = x_img_goal.view(b * seq, 3, self.img_size, self.img_size)
        if self.crop_size < self.img_size:
            x_img_goal_grid_shifted = self.random_crop_grid(x_img_goal, self.grid_source)
            x_img_goal = F.grid_sample(x_img_goal, x_img_goal_grid_shifted, align_corners=True)
        if self.use_tcl_loss:
            x_img_goal_aug = x_img_goal.clone()
            for idx in range(x_img_goal.shape[0]):
                x = random.randint(0, min(self.img_size,self.crop_size) - 1)
                y = random.randint(0, min(self.img_size,self.crop_size) - 1)
                x_img_goal_aug[idx]=add_gaussian_spot_to_image(x_img_goal_aug[idx], size=50,sigma=10, position=(x, y),to_device=True)
        else:
            if self.use_data_augmentation:
                for idx in range(x_img.shape[0]):
                    x = random.randint(0, min(self.img_size, self.crop_size) - 1)
                    y = random.randint(0, min(self.img_size, self.crop_size) - 1)
                    x_img_goal[idx] = add_gaussian_spot_to_image(x_img_goal[idx], size=50, sigma=10, position=(x, y),to_device=True)

        if not self.use_siamese:
            x_img_goal = self.img_enc_goal(x_img_goal)
            if self.use_tcl_loss:
                x_img_goal_feat = x_img_goal.clone()
            x_img_goal = self.spatial_softmax_goal(x_img_goal)
            x_img_goal = self.ee_ln_goal(x_img_goal)
            x_img_goal = x_img_goal.view(b * seq, -1).contiguous()
            if self.use_tcl_loss:
                x_img_goal_aug = self.img_enc_goal(x_img_goal_aug)
                if self.use_tcl_loss:
                    x_img_goal_aug_feat = x_img_goal_aug.clone()
                x_img_goal_aug = self.spatial_softmax_goal(x_img_goal_aug)
                x_img_goal_aug = self.ee_ln_goal(x_img_goal_aug)
                x_img_goal_aug = x_img_goal_aug.view(b * seq, -1).contiguous()
        else:
            x_img_goal = self.img_enc(x_img_goal)
            if self.use_tcl_loss:
                x_img_goal_feat = x_img_goal.clone()
            x_img_goal = self.spatial_softmax(x_img_goal)
            x_img_goal = self.ee_ln(x_img_goal)
            x_img_goal = x_img_goal.view(b * seq, -1).contiguous()
            if self.use_tcl_loss:
                x_img_goal_aug = self.img_enc(x_img_goal_aug)
                if self.use_tcl_loss:
                    x_img_goal_aug_feat = x_img_goal_aug.clone()
                x_img_goal_aug = self.spatial_softmax(x_img_goal_aug)
                x_img_goal_aug = self.ee_ln(x_img_goal_aug)
                x_img_goal_aug = x_img_goal_aug.view(b * seq, -1).contiguous()

        x_low_dim = self.mlp_pos(x_low_dim)  # [b*seq,ldhs]
        input_tensor = torch.cat((x_img, x_img_goal, x_low_dim), dim=-1)  # [b*seq,hs]
        input_tensor=input_tensor.view(b , seq, -1).contiguous() # [b,seq,hs]

        if self.use_tcl_loss:
            input_tensor_aug = torch.cat((x_img_aug, x_img_goal_aug, x_low_dim), dim=-1)  # [b*seq,hs]
            input_tensor_aug = input_tensor_aug.view(b, seq, -1).contiguous()  # [b,seq,hs]

        N = seq
        output_tensor = None
        output_tensor_aug=None
        if self.is_training:
            for i in range(N):
                idx = input_tensor[:, :(i + 1), :]
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.gpt_model.block_size else idx[:, -self.gpt_model.block_size:]
                # forward the model to get the logits for the index in the sequence
                logits, loss = self.gpt_model(idx_cond)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1:, :]
                # print('idx.shape, logits.shape: ', idx.shape, logits.shape)
                if output_tensor == None:
                    output_tensor = logits.clone().contiguous()
                else:
                    output_tensor = torch.cat((output_tensor, logits), dim=1).contiguous()

                if self.use_tcl_loss:
                    idx_aug = input_tensor_aug[:, :(i + 1), :]
                    # if the sequence context is growing too long we must crop it at block_size
                    idx_cond_aug = idx_aug if idx_aug.size(1) <= self.gpt_model.block_size else idx_aug[:, -self.gpt_model.block_size:]
                    # forward the model to get the logits for the index in the sequence
                    logits_aug, loss_aug = self.gpt_model(idx_cond_aug)
                    # pluck the logits at the final step and scale by desired temperature
                    logits_aug = logits_aug[:, -1:, :]
                    # print('idx.shape, logits.shape: ', idx.shape, logits.shape)
                    if output_tensor_aug == None:
                        output_tensor_aug = logits_aug.clone().contiguous()
                    else:
                        output_tensor_aug = torch.cat((output_tensor_aug, logits_aug), dim=1).contiguous()
        else:
            self.buffer.append(input_tensor.clone())
            if len(self.buffer) > self.gpt_model.block_size:
                self.buffer = self.buffer[-self.gpt_model.block_size:]

            idx = torch.cat(self.buffer, dim=1).contiguous()
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.gpt_model.block_size else idx[:, -self.gpt_model.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, loss = self.gpt_model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1:, :]
            output_tensor = logits.contiguous()

        if not self.use_GMM:
            output_tensor = self.mlp_output_head(output_tensor)
            if self.use_tcl_loss:
                output_tensor_aug = self.mlp_output_head(output_tensor_aug)
            if self.output_activation is not None:
                output_tensor = self.output_activation(output_tensor)
                if self.use_tcl_loss:
                    output_tensor_aug = self.output_activation(output_tensor_aug)
        else:
            x_means = self.mlp_decoder_mean(output_tensor)
            x_scales = self.mlp_decoder_scale(output_tensor)
            x_logits = self.mlp_decoder_logits(output_tensor)

            x_means = x_means.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
            x_scales = x_scales.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
            x_logits = x_logits.view(b, seq, self.gmm_modes).contiguous()

            if self.low_noise_eval and seq == 1:
                # low-noise for all Gaussian dists
                x_scales = torch.ones_like(x_means) * 1e-4
            else:
                # post-process the scale accordingly
                x_scales = self.activations[self.std_activation](x_scales) + self.min_std

            component_distribution = D.Normal(loc=x_means, scale=x_scales)
            component_distribution = D.Independent(component_distribution, 1)  # shift action dim to event shape

            # unnormalized logits to categorical distribution for mixing the modes
            mixture_distribution = D.Categorical(logits=x_logits)

            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )
            output_tensor = dists.mean

            if self.use_tcl_loss:
                x_means_aug = self.mlp_decoder_mean(output_tensor_aug)
                x_scales_aug = self.mlp_decoder_scale(output_tensor_aug)
                x_logits_aug = self.mlp_decoder_logits(output_tensor_aug)

                x_means_aug = x_means_aug.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
                x_scales_aug = x_scales_aug.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
                x_logits_aug = x_logits_aug.view(b, seq, self.gmm_modes).contiguous()

                if self.low_noise_eval and seq == 1:
                    # low-noise for all Gaussian dists
                    x_scales_aug = torch.ones_like(x_means_aug) * 1e-4
                else:
                    # post-process the scale accordingly
                    x_scales_aug = self.activations[self.std_activation](x_scales_aug) + self.min_std

                component_distribution_aug = D.Normal(loc=x_means_aug, scale=x_scales_aug)
                component_distribution_aug = D.Independent(component_distribution_aug, 1)  # shift action dim to event shape

                # unnormalized logits to categorical distribution for mixing the modes
                mixture_distribution_aug = D.Categorical(logits=x_logits_aug)

                dists_aug = D.MixtureSameFamily(
                    mixture_distribution=mixture_distribution_aug,
                    component_distribution=component_distribution_aug,
                )
                output_tensor_aug = dists_aug.mean
        if not self.use_tcl_loss:
            return output_tensor
        else:
            return {"output_tensor": output_tensor, "output_tensor_aug": output_tensor_aug,"x_img_feat": x_img_feat,"x_img_goal_feat": x_img_goal_feat,"x_img_aug_feat": x_img_aug_feat,"x_img_goal_aug_feat": x_img_goal_aug_feat}