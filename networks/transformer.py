import collections
import random
import warnings

import torch
import math
import numpy as np
import torch.nn as nn
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

        #initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.freeze_encoder=encoder['freeze']
        self.encoder_name = encoder['name']
        self.use_siamese = encoder['siamese']
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.ss_num_kp = encoder['params']['SpatialSoftmax']['num_kp']
        self.ss_in_c = encoder['params']['SpatialSoftmax']['in_c']
        self.ss_in_h = encoder['params']['SpatialSoftmax']['in_h']
        self.ss_in_w = encoder['params']['SpatialSoftmax']['in_w']


        self.input_low_dim = input_low_dim
        self.output_dim = output_dim
        self.img_size = encoder['params']['img_size']
        self.crop_size = encoder['params']['crop_size']

        self.obs_keys = obs_keys
        self.low_dim_keys = []
        for obs_key in obs_keys:
            if "image" not in obs_key and "img" not in obs_key:
                self.low_dim_keys.append(obs_key)

        self.low_dim_hidden_sizes = low_dim_hidden_sizes if self.input_low_dim != 0 else [0]

        self.feat_dim = embedding_size
        self.output_head_sizes = output_head_sizes
        self.use_GMM = use_gmm
        self.is_training = training

        self.num_cameras = encoder['num_cameras'] if "num_cameras" in encoder else 1
        if self.is_training:
            self.use_data_augmentation = encoder['data_augmentation']
            self.use_tcl_loss = encoder['task_consistency_loss']
            self.create_mixed_light_dataset = encoder['mixed_light_dataset'] if "mixed_light_dataset" in encoder else False
        else:
            self.use_data_augmentation = False
            self.use_tcl_loss = False
            self.create_mixed_light_dataset = False

        if self.use_tcl_loss:
            assert self.use_data_augmentation

        self.activation = get_activation_fn(activation)
        self.output_activation = get_activation_fn(output_activation) if output_activation is not None else None

        self.model_config = GPT.get_default_config()
        self.model_config.vocab_size = embedding_size
        self.model_config.n_embd = embedding_size
        self.model_config.n_layer = n_layer
        self.model_config.n_head = n_head
        self.model_config.block_size = block_size

        #img
        assert self.encoder_name in ['ResNet18','YOLO_v11']
        self.grid_source = self.build_grid(self.crop_size, self.crop_size)

        self.img_encs = nn.ModuleList()
        self.img_enc_goals = nn.ModuleList()
        self.spatial_softmaxs = nn.ModuleList()
        self.spatial_softmax_goals = nn.ModuleList()
        self.ee_lns = nn.ModuleList()
        self.ee_ln_goals = nn.ModuleList()

        if self.encoder_name == 'ResNet18':
            for _ in range(self.num_cameras):
                resnet18 = models.resnet18()
                resnet18_goal = models.resnet18()
                self.img_encs.append(torch.nn.Sequential(*(list(resnet18.children())[:-2])))
                self.img_enc_goals.append(torch.nn.Sequential(*(list(resnet18_goal.children())[:-2])))
                self.spatial_softmaxs.append(SpatialSoftmax(self.ss_in_c,self.ss_in_h,self.ss_in_w,self.ss_num_kp))
                self.spatial_softmax_goals.append(SpatialSoftmax(self.ss_in_c, self.ss_in_h, self.ss_in_w, self.ss_num_kp))
                self.ee_lns.append(nn.Linear(self.ss_num_kp * 2, (self.feat_dim-self.low_dim_hidden_sizes[-1])//(2*self.num_cameras)))
                self.ee_ln_goals.append(nn.Linear(self.ss_num_kp * 2, (self.feat_dim-self.low_dim_hidden_sizes[-1])//(2*self.num_cameras)))

        if self.freeze_encoder:
            for img_enc in self.img_encs:
                for param in img_enc.parameters():
                    param.requires_grad = False
            for img_enc in self.img_enc_goals:
                for param in img_enc.parameters():
                    param.requires_grad = False

        #for low_dim
        if self.input_low_dim != 0:
            self.mlp_pos = nn.Sequential(
                nn.Linear(self.input_low_dim, self.low_dim_hidden_sizes[0]),
                self.activation(),
                nn.Linear(self.low_dim_hidden_sizes[0], self.low_dim_hidden_sizes[1]),
                self.activation(),
                nn.Linear(self.low_dim_hidden_sizes[1], self.low_dim_hidden_sizes[2]),
            )
        #policy
        self.gpt_model = GPT(self.model_config)
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

        self.print_training_settings()

    def gmm_output_head(self,x,b,seq):
        x_means = self.mlp_decoder_mean(x)
        x_scales = self.mlp_decoder_scale(x)
        x_logits = self.mlp_decoder_logits(x)

        x_means = x_means.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
        x_scales = x_scales.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
        x_logits = x_logits.view(b, seq, self.gmm_modes).contiguous()
        return x_means, x_scales, x_logits

    def forward(self, x):
        # determine b,seq
        b,seq = self.determine_batch_and_seq_len(x['robot0_eye_in_hand_image'].shape)

        # change dimension
        for itms in x.keys():
            if "image" in itms or "img" in itms:
                x[itms] = x[itms].view(-1, 3, self.img_size, self.img_size)

        # low_dim
        if self.input_low_dim != 0:
            x_low_dim = torch.tensor([])
            assert len(self.low_dim_keys) == 1
            for k in x:
                if k in self.low_dim_keys:
                    x_low_dim = torch.cat((x_low_dim, x[k].view(b * seq, -1)),
                                          dim=-1).contiguous()  # [b*seq,total_len]
            x_low_dim = x_low_dim / 360
            x_low_dim = x_low_dim.to(self.device)
            x_low_dim = self.mlp_pos(x_low_dim)  # [b*seq,ldhs]
        else:
            x_low_dim = None

        # imgs
        x = self.create_util_img_tensors(x)
        x = self.preprocess_imgs(x)
        x = self.img_branch(x, b=b, seq=seq)  # sequentially:  id + "goal" + "aug" + "feat"

        # policy
        plc, plc_aug, x = self.determine_policy_inputs(x, x_low_dim)

        plc = plc.view(b, seq, -1).contiguous()
        if self.use_tcl_loss:
            plc_aug = plc_aug.view(b, seq, -1).contiguous()

        N = seq
        output = None
        output_aug=None

        if self.is_training:
            for i in range(N):
                idx = plc[:, :(i + 1), :]
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.gpt_model.block_size else idx[:, -self.gpt_model.block_size:]
                # forward the model to get the logits for the index in the sequence
                logits, loss = self.gpt_model(idx_cond)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1:, :]
                # print('idx.shape, logits.shape: ', idx.shape, logits.shape)
                if output == None:
                    output = logits.clone().contiguous()
                else:
                    output = torch.cat((output, logits), dim=1).contiguous()

                if self.use_tcl_loss:
                    idx_aug = plc_aug[:, :(i + 1), :]
                    # if the sequence context is growing too long we must crop it at block_size
                    idx_cond_aug = idx_aug if idx_aug.size(1) <= self.gpt_model.block_size else idx_aug[:, -self.gpt_model.block_size:]
                    # forward the model to get the logits for the index in the sequence
                    logits_aug, loss_aug = self.gpt_model(idx_cond_aug)
                    # pluck the logits at the final step and scale by desired temperature
                    logits_aug = logits_aug[:, -1:, :]
                    # print('idx.shape, logits.shape: ', idx.shape, logits.shape)
                    if output_aug == None:
                        output_aug = logits_aug.clone().contiguous()
                    else:
                        output_aug = torch.cat((output_aug, logits_aug), dim=1).contiguous()
        else:
            self.buffer.append(plc.clone())
            if len(self.buffer) > self.gpt_model.block_size:
                self.buffer = self.buffer[-self.gpt_model.block_size:]

            idx = torch.cat(self.buffer, dim=1).contiguous()
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.gpt_model.block_size else idx[:, -self.gpt_model.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, loss = self.gpt_model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1:, :]
            output = logits.contiguous()

        if not self.use_GMM:
            output = self.mlp_output_head(output)
            if self.output_activation is not None:
                output = self.output_activation(output)

            if self.use_tcl_loss:
                output_aug = self.mlp_output_head(output_aug)
                if self.output_activation is not None:
                    output_aug = self.output_activation(output_aug)

        else:
            x_means, x_scales, x_logits = self.gmm_output_head(x=output, b=b, seq=seq)
            dists = self.create_mixed_distribution(x_means, x_scales, x_logits, seq)
            output = dists.mean

            if self.use_tcl_loss:
                x_means_aug, x_scales_aug, x_logits_aug = self.gmm_output_head(x=output_aug, b=b, seq=seq)
                dists_aug = self.create_mixed_distribution(x_means_aug, x_scales_aug, x_logits_aug, seq)
                output_aug = dists_aug.mean

        if not self.use_tcl_loss:
            return output
        else:
            rtn_dict = {"output_tensor": output, "output_tensor_aug": output_aug, "x_img_feat": x["x_0_feat"],
                        "x_img_goal_feat": x["x_0_goal_feat"], "x_img_aug_feat": x["x_0_aug_feat"],
                        "x_img_goal_aug_feat": x["x_0_goal_aug_feat"]}

            if self.num_cameras == 2:
                rtn_dict["x_img_feat"] = torch.cat((rtn_dict["x_img_feat"], x["x_1_feat"]), dim=-1)
                rtn_dict["x_img_aug_feat"] = torch.cat((rtn_dict["x_img_aug_feat"], x["x_1_aug_feat"]), dim=-1)
                rtn_dict["x_img_goal_feat"] = torch.cat((rtn_dict["x_img_goal_feat"], x["x_1_goal_feat"]), dim=-1)
                rtn_dict["x_img_goal_aug_feat"] = torch.cat((rtn_dict["x_img_goal_aug_feat"], x["x_1_goal_aug_feat"]),
                                                            dim=-1)
            return rtn_dict