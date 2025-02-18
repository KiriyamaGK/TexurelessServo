from torch.nn import Module
from typing import Callable
from torch.nn.modules import activation
from torch.nn.modules import loss
import torch.optim as optim
from networks.Network import NetworkBase
from networks.losses import CustomLoss,ModifiedFocalLoss,MSE_and_ModifiedFocalLoss,TCL_MSE

Activation = Callable[..., Module]
def get_activation_fn(act: str) -> Activation:
    # get list from activation submodule as lower-case
    activations_lc = [str(a).lower() for a in activation.__all__]
    if (act := str(act).lower()) in activations_lc:
        # match actual name from lower-case list, return function/factory
        idx = activations_lc.index(act)
        act_name = activation.__all__[idx]
        act_func = getattr(activation, act_name)
        return act_func
    else:
        raise ValueError(f"Cannot find activation funct ion for string <{act}>")

def get_loss_fn(lss: str,weight:dict,seq_length:int,output_dim:int) -> Callable[..., Module]:
    # get list from loss submodule as lower-case
    losses_lc = [str(l).lower() for l in loss.__all__]
    if (lss := str(lss).lower()) in losses_lc:
        # match actual name from lower-case list, return function/factory
        idx = losses_lc.index(lss)
        loss_name = loss.__all__[idx]
        loss_func = getattr(loss, loss_name)
        return loss_func
    elif lss =='customloss':
        return CustomLoss(weight,seq_length,output_dim)
    elif lss =="focal":
        return ModifiedFocalLoss()
    elif lss == "focal_and_mse":
        return MSE_and_ModifiedFocalLoss(weight,seq_length,output_dim)
    elif lss=="tcl_loss":
        return TCL_MSE(weight,seq_length,output_dim)
    else:
        raise ValueError(f"Cannot find loss function for string <{lss}>")

def get_optimizer_cls(opt: str) -> Callable[..., optim.Optimizer]:
    # get list from optim submodule as lower-case
    # optimizers_lc = [str(o).lower() for o in optim.__all__]
    optimizers = [o for o in dir(optim) if not o.startswith('_')]
    optimizers_lc = [str(o).lower() for o in optimizers]
    if (opt := str(opt).lower()) in optimizers_lc:
        # match actual name from lower-case list, return function/factory
        idx = optimizers_lc.index(opt)
        opt_name = optimizers[idx]
        opt_class = getattr(optim, opt_name)
        return opt_class
    else:
        raise ValueError(f"Cannot find optimizer function for string <{opt}>")

def get_network_cls(net: str)->(NetworkBase,bool):
    from networks.mlp import MLP
    from networks.transformer import Transformer
    from networks.transformer_single import TransformerSingle
    # from outdated_codes.CenterNet2 import CenterNet_ResNet18
    # from outdated_codes.transformer_centernet import TransformerCenternet
    net_dct = {
        "mlp": MLP,
        "transformer": Transformer,
        "transformersingle": TransformerSingle,
        # "centernet": CenterNet_ResNet18,
        # "transformercenternet": TransformerCenternet,
    }

    if (net := str(net).lower()) in net_dct:
        if (net := str(net).lower())!='centernet':
            return net_dct[net], True
        else:
            return net_dct[net], False
    raise ValueError(f"Cannot find network class for string <{net}>")
