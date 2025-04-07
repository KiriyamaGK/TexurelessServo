from abc import ABC
import torch
from torch import nn
import random
from utils.input_process import add_gaussian_spot_to_image
from torch.nn import functional as F
import torch.distributions as D
import warnings


class NetworkBase(ABC, nn.Module):
    def __init__(self,input_low_dim, output_dim):
        super().__init__()

        self.input_dim = input_low_dim
        self.output_dim = output_dim

    def random_crop_grid(self, x, grid):
        delta = x.size(2) - grid.size(1)
        grid = grid.repeat(x.size(0), 1, 1, 1).cuda()
        if self.training:
            # Add random shifts by x
            grid[:, :, :, 0] = grid[:, :, :, 0] + torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(
                -1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
            # Add random shifts by y
            grid[:, :, :, 1] = grid[:, :, :, 1] + torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(
                -1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
        else:
            center_offset = delta // 2
            grid[:, :, :, 0] = grid[:, :, :, 0] + center_offset / x.size(2)
            grid[:, :, :, 1] = grid[:, :, :, 1] + center_offset / x.size(2)
        return grid

    def build_grid(self, source_size, target_size):
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-k, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full.cuda()

    def crop_img(self,x):
        if self.crop_size < self.img_size:
            x_grid_shifted = self.random_crop_grid(x, self.grid_source)
            x = F.grid_sample(x, x_grid_shifted, align_corners=True)
        return x

    def gaussian_augmentation(self, x):
        for idx in range(x.shape[0]):
            x0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
            y0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
            x[idx] = add_gaussian_spot_to_image(x[idx], size=50, sigma=10, position=(x0, y0),
                                                    to_device=True)
        return x

    def return_name_and_type_from_key(self,key:str):
        assert "image" in key
        if "_2" in key:
            start,is_cam2 = "_1",True
        else:
            start,is_cam2 = "_0",False

        if "_goal" in key:
            mid, is_goal = "_goal", True
        else:
            mid,is_goal = "",False

        if "light" in key:
            end,is_aug = "_aug",True
        else:
            end,is_aug = "",False

        return {"name":"x"+start+mid+end,"is_cam2":is_cam2,"is_goal":is_goal,"is_aug":is_aug}

    def determine_batch_and_seq_len(self,shape):
        assert shape[-3:] == (3, self.img_size, self.img_size)
        if len(shape) == 3:
            b, seq = 1, 1
        elif len(shape) == 4:
            assert self.batch_size >= shape[0]
            b, seq = shape[0], 1
        elif len(shape) == 5:
            assert self.batch_size >= shape[0] and self.seq_length == shape[1]
            b, seq = shape[0], shape[1]
        else:
            raise RuntimeError('x_img.shape should between 3 and 5')
        return b, seq

    def merging_depth(self,x):
        if not self.using_depth:
            return x
        else:
            assert "depth_image" in x
            img_1_keys = [k for k in x.keys() if "depth" not in k and "_2" not in k and ("image" in k or "img" in k)]
            dep_1_key="depth_image"
            dep_1_goal_key = "depth_image_goal"
            for k in img_1_keys:
                x[k]=torch.concatenate((x[k],x[dep_1_key].clone()), dim=1) if "goal" not in k else torch.concatenate(
                    (x[k],x[dep_1_goal_key].clone()), dim=1) #[_,c,h,w]
            del x[dep_1_key]
            del x[dep_1_goal_key]


            if self.num_cameras==2:
                img_2_keys = [k for k in x.keys() if "depth" not in k and "_2" in k and ("image" in k or "img" in k)]
                dep_2_key = "depth_image_2"
                dep_2_goal_key = "depth_image_2_goal"
                for k in img_2_keys:
                    x[k] = torch.concatenate((x[k], x[dep_2_key].clone()), dim=1) if "goal" not in k else torch.concatenate(
                        (x[k], x[dep_2_goal_key].clone()), dim=1)  # [_,c,h,w]
                del x[dep_2_key]
                del x[dep_2_goal_key]
            return x

    def create_util_img_tensors(self,x):
        dic = {}
        for k in list(x.keys()): #迭代字典的键时不能修改字典的键，所以把x.keys（）换成list(x.keys()）
            if "img" not in k and "image" not in k:
                del x[k]  #only process imgs
                continue
            attr = self.return_name_and_type_from_key(k)
            dic[attr["name"]] = x[k]

        if self.create_mixed_light_dataset:  #仅代表tcl_raw，慎选
            assert any("_aug" in k for k in dic.keys())
            for k in list(dic.keys()):
                if "_aug" in k:
                    continue
                else:
                    if self.use_tcl_loss:  # tcl_raw,image和image_light在除了高斯亮斑的因素外，其他方面一致
                        for idx in range(dic[k].shape[0]): #(b*seq,3,sz,sz)
                            if random.randint(0, 1) > 0:
                                dic[k+"_aug"][idx] = dic[k][idx] #张量切片赋值不会影响原始张量的内容
                            else:
                                dic[k][idx] = dic[k + "_aug"][idx]
                    else:
                        for idx in range(dic[k].shape[0]):
                            if random.randint(0, 1) > 0:
                                dic[k][idx] = dic[k + "_aug"][idx]

        if self.use_tcl_loss and all("_aug" not in k for k in dic.keys()):  # tcl_raw
            warnings.warn("robot0_eye_in_hand_image_light not in dataset, adding...", UserWarning)
            for k in list(dic.keys()):
                dic[k+"_aug"] = dic[k].clone()

        keys_to_delete = [k for k in dic.keys() if (not self.use_tcl_loss) and "_aug" in k]
        for k in keys_to_delete:
            del dic[k]

        return dic

    def print_training_settings(self):
        print("=====================================training setting=======================================")
        print("crop: ",bool(self.crop_size<self.img_size))
        if self.use_tcl_loss:
            if all("light" not in k for k in self.obs_keys):
                print("tcl_option: tcl_raw ---- no_mixed_light")
            elif self.create_mixed_light_dataset:
                print("tcl_option: tcl_raw ---- mixed_light")
            else:
                print("tcl_option: tcl")
        else:
            print("tcl_option: no_tcl")
        print("gaussian_augmentation: ",bool(self.use_data_augmentation))
        print("=====================================training setting=======================================")

    def preprocess_imgs(self, x):
        for k in list(x.keys()):
            x[k]=x[k].to(self.device)
            x[k]=self.crop_img(x[k])
            if (self.use_tcl_loss and "_aug" in k) or (not self.use_tcl_loss and self.use_data_augmentation):
                x[k]=self.gaussian_augmentation(x[k])
        return x

    def post_image_encoder(self,x:torch.Tensor,b:int,seq:int,cam_index:int,is_goal:bool):
        '''
        spatial_softmax,linear,and,reshape
        '''
        x = self.spatial_softmaxs[cam_index](x) if not is_goal else self.spatial_softmax_goals[cam_index](x)
        x = self.ee_lns[cam_index](x) if not is_goal else self.ee_ln_goals[cam_index](x)
        x = x.view(b * seq, -1).contiguous()
        return x

    def img_branch(self,x,b,seq):
        for k in list(x.keys()):    # 迭代这个列表,否则for k in x.keys()会报错
            cam_id=0 if "_0" in k else 1
            is_goal=True if "goal" in k else False
            x[k]=self.img_encs[cam_id](x[k]) if not is_goal else self.img_enc_goals[cam_id](x[k])
            if self.use_tcl_loss:
                x[k+"_feat"]=x[k].clone()
            x[k]=self.post_image_encoder(x=x[k], b=b, seq=seq, cam_index=cam_id, is_goal=is_goal)
        return x

    def determine_policy_inputs(self,img_dict:dict,x_low_dim):
        plc_aug = None

        plc=torch.cat((img_dict["x_0"],img_dict["x_0_goal"],x_low_dim), dim=-1).contiguous() if x_low_dim is not None \
            else torch.cat((img_dict["x_0"],img_dict["x_0_goal"]), dim=-1).contiguous() #torch.cat 本身不会修改原始张量，它会创建一个新的张量
        if self.num_cameras==2:
            plc=torch.cat((plc,img_dict["x_1"],img_dict["x_1_goal"]), dim=-1).contiguous()

        if self.use_tcl_loss:
            plc_aug=torch.cat((img_dict["x_0_aug"],img_dict["x_0_goal_aug"],x_low_dim), dim=-1).contiguous() if x_low_dim is not None \
                else torch.cat((img_dict["x_0_aug"],img_dict["x_0_goal_aug"]), dim=-1).contiguous()
            if self.num_cameras == 2:
                plc_aug = torch.cat((plc_aug, img_dict["x_1_aug"], img_dict["x_1_goal_aug"]), dim=-1).contiguous()

        return plc,plc_aug,img_dict

    def create_mixed_distribution(self,x_means,x_scales,x_logits,seq):
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
        return dists