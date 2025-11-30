import cv2
import matplotlib.pyplot as plt
import torch
import time
import json
from utils.paths import path_completion,PROJECT_ROOT_DIR
from experiments.rollout_real import _setup_model
from utils.input_process import input_dict_preprocess

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_module = dict([*self.model.named_modules()])[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_dict, target_class=None):
        self.model.zero_grad()
        # 前向传播 - 传入字典
        output = self.model(input_dict)  # 不是 input_tensor

        # 获取输出张量
        if isinstance(output, dict):
            output_tensor = output["output_tensor"]  # 根据你的网络输出结构调整
        else:
            output_tensor = output

        if target_class is None:
            target_class = output_tensor.argmax(dim=1)

        # 反向传播
        one_hot = torch.zeros_like(output_tensor)
        one_hot[0][target_class] = 1
        output_tensor.backward(gradient=one_hot)

        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU激活

        # 归一化
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.detach().squeeze().cpu().numpy()


def visualize_gradcam(model, input_dict, target_layer):
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_dict)  # 传入字典

    # 从输入字典中提取图像用于可视化
    num = target_layer.split(".")[1][0]
    assert num in ["0","1"]

    image_tensor = input_dict['robot0_eye_in_hand_image'][0]  if num == "0" else input_dict['robot0_eye_in_hand_image_2'][0]

    # 叠加原始图像和热力图
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')  # cam 应该是2D热力图
    plt.title('Grad-CAM')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_recommended_target_layers(mlp_model):
    """获取推荐的可视化目标层"""

    target_layers = []

    # 1. 图像编码器的不同深度层
    for cam_idx in range(mlp_model.num_cameras):
        # ResNet的卷积块
        target_layers.extend([
            # f'img_encs.{cam_idx}.0',  # 初始卷积层 - 看基础特征提取
            # f'img_encs.{cam_idx}.3',  # layer1 - 低级特征
            # f'img_encs.{cam_idx}.4',  # layer2 - 中级特征
            f'img_encs.{cam_idx}.7',  # layer4 - 高级语义特征
        ])

        # 目标图像的编码器
        target_layers.extend([
            # f'img_enc_goals.{cam_idx}.0',
            f'img_enc_goals.{cam_idx}.7',
        ])

    # # 2. 空间软最大值层 - 可视化注意力关键点
    # for cam_idx in range(mlp_model.num_cameras):
    #     target_layers.append(f'spatial_softmaxs.{cam_idx}')
    #     target_layers.append(f'spatial_softmax_goals.{cam_idx}')
    #
    # # 3. 末端连接层 - 特征融合点
    # for cam_idx in range(mlp_model.num_cameras):
    #     target_layers.append(f'ee_lns.{cam_idx}')
    #     target_layers.append(f'ee_ln_goals.{cam_idx}')
    #
    # # 4. MLP层
    # if mlp_model.input_low_dim != 0:
    #     target_layers.append('mlp_pos.0')  # 低维MLP的第一层
    #     target_layers.append('mlp_pos.-1')  # 低维MLP的最后一层
    #
    # if not mlp_model.use_GMM:
    #     target_layers.append('policy_mlp.0')  # 策略MLP的第一层
    #     target_layers.append('policy_mlp.-1')  # 策略MLP的最后一层
    # else:
    #     target_layers.extend([
    #         'mlp_decoder_mean.0',
    #         'mlp_decoder_scale.0',
    #         'mlp_decoder_logits.0'
    #     ])
    #
    # # 5. 位姿估计层（如果使用）
    # if mlp_model.using_pos_estm:
    #     target_layers.extend([
    #         'pos_estm_layer.0',
    #         'pos_estm_bottleneck.0'
    #     ])

    return target_layers

if __name__ == '__main__':
    config_dir = "/home/kiriyamagk/桌面/AlignAnything/configs/rollout_real.json"
    ckpts_dir = "/home/kiriyamagk/桌面/AlignAnything/trained_models/trial/2025-11-08_00-00-00/dagger_episode_199_epoch_5_loss_0.0006.pth"
    # ckpts_dir = None
    img_1_pth = "/home/kiriyamagk/桌面/imgs_to_pred/1763981677_1.png"
    img_2_pth = "/home/kiriyamagk/桌面/imgs_to_pred/1763981677_2.png"
    bgr2rgb = False


    #setup model
    with open(config_dir, "r") as j:
        config = json.load(j)
    model_config_dir=path_completion(config["logs_dir"],PROJECT_ROOT_DIR)
    with open(model_config_dir, "r") as j:
        model_config = json.load(j)
    model = _setup_model(model_config)

    if ckpts_dir is not None:
        state_dict = torch.load(ckpts_dir, weights_only=False)
        model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    #prepare inputs
    img_1 = cv2.imread(img_1_pth)
    img_2 = cv2.imread(img_2_pth)
    if bgr2rgb:
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    sample_input = {
        "robot0_eye_in_hand_image": img_1,
        "robot0_eye_in_hand_image_goal": img_1,
        "robot0_eye_in_hand_image_2": img_2,
        "robot0_eye_in_hand_image_2_goal": img_2,
    }
    sample_input = input_dict_preprocess(sample_input, rollout=True)

    recommended_layers = get_recommended_target_layers(model)
    print("推荐的可视化层:")
    for i, layer in enumerate(recommended_layers):
        print(f"{i + 1}. {layer}")

    # 特征图可视化示例
    for layer_name in recommended_layers[:]:  # 先看前4个重要层
        try:
            real_input = layer_name.split(".")
            visualize_gradcam(model, sample_input, layer_name)
            print(f"成功可视化层: {layer_name}")
        except Exception as e:
            print(f"无法可视化层 {layer_name}: {e}")

    # Grad-CAM注意力可视化 - 选择高级语义层
    # gradcam_layers = [
    #     'img_encs.0.7',  # 最后一个残差块 - 最抽象的特征
    #     'img_encs.0.4',  # 中间层 - 平衡抽象和细节
    # ]

    # for target_layer in gradcam_layers:
    #     try:
    #         visualize_gradcam(model, sample_input, target_layer)
    #     except Exception as e:
    #         print(f"Grad-CAM失败 {target_layer}: {e}")

