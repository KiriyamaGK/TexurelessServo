from ultralytics import YOLO

# 1. 加载训练好的模型
model = YOLO('/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train6(yolov8)/weights/best.pt')  # 替换为你的模型路径
data_yaml_pth = "/media/kiriyamagk/One Touch/AlignAnything_real/yolovn.v1i.yolov9/data.yaml"
# 2. 在验证集上进行评估
#    key参数说明：
#    data: 数据配置文件的路径（如 data.yaml）
#    split: 使用哪个数据集进行评估，通常是 'val'
#    imgsz: 评估时使用的图像尺寸
#    conf: 置信度阈值
#    iou: NMS 的 IoU 阈值
#    device: 指定GPU (如 '0') 或 CPU (如 'cpu')
results = model.val(
    data=data_yaml_pth,
    split='val',        # 或者 'test'
    imgsz=640,
    conf=0.001,         # 低置信度以生成完整的 PR 曲线
    iou=0.6,
    device=0
)

# 打印关键结果
print(f"mAP50: {results.box.map50}")
print(f"mAP50-95: {results.box.map}")
print(f"Precision: {results.box.mp}")
print(f"Recall: {results.box.mr}")

# 结果保存在 runs/detect/val 目录下
# 你可以找到：
# - 混淆矩阵 (confusion_matrix.png)
# - F1-置信度曲线 (F1_curve.png)
# - 精确率-召回率曲线 (PR_curve.png)
# - 等等...