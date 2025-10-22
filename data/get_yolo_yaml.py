from ultralytics import YOLO
import yaml

# 方法3.1: 通过创建模型并获取配置
model = YOLO('yolov9s.yaml')  # 这会从ultralytics/cfg/models/v9下载
config = model.model.yaml

# 保存到文件
with open('../configs/yolov9s_official.yaml', 'w') as f:
    yaml.dump(config, f)

print("YOLOv9s配置已保存到 yolov9s_official.yaml")