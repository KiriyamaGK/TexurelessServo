from ultralytics import YOLO
# # Build a YOLOv9c model from scratch
# model = YOLO("yolov9c.yaml")


if __name__ == '__main__':
    _is_custom = True
    data_yaml_pth = "/media/kiriyamagk/One Touch/AlignAnything_real/1021yolovn.v3i.yolov9/data.yaml"

    if _is_custom:
        model = YOLO('/home/kiriyamagk/桌面/AlignAnything/configs/new_modified_yolo.yaml')
        model.info()
        # results = model.train(data=data_yaml_pth,epochs=100, imgsz=640,cfg="/home/kiriyamagk/桌面/AlignAnything/configs/industrial_train.yaml")
        results = model.train(data=data_yaml_pth, batch=16,epochs=150, imgsz=640)
    else:

        model = YOLO("yolov9s.pt")
        model.info()
        # results = model.train(data="/media/kiriyamagk/One Touch/AlignAnything_real/25.06.23/hdf5/augmentation/GK_NEW.v3i.yolov9/data.yaml", epochs=100, imgsz=640)
        results = model.train(data=data_yaml_pth, epochs=100, imgsz=640)