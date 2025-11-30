import cv2
import os


def extract_frames_opencv(video_path, output_dir, interval=1):
    """
    从视频中提取帧
    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param interval: 每隔多少帧提取一张图片
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 按间隔保存帧
        if frame_count % interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"已保存第 {saved_count} 帧")

        frame_count += 1

    cap.release()
    print(f"提取完成！总共处理了 {frame_count} 帧，保存了 {saved_count} 张图片")


if __name__ == "__main__":
    # 使用示例
    video_base = "/home/kiriyamagk/桌面/paper_imgs/fine-positioning"
    output_base = video_base + "/imgs"

    video_name = "video3.mp4"
    output_dir = "wrist_task3"

    extract_frames_opencv(video_base + "/" + video_name, output_base + "/" + output_dir, interval=10)