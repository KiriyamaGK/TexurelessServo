import h5py
import numpy as np
import cv2
import os

if __name__ == '__main__':
    hdf_pth='/media/kiriyamagk/One Touch/AlignAnything/25.01.21/hdf5/mimic.hdf5'

    # temp_dir='/media/kiriyamagk/One Touch/AlignAnything/25.01.17/hdf5/temp'
    hdf_base=os.path.dirname(hdf_pth)
    fps=30
    vis_h,vis_w=480,480
    only_rot_caption=False
    img_wrist_vis=True
    img_right_vis=False
    img_left_vis=False
    img_wrist_goal_vis=False

    hdf = h5py.File(hdf_pth, 'r')
    mp4=cv2.VideoWriter_fourcc(*'mp4v')
    if img_wrist_vis:
        print("processing wrist video....")
        video_path = os.path.join(hdf_base, 'wrist_video.mp4')
        out = cv2.VideoWriter(video_path, mp4, fps, (vis_w, vis_h))
        for i in range(len(hdf['data'])):
            print("processing demo_{}".format(i))
            demo_caption = 'demo_{}'.format(i)
            # os.makedirs(os.path.join(temp_dir,demo_caption), exist_ok=True)
            for j in range(len(hdf['data/demo_{}/actions'.format(i)])):
                image=hdf['data/demo_{}/obs/robot0_eye_in_hand_image'.format(i)][j]
                raw_size=image.shape[0]
                image=image.astype('uint8')
                image=cv2.resize(image, (vis_w, vis_h))
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.imwrite(os.path.join(temp_dir,demo_caption,demo_caption +'_{}'.format(j)+ '.jpg'), image)
                if only_rot_caption:
                    action=hdf['data/demo_{}/actions'.format(i)][j,2]
                else:
                    action=hdf['data/demo_{}/actions'.format(i)][j]
                action_caption='action: {}'.format(action)
                # cv2.putText(image, action_caption, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                action_caption_width = cv2.getTextSize(action_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                action_caption_x = (vis_w - action_caption_width) // 2

                cv2.putText(image, action_caption, (action_caption_x,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
                cv2.putText(image, demo_caption, (vis_w - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                raw_size_caption = 'img_size: {}'.format(raw_size)
                cv2.putText(image, raw_size_caption, (10, vis_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out.write(image)
        out.release()

    if img_wrist_goal_vis:
        print("processing wrist goal video....")
        video_path = os.path.join(hdf_base, 'wrist_goal_video.mp4')
        out = cv2.VideoWriter(video_path, mp4, fps, (vis_w, vis_h))
        for i in range(len(hdf['data'])):
            print("processing demo_{}".format(i))
            demo_caption = 'demo_{}'.format(i)
            for j in range(len(hdf['data/demo_{}/actions'.format(i)])):
                image=hdf['data/demo_{}/obs/robot0_eye_in_hand_image_goal'.format(i)][j]
                raw_size = image.shape[0]
                image = image.astype('uint8')
                image = cv2.resize(image, (vis_w, vis_h))
                if only_rot_caption:
                    action=hdf['data/demo_{}/actions'.format(i)][j,5]
                else:
                    action=hdf['data/demo_{}/actions'.format(i)][j]
                action_caption='action: {}'.format(action)
                action_caption_width = cv2.getTextSize(action_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                action_caption_x = (vis_w - action_caption_width) // 2

                cv2.putText(image, action_caption, (action_caption_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
                cv2.putText(image, demo_caption, (vis_w - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                raw_size_caption = 'img_size: {}'.format(raw_size)
                cv2.putText(image, raw_size_caption, (10, vis_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out.write(image)
        out.release()

    if img_left_vis:
        print("processing left video....")
        video_path = os.path.join(hdf_base, 'left_video.mp4')
        out = cv2.VideoWriter(video_path, mp4, fps, (vis_w, vis_h))
        for i in range(len(hdf['data'])):
            print("processing demo_{}".format(i))
            demo_caption = 'demo_{}'.format(i)
            for j in range(len(hdf['data/demo_{}/actions'.format(i)])):
                image=hdf['data/demo_{}/obs/agentview_image_2'.format(i)][j]
                raw_size = image.shape[0]
                image = image.astype('uint8')
                image = cv2.resize(image, (vis_w, vis_h))
                if only_rot_caption:
                    action=hdf['data/demo_{}/actions'.format(i)][j,5]
                else:
                    action=hdf['data/demo_{}/actions'.format(i)][j]
                action_caption='action: {}'.format(action)
                action_caption_width = cv2.getTextSize(action_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                action_caption_x = (vis_w - action_caption_width) // 2

                cv2.putText(image, action_caption, (action_caption_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
                cv2.putText(image, demo_caption, (vis_w - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                raw_size_caption = 'img_size: {}'.format(raw_size)
                cv2.putText(image, raw_size_caption, (10, vis_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out.write(image)
        out.release()

    if img_right_vis:
        print("processing right video....")
        video_path = os.path.join(hdf_base, 'right_video.mp4')
        out = cv2.VideoWriter(video_path, mp4, fps, (vis_w, vis_h))
        for i in range(len(hdf['data'])):
            print("processing demo_{}".format(i))
            demo_caption = 'demo_{}'.format(i)
            for j in range(len(hdf['data/demo_{}/actions'.format(i)])):
                image = hdf['data/demo_{}/obs/agentview_image'.format(i)][j]
                raw_size = image.shape[0]
                image = image.astype('uint8')
                image = cv2.resize(image, (vis_w, vis_h))
                if only_rot_caption:
                    action = hdf['data/demo_{}/actions'.format(i)][j, 5]
                else:
                    action = hdf['data/demo_{}/actions'.format(i)][j]
                action_caption = 'action: {}'.format(action)
                action_caption_width = cv2.getTextSize(action_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                action_caption_x = (vis_w - action_caption_width) // 2

                cv2.putText(image, action_caption, (action_caption_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
                cv2.putText(image, demo_caption, (vis_w - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                raw_size_caption = 'img_size: {}'.format(raw_size)
                cv2.putText(image, raw_size_caption, (10, vis_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out.write(image)
        out.release()
    hdf.close()








