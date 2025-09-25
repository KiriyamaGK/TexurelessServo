import h5py
import numpy as np
import cv2
import os
from utils.paths import return_disc_route

if __name__ == '__main__':
    hdf_pth=return_disc_route('/media/kiriyamagk/One Touch/AlignAnything_real/25.09.24/hdf5/mimic.hdf5')
    # hdf_pth = return_disc_route('One Touch/AlignAnything/25.04.144/hdf5/mimic.hdf5')
    # temp_dir='/media/kiriyamagk/One Touch/AlignAnything/25.01.17/hdf5/temp'
    hdf_base=os.path.dirname(hdf_pth)
    fps=30
    vis_h,vis_w=400,400
    only_rot_caption=False
    color_channel_inverse = False

    img_wrist_vis = True
    img2_wrist_vis = True
    img_wrist_light_vis = True
    img2_wrist_light_vis = True
    goal_image = True
    goal_image2 = True
    save_goal = False

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
                if color_channel_inverse:
                    image=image[:,:,::-1]
                raw_size=image.shape[0]
                image=image.astype('uint8')
                image=cv2.resize(image, (vis_w, vis_h))
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # if i==0:
                #     cv2.imwrite(os.path.join(hdf_base, str(j).zfill(5)+'_wrist.jpg'), image)
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

    if img2_wrist_vis:
        print("processing wrist video....")
        video_path = os.path.join(hdf_base, 'wrist2_video.mp4')
        out = cv2.VideoWriter(video_path, mp4, fps, (vis_w, vis_h))
        for i in range(len(hdf['data'])):
            print("processing demo_{}".format(i))
            demo_caption = 'demo_{}'.format(i)
            # os.makedirs(os.path.join(temp_dir,demo_caption), exist_ok=True)
            for j in range(len(hdf['data/demo_{}/actions'.format(i)])):
                image=hdf['data/demo_{}/obs/robot0_eye_in_hand_image_2'.format(i)][j]
                if color_channel_inverse:
                    image=image[:,:,::-1]
                raw_size=image.shape[0]
                image=image.astype('uint8')
                image=cv2.resize(image, (vis_w, vis_h))
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # if i==0:
                #     cv2.imwrite(os.path.join(hdf_base, str(j).zfill(5)+'_wrist.jpg'), image)
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


    if img_wrist_light_vis:
        print("processing light video....")
        video_path = os.path.join(hdf_base, 'wrist_light_video.mp4')
        out = cv2.VideoWriter(video_path, mp4, fps, (vis_w, vis_h))
        for i in range(len(hdf['data'])):
            print("processing demo_{}".format(i))
            demo_caption = 'demo_{}'.format(i)
            # os.makedirs(os.path.join(temp_dir,demo_caption), exist_ok=True)
            for j in range(len(hdf['data/demo_{}/actions'.format(i)])):
                image=hdf['data/demo_{}/obs/robot0_eye_in_hand_image_light'.format(i)][j]
                if color_channel_inverse:
                    image=image[:,:,::-1]
                raw_size=image.shape[0]
                image=image.astype('uint8')
                image=cv2.resize(image, (vis_w, vis_h))
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.imwrite(os.path.join(temp_dir,demo_caption,demo_caption +'_{}'.format(j)+ '.jpg'), image)
                # if i == 0:
                #     cv2.imwrite(os.path.join(hdf_base, str(j).zfill(5) + '_light.jpg'), image)
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

    if img2_wrist_light_vis:
        print("processing light video....")
        video_path = os.path.join(hdf_base, 'wrist2_light_video.mp4')
        out = cv2.VideoWriter(video_path, mp4, fps, (vis_w, vis_h))
        for i in range(len(hdf['data'])):
            print("processing demo_{}".format(i))
            demo_caption = 'demo_{}'.format(i)
            # os.makedirs(os.path.join(temp_dir,demo_caption), exist_ok=True)
            for j in range(len(hdf['data/demo_{}/actions'.format(i)])):
                image=hdf['data/demo_{}/obs/robot0_eye_in_hand_image_2_light'.format(i)][j]
                if color_channel_inverse:
                    image=image[:,:,::-1]
                raw_size=image.shape[0]
                image=image.astype('uint8')
                image=cv2.resize(image, (vis_w, vis_h))
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.imwrite(os.path.join(temp_dir,demo_caption,demo_caption +'_{}'.format(j)+ '.jpg'), image)
                # if i == 0:
                #     cv2.imwrite(os.path.join(hdf_base, str(j).zfill(5) + '_light.jpg'), image)
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

    if goal_image:
        print("processing wrist video....")
        video_path = os.path.join(hdf_base, 'goal_video.mp4')
        out = cv2.VideoWriter(video_path, mp4, 1, (vis_w, vis_h))
        for i in range(len(hdf['data'])):
            print("processing demo_{}".format(i))
            demo_caption = 'demo_{}'.format(i)
            # os.makedirs(os.path.join(temp_dir,demo_caption), exist_ok=True)

            image=hdf['data/demo_{}/obs/robot0_eye_in_hand_image'.format(i)][-1]
            if color_channel_inverse:
                image=image[:,:,::-1]
            raw_size=image.shape[0]
            image=image.astype('uint8')

            if save_goal:
                save_base=os.path.join(hdf_base, 'goal_images/img1')
                os.makedirs(save_base, exist_ok=True)
                cv2.imwrite(save_base+"/"+f"{i}" + ".png",image)

            image=cv2.resize(image, (vis_w, vis_h))
            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            # if i==0:
            #     cv2.imwrite(os.path.join(hdf_base, str(j).zfill(5)+'_wrist.jpg'), image)
            if only_rot_caption:
                action=hdf['data/demo_{}/actions'.format(i)][-1,2]
            else:
                action=hdf['data/demo_{}/actions'.format(i)][-1]
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
    if goal_image2:
        print("processing wrist video....")
        video_path = os.path.join(hdf_base, 'goal2_video.mp4')
        out = cv2.VideoWriter(video_path, mp4, 1, (vis_w, vis_h))
        for i in range(len(hdf['data'])):
            print("processing demo_{}".format(i))
            demo_caption = 'demo_{}'.format(i)
            # os.makedirs(os.path.join(temp_dir,demo_caption), exist_ok=True)

            image=hdf['data/demo_{}/obs/robot0_eye_in_hand_image_2'.format(i)][-1]
            if color_channel_inverse:
                image=image[:,:,::-1]
            raw_size=image.shape[0]
            image=image.astype('uint8')

            if save_goal:
                save_base=os.path.join(hdf_base, 'goal_images/img2')
                os.makedirs(save_base, exist_ok=True)
                cv2.imwrite(save_base+"/"+f"{i}" + ".png",image)

            image=cv2.resize(image, (vis_w, vis_h))
            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            # if i==0:
            #     cv2.imwrite(os.path.join(hdf_base, str(j).zfill(5)+'_wrist.jpg'), image)
            if only_rot_caption:
                action=hdf['data/demo_{}/actions'.format(i)][-1,2]
            else:
                action=hdf['data/demo_{}/actions'.format(i)][-1]
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

    hdf.close()








