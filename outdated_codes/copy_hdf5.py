f = h5py.File(hdf5_path, "r+")
    for ep in f['data']:
         num=len(f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)])
         if num==1:
             print("{} len is only 1,adding......".format(ep))
             row_1=num+1
             del f['data/{}/dones'.format(ep)]
             del f['data/{}/interventions'.format(ep)]
             del f['data/{}/policy_acting'.format(ep)]
             del f['data/{}/rewards'.format(ep)]
             del f['data/{}/states'.format(ep)]
             del f['data/{}/user_acting'.format(ep)]

             f.create_dataset('data/{}/dones'.format(ep), data=np.zeros((row_1 - 1)))
             f.create_dataset('data/{}/interventions'.format(ep), data=np.zeros((row_1, 1)))
             f.create_dataset('data/{}/policy_acting'.format(ep), data=np.zeros((row_1)))
             f.create_dataset('data/{}/rewards'.format(ep), data=np.zeros((row_1 - 1)))
             f.create_dataset('data/{}/states'.format(ep), data=np.zeros((0)))
             f.create_dataset('data/{}/user_acting'.format(ep), data=np.zeros((row_1, 1)))

             act=f['data/{}/actions'.format(ep)][0].copy()

             del f['data/{}/actions'.format(ep)]

             f['data/{}/actions'.format(ep)]=[act,act]


             wrist_img = f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)][0].copy()
             del f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)]
             f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)] = [wrist_img, wrist_img]

             if 'robot0_eye_in_hand_image_goal' in f['data/{}/obs'.format(ep)]:
                 wrist_img = f['data/{}/obs/robot0_eye_in_hand_image_goal'.format(ep)][0].copy()
                 del f['data/{}/obs/robot0_eye_in_hand_image_goal'.format(ep)]
                 f['data/{}/obs/robot0_eye_in_hand_image_goal'.format(ep)] = [wrist_img, wrist_img]

             if 'abs_rot' in f['data/{}/obs'.format(ep)]:
                 abs_rot = f['data/{}/obs/abs_rot'.format(ep)][0].copy()
                 del f['data/{}/obs/abs_rot'.format(ep)]
                 f['data/{}/obs/abs_rot'.format(ep)] = [abs_rot, abs_rot]

             if "gaussian_img" in f['data/{}/obs'.format(ep)]:
                 gaussian_img = f['data/{}/obs/gaussian_img'.format(ep)][0].copy()
                 del f['data/{}/obs/gaussian_img'.format(ep)]
                 f['data/{}/obs/gaussian_img'.format(ep)]=[gaussian_img, gaussian_img]

             if "gaussian_img_goal" in f['data/{}/obs'.format(ep)]:
                 gaussian_img_goal = f['data/{}/obs/gaussian_img_goal'.format(ep)][0].copy()
                 del f['data/{}/obs/gaussian_img_goal'.format(ep)]
                 f['data/{}/obs/gaussian_img_goal'] = gaussian_img_goal

    f.close()