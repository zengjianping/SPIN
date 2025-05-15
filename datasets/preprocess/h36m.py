import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
from spacepy import pycdf

def h36m_extract(dataset_path, out_path, protocol=1, extract_img=False):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_  = [], [], [], [], []

    # users in validation set
    user_list = [9, 11]

    # go over each user
    for ui, user_i in enumerate(user_list):
        print(f'Processing user {ui+1}/{len(user_list)}...')

        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bs')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for si, seq_i in enumerate(seq_list):
            print(f'Processing seq {si+1}/{len(seq_list)}...')

            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')
                os.makedirs(imgs_path, exist_ok=True)
                vidcap = cv2.VideoCapture(vid_file)
                success, image = vidcap.read()

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                print(f'Processing frame {frame_i}/{poses_3d.shape[0]}...', end='\r')

                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break

                # check if you can keep this frame
                if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)

                    # read GT bounding box
                    #mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    mask = np.array(bbox_h5py[bbox_h5py['Masks'][frame_i,0]], dtype=np.uint8).T
                    #print(image.shape, mask.shape)
                    #cv2.imshow('image', image)
                    #cv2.imshow('mask', (mask * 255))
                    #cv2.waitKey(5)
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0] # root-centered
                    S24 = np.zeros([24,4])
                    S24[global_idx, :3] = S17
                    S24[global_idx, 3] = 1

                    # store data
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    Ss_.append(S24)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 
        'h36m_valid_protocol%d.npz' % protocol)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       S=Ss_)
