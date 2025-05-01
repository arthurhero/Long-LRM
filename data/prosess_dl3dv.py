import os
import json
import numpy as np
import cv2

"""
This file provides a function to process one scene in the DL3DV dataset into required format.
"""

def process_one_scene(scene):
    """
    scene: path to one scene folder, no trailing /
    """
    print(scene)
    scene_name = scene.split('/')[-1]
    json_file = scene + '/nerfstudio/transforms.json'
    json_data = json.load(open(json_file, 'r'))
    new_json_file = scene + '/opencv_cameras.json'
    new_data = {"scene_name": scene_name, "frames": []}
    w_ = json_data['w']
    h_ = json_data['h']
    fx_ = json_data['fl_x']
    fy_ = json_data['fl_y']
    cx_ = json_data['cx']
    cy_ = json_data['cy']
    k1,k2,p1,p2 = json_data['k1'], json_data['k2'], json_data['p1'], json_data['p2']
    distort = np.asarray([k1,k2,p1,p2])
    if h_ > w_:
        print("skip vertical videos for now", scene, h_,w_)
        return
    num_frams = len(json_data['frames'])
    print("num_frames: ", num_frams)

    # create undistort folder
    os.makedirs(scene+'/images_undistort', exist_ok=True)

    for i in range(num_frams):
        frame = json_data['frames'][i]
        file_path = 'images_4/' + frame['file_path'].split('/')[-1]

        # undistort
        image = cv2.imread(scene+'/nerfstudio/'+file_path, cv2.IMREAD_COLOR)
        h,w,_ = image.shape
        fx,fy,cx,cy = fx_/w_*w ,fy_/h_*h,cx_/w_*w,cy_/h_*h
        intr = np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])

        new_intr, roi = cv2.getOptimalNewCameraMatrix(intr, distort, (w,h), 0, (w,h))
        dst = cv2.undistort(image, intr, distort, None, new_intr)
        image = dst
        h,w,_ = image.shape
        fx,fy,cx,cy = new_intr[0,0], new_intr[1,1], new_intr[0,2], new_intr[1,2]

        file_path = 'images_undistort/' + frame['file_path'].split('/')[-1]
        cv2.imwrite(scene+'/'+file_path, image)

        c2w = np.asarray(frame["transform_matrix"])
        c2w[0:3, 1:3] *= -1
        c2w = c2w[[1,0,2,3], :]
        c2w[2,:] *= -1
        w2c = np.linalg.inv(c2w)
        frame_new = {"file_path": file_path, "w2c": w2c.tolist(), "h": h, "w": w, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
        new_data["frames"].append(frame_new)
    json.dump(new_data, open(new_json_file, 'w'), indent=4)