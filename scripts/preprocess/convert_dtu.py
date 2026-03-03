'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import numpy as np
import json
from argparse import ArgumentParser
import os
import cv2
from PIL import Image, ImageFile
from glob import glob
import math
import sys
from pathlib import Path


dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
sys.path.append(dir_path.__str__())

from database import COLMAPDatabase  # NOQA
from read_write_model import read_model, rotmat2qvec  # NOQA

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_K_Rt_from_P(filename, P=None):
    # This function is borrowed from IDR: https://github.com/lioryariv/idr
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def create_init_files(pinhole_dict_file, db_file, out_dir):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create template
    with open(pinhole_dict_file) as fp:
        pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = '{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        fy = params[3]
        # fx = str(0.6 * float(w))
        # fy = str(0.6 * float(w))
        cx = str(float(w) / 2.0)
        cy = str(float(h) / 2.0)
        # cx = params[4]
        # cy = params[5]
        qvec = params[6:10]
        tvec = params[10:13]

        cam_line = cameras_line_template.format(
            camera_id="{camera_id}", width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                               tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}",
                                               image_name=img_name)
        template[img_name] = (cam_line, img_line)

    # read database
    db = COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    cameras_txt_lines = [template[img_name][0].format(camera_id=1)]
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        image_line = template[img_name][1].format(image_id=img_id, camera_id=1)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()
    
def dtu_to_json(args):
    assert args.dtu_path, "Provide path to DTU dataset"
    scene_list = os.listdir(args.dtu_path)

    for scene in scene_list:
        scene_path = os.path.join(args.dtu_path, scene)
        if not os.path.isdir(scene_path) or 'scan' not in scene:
            continue
        
        os.system(f"rm -rf {scene_path}/images/*")
        os.system(f"rm -rf {scene_path}/sparse/*")
        os.system(f"rm {scene_path}/database.db")
        # exit()

        # extract features
        os.system(f"colmap feature_extractor --database_path {scene_path}/database.db \
                --image_path {scene_path}/image \
                --ImageReader.single_camera 1 \
                --ImageReader.camera_model=PINHOLE \
                --SiftExtraction.use_gpu=true \
                --SiftExtraction.num_threads=32"
                  )

        # match features
        os.system(f"colmap exhaustive_matcher \
                --database_path {scene_path}/database.db \
                --SiftMatching.use_gpu=true"
                  )
        
        # read pose
        camera_param = dict(np.load(os.path.join(scene_path, 'cameras_sphere.npz')))
        images_lis = sorted(glob(os.path.join(scene_path, 'image/*.png')))
        w, h = Image.open(images_lis[0]).size
        pinhole_dict = {}
        for idx, image in enumerate(images_lis):
            image = os.path.basename(image)

            world_mat = camera_param['world_mat_%d' % idx]
            scale_mat = camera_param['scale_mat_%d' % idx]

            # scale and decompose
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic_param, c2w = load_K_Rt_from_P(None, P)
            
            w2c = np.linalg.inv(c2w)
            print(f"w2c\n {w2c}")
            qvec = rotmat2qvec(w2c[:3, :3])
            tvec = w2c[:3, 3]
            print(f"tvec {tvec}")
            fx = intrinsic_param[0][0]
            fy = intrinsic_param[1][1]
            cx = intrinsic_param[0][2]
            cy = intrinsic_param[1][2]
            params = [str(w), str(h), str(fx), str(fy), str(cx), str(cy),
                    str(qvec[0]), str(qvec[1]), str(qvec[2]), str(qvec[3]),
                    str(tvec[0]), str(tvec[1]), str(tvec[2])]
            pinhole_dict[image] = params

        
        # convert to colmap files
        pinhole_dict_file = os.path.join(scene_path, 'pinhole_dict.json')
        with open(pinhole_dict_file, 'w') as fp:
            json.dump(pinhole_dict, fp, indent=2, sort_keys=True)
        db_file = os.path.join(scene_path, 'database.db')
        sfm_dir = os.path.join(scene_path, 'sparse')
        create_init_files(pinhole_dict_file, db_file, sfm_dir)
        
        # bundle adjustment
        os.system(f"colmap point_triangulator \
                --database_path {scene_path}/database.db \
                --image_path {scene_path}/image \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --Mapper.tri_ignore_two_view_tracks=true"
                  )
        os.system(f"colmap bundle_adjuster \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --BundleAdjustment.refine_extrinsics=true"
                  )

        # undistortion
        os.system(f"colmap image_undistorter \
            --image_path {scene_path}/image \
            --input_path {scene_path}/sparse \
            --output_path {scene_path} \
            --output_type COLMAP"
                  )
        # exit()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dtu_path', type=str, default=None)

    args = parser.parse_args()

    dtu_to_json(args)
