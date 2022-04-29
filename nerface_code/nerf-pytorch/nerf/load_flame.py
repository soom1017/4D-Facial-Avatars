from http.client import BadStatusLine
import json
import os
import glob

import cv2
import imageio
import numpy as np
import torch

def load_dataset_dirs(basedir):
    file_list = glob.glob(os.path.join(basedir, '*'))
    dir_list = [f for f in file_list if os.path.isdir(f)]
    return dir_list

def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_flame_data(basedir, half_res=False, testskip=1, debug=False, expressions=True, load_bbox=True, test=False):
    print("starting data loading")

    dataset_dirs = load_dataset_dirs(basedir)
    all_imgs = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    ids = []

    splits = ["train", "val", "test"]
    if test:
        splits = ["test"]
    metas = {}

    counts = [0] * len(splits)
    for j, s in enumerate(splits):

        for i, dataset_dir in enumerate(dataset_dirs):
            with open(os.path.join(dataset_dir, f"transforms_{s}.json"), "r") as fp:
                metas[s] = json.load(fp)

            meta = metas[s]
            imgs = []
            poses = []
            expressions = []
            bboxs = []
            if s == "train" or testskip == 0:
                skip = 1
            else:
                skip = testskip

            for frame in meta["frames"][::skip]:
                fname = os.path.join(basedir, frame["file_path"] + ".png")
                imgs.append(imageio.imread(fname))

                poses.append(np.array(frame["transform_matrix"]))
                expressions.append(np.array(frame["expression"]))
                if load_bbox:
                    if "bbox" not in frame.keys():
                        bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                    else:
                        bboxs.append(np.array(frame["bbox"]))

            imgs = (np.array(imgs) / 255.0).astype(np.float32)
            poses = np.array(poses).astype(np.float32)
            expressions = np.array(expressions).astype(np.float32)
            bboxs = np.array(bboxs).astype(np.float32)
            # id coef added
            ids += np.full(imgs.shape[0], i)

            counts[j] += imgs.shape[0]
            all_imgs.append(imgs)
            all_poses.append(poses)
            all_expressions.append(expressions)
            all_bboxs.append(bboxs)
        
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    #focals = (meta["focals"])
    intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
    if meta["intrinsics"]:
        intrinsics = np.array(meta["intrinsics"])
    else:
        intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy


    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = H // 2
        W = W // 2
        #focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * 0.5
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)

    else:
        imgs = [
            torch.from_numpy(imgs[i])
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)

    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    bboxs[:,0:2] *= H
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")

    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, ids, bboxs
