import os
import glob
import shutil
import random
from tqdm import tqdm
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import matplotlib.pyplot as plt
import cv2
import numpy as np


#M003_angry_3_024_0 0
def convert_imgs(args):
    save_root = f"{args.root}/imgs"
    source_root = f"{args.root}/images"
    path_txt = f"{args.root}/video_fps25/_frame_info.txt"
    with open(path_txt, "r") as f:
        data_list = f.read().splitlines()
        f.close()
    for data in tqdm(data_list):
        video_info, num = data.split(" ")
        actor, emo, v_num, frame_num = video_info.split("_")
        source_path = os.path.join(source_root, "{:06d}.png".format(int(num)))
        save_dir = actor + "_" + emo + "_" + v_num
        save_p = os.path.join(save_root, save_dir)
        os.makedirs(save_p, exist_ok=True)
        s_frame_num = save_p + "/" + "{:04}.jpg".format(int(frame_num) + 1)
        shutil.copy(source_path, s_frame_num)


# def convert_mask_align(args):
#     save_root = f"6ID-test"
#     source_root = f"{args.root}/output/exp1/005/image"
#     # source_root = "6ID-data/imgs_cropped/001"
#     data_list = sorted(os.listdir(source_root))
#     for data in tqdm(data_list):
#         _, actor, emo, level, v_num = data.split("_")
#         frames = sorted(os.listdir(os.path.join(source_root, data)))
#         for frame in frames:
#             source_path = os.path.join(source_root, data, frame)
#             save_dir = actor + "_" + emo + f"_{level}_" + v_num
#             save_p = os.path.join(save_root, save_dir, "gen")
#             os.makedirs(save_p, exist_ok=True)
#             s_frame_num = save_p + "/" + frame
#             shutil.copy(source_path, s_frame_num)

if __name__ == "__main__":
    print(1)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default="M-6ID/test", help='Batch size to use')
    args = parser.parse_args()

    convert_imgs(args)
