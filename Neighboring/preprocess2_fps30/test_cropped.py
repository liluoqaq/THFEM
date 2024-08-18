import os
import shutil
from tqdm import tqdm
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def copy_frames(args, T):
    T = T
    save_root = "{:}/imgs_cropped/{:03}".format(args.save_root, T)
    os.makedirs(save_root, exist_ok=True)
    dirs = sorted(os.listdir(args.root))
    for dir in tqdm(dirs):
        os.makedirs(os.path.join(save_root, dir), exist_ok=True)
        img_root = os.path.join(args.root, dir, "images_evp_25/cropped")
        img_frames = sorted(os.listdir(img_root))
        nums = len(img_frames)
        for i in range(nums // T): #同＋1
            n = i * T 
            if n <= nums:
                img_num = os.path.join(img_root, "{:06}.png".format(n))
                save_img = os.path.join(save_root, dir, "{:06}.png".format(n))
                shutil.copy(img_num, save_img)

if __name__ == "__main__":
    print(1)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default="/data/2023_stu/zhenxuan/datasets/gen/NED-gen/video_processed", help='Batch size to use')
    parser.add_argument('--save_root', type=str, default="/data/2023_stu/zhenxuan/datasets/gen/NED-gen", help='Batch size to use')
    args = parser.parse_args()
    copy_frames(args, 1)
#python preprocess2/test_cropped.py  --root /data/2023_stu/zhenxuan/datasets/gen/NED-gen/video_processed --save_root /data/2023_stu/zhenxuan/datasets/gen/NED-gen