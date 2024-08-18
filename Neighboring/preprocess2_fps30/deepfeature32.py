from glob import glob
import os
import numpy as np
from argparse import ArgumentParser


def main(args):

    allfeas = glob(f'{args.video_fps25_path}/*.npy')
    out = args.video_fps25_path.replace('video_fps25', 'deepfeature32')
    os.makedirs(out, exist_ok=True)

    for fea in allfeas:
        feature = np.load(fea).astype(np.float32)
        name = os.path.basename(fea)
        np.save(os.path.join(out, name), feature)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--video_fps25_path', type=str, default="preprocess2/video_fps25", help='Batch size to use')

    args = parser.parse_args()
    main(args)
