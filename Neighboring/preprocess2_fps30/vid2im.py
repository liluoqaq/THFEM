import os
import cv2
import numpy as np
from PIL import Image
import torch
import argparse
from tqdm import tqdm

VID_EXTENSIONS = ['.mp4']


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)


def tensor2npimage(image_tensor, imtype=np.uint8):
    # Tesnor in range [0,255]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2npimage(image_tensor[i], imtype))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, transpose=True):
    if transpose:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_images(images, folder, start_i, args):
    for i in range(len(images)):
        # if i < args.num_save:
        n_frame = "{:06d}".format(i + start_i)
        save_dir = os.path.join(args.celeb, folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(images[i], os.path.join(save_dir, n_frame + '.png'), transpose=folder == 'images')
        # else:
        #     break


def get_video_paths(dir):
    # Returns list of paths to video files
    video_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_video_file(fname):
                path = os.path.join(root, fname)
                video_files.append(path)
    return video_files


def detect_and_save_faces(mp4_path, start_i, args):

    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    images = []
    frame_infos = [f'{os.path.basename(mp4_path)[:-4]}_{i} {i+start_i}\n' for i in range(n_frames)]
    previous_box = None

    print('Reading %s, extracting faces, and saving images' % mp4_path)
    for i in tqdm(range(n_frames)):
        _, image = reader.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    dir_name = mp4_path.split("/")[-1][:-4]
    if args.save_full_frames:
        save_images(images, f'imgs/{dir_name}', start_i, args)

    reader.release()
    return start_i, frame_infos


def print_args(parser, args):
    message = ''
    message += '----------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '-------------------------------------------'
    print(message)


def main():
    print('-------------- Face detection -------------- \n')
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',
                        type=str,
                        default='0',
                        help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--celeb', type=str, default='test_data/ned', help='Path to celebrity folder.')
    parser.add_argument('--num_save',
                        type=int,
                        default=3000,
                        help='Whether to save full video frames (for reproducing the original clip)')
    parser.add_argument('--save_full_frames',
                        action='store_true',
                        default=True,
                        help='Whether to save full video frames (for reproducing the original clip)')

    args = parser.parse_args()
    print_args(parser, args)

    # check if face detection has already been done
    images_dir = os.path.join(args.celeb, 'images')
    if os.path.isdir(images_dir):
        print('Face detection already done!')

    else:
        # Figure out the device
        gpu_id = int(args.gpu_id)
        if gpu_id < 0:
            device = 'cpu'
        elif torch.cuda.is_available():
            if gpu_id >= torch.cuda.device_count():
                device = 'cuda:0'
            else:
                device = 'cuda:' + str(gpu_id)
        else:
            print('GPU device not available. Exit')
            exit(0)

        # subfolder containing videos
        videos_path = os.path.join(args.celeb, "video_fps25")

        # Store video paths in list.
        mp4_paths = get_video_paths(videos_path)
        n_mp4s = len(mp4_paths)
        print('Number of videos to process: %d \n' % n_mp4s)

        # Run detection
        n_completed = 0
        start_i = 0
        for path in mp4_paths:
            n_completed += 1
            start_i, frame_infos = detect_and_save_faces(path, start_i, args)

            print('(%d/%d) %s [SUCCESS]' % (n_completed, n_mp4s, path))
        print('DONE!')


if __name__ == "__main__":
    main()