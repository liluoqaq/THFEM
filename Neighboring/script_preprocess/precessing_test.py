from glob import glob
import os
from argparse import ArgumentParser
import sys

sys.path.append(os.getcwd())


def main(args):
    allmp4s = glob(f'{args.video_path}/videos/*.mp4')

    path_fps25 = args.video_path + "/video_fps25"
    drivng_audio_path = args.video_path + "/driving_audio"
    os.makedirs(path_fps25, exist_ok=True)
    os.makedirs(drivng_audio_path, exist_ok=True)

    for mp4 in allmp4s:
        name = os.path.basename(mp4)
        os.system(f'ffmpeg -y -i {mp4} -filter:v fps=30 -ac 1 -ar 16000 -crf 10 {path_fps25}/{name}')
        os.system(f'ffmpeg -y -i {path_fps25}/{name} {path_fps25}/{name[:-4]}.wav')

    ##============from audio to extract the deep feature============
    print('======= extract speech in deepspeech_features =======')

    os.system(f'python preprocess2_fps30/deepspeech_features/extract_ds_features.py --input={path_fps25}')
    os.system(f'python preprocess2_fps30/deepfeature32.py --video_fps25_path {path_fps25}') 

    os.system(f'python preprocess2_fps30/vid2im.py --celeb {args.video_path}')
    ##是否按照NED 的处理方式


    #========== extract latent from cropped videos =======
    print('========== extract latent from cropped videos =======')

    os.system(f'python preprocess2_fps30/latent_extractor.py --root {args.video_path}')

    #=========== extract poseimg from latent =============
    print('=========== extract poseimg from latent =============')
    os.system(f'python preprocess2_fps30/generate_poseimg.py --root {args.video_path}')

    print('============== organize file for demo ===============')

    for mp4 in allmp4s:
        name = os.path.basename(mp4)[:-4]

        filename = f'{args.save_path}/video_processed/{name}'
        os.makedirs(f'{filename}/deepfeature32', exist_ok=True)
        os.makedirs(f'{filename}/latent_evp_25', exist_ok=True)
        os.makedirs(f'{filename}/poseimg', exist_ok=True)
        os.makedirs(f'{filename}/images_evp_25/cropped', exist_ok=True)

        # wav
        wav_path = f'{args.video_path}/driving_audio/{name}' + '.wav'
        os.system(f'cp {args.video_path}/video_fps25/{name}.wav {wav_path}')
        os.system(f'cp {args.video_path}/video_fps25/{name}.wav {filename}')
        # deepfeature32
        os.system(f'cp  {args.video_path}/deepfeature32/{name}.npy {filename}/deepfeature32/')
        # latent
        os.system(f'cp  {args.video_path}/latents/{name}.npy {filename}/latent_evp_25/')
        # poseimg
        os.system(f'cp  {args.video_path}/poseimg/{name}.npy.gz {filename}/poseimg/')
        # images_evp_25
        os.system(f'cp  {args.video_path}/imgs/{name}/* {filename}/images_evp_25/cropped/')
    os.system(f'python preprocess2_fps30/test_cropped.py --root {args.video_path}/video_processed --save_root {args.save_path}')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str, default="/data/2023_stu/zhenxuan/datasets/RAVDESS_Celeb_ours/test/dsm_editing/cross_id", help='Batch size to use')
    parser.add_argument('--save_path', type=str, default="/data/2023_stu/zhenxuan/datasets/RAVDESS_Celeb_ours/test/dsm_editing/cross_id", help='Batch size to use')
    args = parser.parse_args()
    main(args)

