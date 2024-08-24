# TFTEM - Official Implementation

## 1. Environment
Please note that our preprocessing and training code should be executed locally, and requires the following environmental configuration:

conda/mamba env create -f environment.yml

Note: We recommend using mamba to install dependencies, which is faster than conda.

## 2. Download Pretrained Models



## 3. Prepare Dataset

Download the [MEAD](https://wywu.github.io/projects/MEAD/MEAD.html) dataset. 

## 4. precessing
```bash
python script_preprocess/precessing_test.py \
--video_path data/MEAD \
--save_path data/MEAD
```

```bash
  python scripts/finetune_stage1.py --exp_dir exp/stage1 \
    --start_from_latent_avg \
    --use_skip --use_att 1 \
    --learning_rate 0.0001 \
    --max_steps 50000 \
    --val_interval 5000 \
    --checkpoint_path pretrained_models/pretrain.pt \
  ```

## 5. Test 

# python ref/demo_skip_change_30fps.py \
#     --root data/MEAD \
#     --root_wav data/MEAD/driving_audio \
#     --save_path result/images \
#     --ckpt ckpt/00000429-checkpoint.pth.tar \
