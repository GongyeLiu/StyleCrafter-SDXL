## If it's unaccessable for your machine to reach Internet,
## uncomment the following environment variable,
## these allows you to read model files directly from the local cache.
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export DIFFUSERS_OFFLINE=1
# export HF_HOME=/Path/To/Your/huggingFace
# export HUGGINFACE_HUB_CACHE=/Path/To/Your/huggingFace
# export TORCH_HOME=/Path/To/Your/torch


## For single GPU debugging
# python3 train_sdxl.py --config config/train/style_crafter_sdxl_512.yaml


## For real training

## Stage 1
OUTPUT_DIR="/Path/To/Your/Output/Directory"

NAME="stylecrafter_sdxl_512"
mkdir -p $OUTPUT_DIR/$NAME

accelerate launch --num_processes 8 --multi_gpu --mixed_precision "bf16" \
  train_sdxl.py \
  --config config/train/style_crafter_sdxl_512.yaml \
  --output_dir=$OUTPUT_DIR/$NAME \
  # --pretrained "/Path/To/Your/Output/Directory/pretrained.ckpt"


## Stage 2
OUTPUT_DIR="/Path/To/Your/Output/Directory"

NAME="stylecrafter_sdxl_1024"
mkdir -p $OUTPUT_DIR/$NAME

accelerate launch --num_processes 8 --multi_gpu --mixed_precision "bf16" \
  train_sdxl.py \
  --config config/train/style_crafter_sdxl_1024.yaml \
  --output_dir=$OUTPUT_DIR/$NAME \
  --pretrained "/Path/To/Your/Output/Directory/pretrained_stage1.ckpt"


## Stage 3
OUTPUT_DIR="/Path/To/Your/Output/Directory"

NAME="stylecrafter_sdxl_1024_noise_offset"
mkdir -p $OUTPUT_DIR/$NAME

accelerate launch --num_processes 8 --multi_gpu --mixed_precision "bf16" \
  train_sdxl.py \
  --config config/train/style_crafter_sdxl_1024_noise_offset.yaml \
  --output_dir=$OUTPUT_DIR/$NAME \
  --pretrained "/Path/To/Your/Output/Directory/pretrained_stage2.ckpt"