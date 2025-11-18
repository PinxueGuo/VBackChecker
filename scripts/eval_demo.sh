#!/bin/bash

# Arguments instruction:
# --val_dataset="R2_HalBench|unc|val", the format is Dataset name | version | split, e.g., "R2_HalBench|unc|val", "pope|unc|val" .
# --segmentation_model_path="****/sam_vit_h_4b8939.pth", path to the pretrained SAM pth file.
# --mllm_model_path="****/LLaVA-Lightning-7B-v1-1", path to a directory where LLaVA hugginface model stores.
# --vision-tower="****/clip-vit-large-patch14", path to a directory where CLIP-ViT-L hugginface model stores.
# --dataset_dir="****/data", path to the dataset directory. 
# --weight="****/...bin", path to a trained checkpoint.
# --precision="fp32", precision for evaluation.
# --lora_r=8 , r = 8 for 7B model, r = 64 for 13B model.
# --eval_only, use this flag to perform evaluation.

export TRANSFORMERS_OFFLINE=0
export DS_SKIP_CUDA_CHECK=1

CUDA_VISIBLE_DEVICES=0 ds main.py \
  --exp_name="exp_vbackchecker" \
  --val_dataset="R2_HalBench|unc|val" \
  --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights/LLaVA-Lightning-7B-v1-1" \
  --vision-tower="openai/clip-vit-large-patch14" \
  --dataset_dir="../lisa_datasets" \
  --weight="..." \
  --precision="fp16" \
  --lora_r=8 \
  --eval_only
