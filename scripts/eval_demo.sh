#!/bin/bash

# Arguments instruction:
# --val_dataset="grefcoco|unc|val", the format is Dataset name | version | split, e.g., "grefcoco|unc|val", "refcoco+|unc|testA" .
# --segmentation_model_path="****/sam_vit_h_4b8939.pth", path to the pretrained SAM pth file.
# --mllm_model_path="****/llava-v1_1-7b", path to a directory where LLaVA hugginface model stores.
# --vision-tower="****/clip-vit-large-patch14", path to a directory where CLIP-ViT-L hugginface model stores.
# --dataset_dir="****/data", path to the dataset directory. 
# --weight="****/gsva-7b-ft-gres.bin", path to a GSVA checkpoint.
# --precision="fp32", precision for evaluation.
# --lora_r=8 , r = 8 for 7B model, r = 64 for 13B model.
# --eval_only, use this flag to perform evaluation.

export TRANSFORMERS_OFFLINE=0
export DS_SKIP_CUDA_CHECK=1

CUDA_VISIBLE_DEVICES=0 ds main.py \
  --exp_name="exp19_hal-detail" \
  --val_dataset="grefcoco_hal|unc|val" \
  --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights/LLaVA-Lightning-7B-v1-1" \
  --vision-tower="openai/clip-vit-large-patch14" \
  --dataset_dir="../lisa_datasets" \
  --weight="outputs/full_model_exp19-e5.bin" \
  --precision="fp16" \
  --lora_r=8 \
  --eval_only

ds --include localhost:1 --master_port 29501 main.py \
  --exp_name="exp19_pope-detail" \
  --val_dataset="grefcoco_hal_pope|unc|val" \
  --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights/LLaVA-Lightning-7B-v1-1" \
  --vision-tower="openai/clip-vit-large-patch14" \
  --dataset_dir="../lisa_datasets" \
  --weight="outputs/full_model_exp19-e5.bin" \
  --precision="fp16" \
  --lora_r=8 \
  --eval_only

# ds main.py \
#   --exp_name="debug" \
#   --val_dataset="grefcoco|unc|val" \
#   --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
#   --mllm_model_path="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
#   --vision-tower="openai/clip-vit-large-patch14" \
#   --dataset_dir="../lisa_datasets" \
#   --weight="outputs/Exp15_13b/ckpt_model_01/full_model.bin" \
#   --precision="fp16" \
#   --lora_r=64 \
#   --eval_only