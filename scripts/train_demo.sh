#!/bin/bash

# Arguments instruction:
# --segmentation_model_path="****/sam_vit_h_4b8939.pth", path to the pretrained SAM pth file.
# --mllm_model_path="****/LLaVA-Lightning-7B-v1-1", path to a directory where LLaVA hugginface model stores.
# --vision-tower="****/clip-vit-large-patch14", path to a directory where CLIP-ViT-L hugginface model stores.
# --dataset_dir="****/data", path to the dataset directory. 
# --weight="****/gsva-7b-ft-gres.bin", path to a pretrained GSVA checkpoint if finetune.
# --precision="bf16", precision for training.
# --lora_r=8 , r = 8 for 7B model, r = 64 for 13B model.
# num_classes_per_sample=5, 5 is one of the optimum values of how many classes / objects sampled in one training example

# export TRANSFORMERS_OFFLINE=0
# export DS_SKIP_CUDA_CHECK=1

ds --master_port=24989 main.py \
  --exp_name="exp_vbackchecker" \
  --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights/LLaVA-Lightning-7B-v1-1" \
  --vision-tower="openai/clip-vit-large-patch14" \
  --weight="weights/gsva-7b-ft-gres.bin" \
  --dataset_dir="../lisa_datasets" \
  --dataset="refer_seg||vqa" \
  --sample_rates="4,1" \
  --refer_seg_data="R_Instruct_B||R_Instruct_A||grefcoco" \
  --lr=0.0001 \
  --precision="bf16" \
  --lora_r=8 \
  --steps_per_epoch 300
