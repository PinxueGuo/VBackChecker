#!/bin/bash

# Arguments instruction:
# --segmentation_model_path="****/sam_vit_h_4b8939.pth", path to the pretrained SAM pth file.
# --mllm_model_path="****/llava-v1_1-7b", path to a directory where LLaVA hugginface model stores.
# --vision-tower="****/clip-vit-large-patch14", path to a directory where CLIP-ViT-L hugginface model stores.
# --dataset_dir="****/data", path to the dataset directory. 
# --weight="****/gsva-7b-ft-gres.bin", path to a pretrained GSVA checkpoint if finetune.
# --precision="bf16", precision for training.
# --lora_r=8 , r = 8 for 7B model, r = 64 for 13B model.
# num_classes_per_sample=5, 5 is one of the optimum values of how many classes / objects sampled in one training example

# export TRANSFORMERS_OFFLINE=0
# export DS_SKIP_CUDA_CHECK=1

##################### 7b ########################
# ds --master_port=24989 main.py \
#   --exp_name="Exp12_PT" \
#   --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
#   --mllm_model_path="weights/LLaVA-Lightning-7B-v1-1" \
#   --vision-tower="openai/clip-vit-large-patch14" \
#   --dataset_dir="../lisa_datasets" \
#   --dataset="sem_seg||refer_seg||vqa" \
#   --sample_rates="9,6,3" \
#   --refer_seg_data="refclef||refcoco||refcoco+||refcocog||grefcoco||grefcoco_syn" \
#   --lr=0.0003 \
#   --precision="bf16" \
#   --lora_r=8

# cd outputs/Exp12_PT/ckpt_model_10
# python zero_to_fp32.py . full_model.bin
# cd -

# ds --master_port=24989 main.py \
#   --exp_name="Exp12_FT" \
#   --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
#   --mllm_model_path="weights/LLaVA-Lightning-7B-v1-1" \
#   --vision-tower="openai/clip-vit-large-patch14" \
#   --weight="outputs/Exp12_PT/ckpt_model_10/full_model.bin" \
#   --dataset_dir="../lisa_datasets" \
#   --dataset="refer_seg" \
#   --sample_rates="1" \
#   --refer_seg_data="grefcoco_syn" \
#   --no_sampling \
#   --lr=0.0001 \
#   --precision="bf16" \
#   --lora_r=8

ds --master_port=24989 main.py \
  --exp_name="Exp22.1" \
  --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights/LLaVA-Lightning-7B-v1-1" \
  --vision-tower="openai/clip-vit-large-patch14" \
  --weight="weights/gsva-7b-ft-gres.bin" \
  --dataset_dir="../lisa_datasets" \
  --dataset="refer_seg||vqa" \
  --sample_rates="4,1" \
  --refer_seg_data="grefcoco_syn2||grefcoco_syn||grefcoco" \
  --lr=0.0001 \
  --precision="bf16" \
  --lora_r=8 \
  --steps_per_epoch 300

# # #################### 13b ########################
# ds --master_port=24989 main.py \
#   --exp_name="gsva-llama2-13B_pt-allmix" \
#   --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
#   --mllm_model_path="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
#   --vision-tower="openai/clip-vit-large-patch14" \
#   --dataset_dir="../lisa_datasets" \
#   --dataset="sem_seg||refer_seg||vqa||reason_seg" \
#   --sample_rates="9,6,3,1" \
#   --refer_seg_data="refclef||refcoco||refcoco+||refcocog||grefcoco" \
#   --lr=0.0003 \
#   --precision="bf16" \
#   --lora_r=64

# ds --master_port=24989 main.py \
#   --exp_name="gsva-llama2-13B_ft-grefcoco" \
#   --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
#   --mllm_model_path="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
#   --vision-tower="openai/clip-vit-large-patch14" \
#   --weight="outputs/gsva_ft-allmix/ckpt_model_10/gsva_ft-allmix_step5000.bin" \
#   --dataset_dir="../lisa_datasets" \
#   --dataset="refer_seg" \
#   --sample_rates="1" \
#   --refer_seg_data="grefcoco" \
#   --lr=0.0001 \
#   --precision="bf16" \
#   --lora_r=64 

# ds --master_port=24989 main.py \
#   --exp_name="Exp15_13b" \
#   --segmentation_model_path="weights/sam_vit_h_4b8939.pth" \
#   --mllm_model_path="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
#   --vision-tower="openai/clip-vit-large-patch14" \
#   --weight="weights/gsva-llama2-13b-ft-gres.bin" \
#   --dataset_dir="../lisa_datasets" \
#   --dataset="refer_seg||vqa" \
#   --sample_rates="4,1" \
#   --refer_seg_data="grefcoco_syn||grefcoco" \
#   --lr=0.0001 \
#   --precision="bf16" \
#   --lora_r=64 \
#   --steps_per_epoch 300 \
#   --auto_resume