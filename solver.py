# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------
# Seeing is Believing: Rich-Context Hallucination Detection for MLLMs via Backward Visual Grounding
# Modified by Pinxue Guo
# --------------------------------------------------------

import torch
import time
import tqdm
from utils import AverageMeter, ProgressMeter, Summary

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

import textwrap
import torch.nn as nn
import torch.nn.functional as F


def train_one_epoch(train_loader, model_engine, epoch, train_iter, args, logger):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")


    progress = ProgressMeter(
        len(train_loader) if args.no_sampling else args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs),
        logger=logger
    )

    # switch to train mode
    model_engine.train()
    end = time.time()
    if args.no_sampling:
        for global_step, input_dict in enumerate(train_loader):
            
            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            output_dict = model_engine(**input_dict)

            if global_step < 1:
                for conv_i in input_dict['conversation_list']:
                    print('------------------')
                    print(conv_i)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            mask_obj_loss = output_dict.get("mask_obj_loss", torch.zeros_like(mask_loss))
            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model_engine.backward(loss)
            model_engine.step()
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % args.print_freq == 0:
                if args.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()
                    losses.all_reduce()
                    ce_losses.all_reduce()
                    mask_bce_losses.all_reduce()
                    mask_dice_losses.all_reduce()
                    mask_losses.all_reduce()

                if args.rank == 0:
                    progress.display(1 + global_step)
                    
                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()

        return train_iter
    else:
        for global_step in range(args.steps_per_epoch):
            for i in range(args.grad_accumulation_steps):
                try:
                    input_dict = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    input_dict = next(train_iter)

                data_time.update(time.time() - end)
                input_dict = dict_to_cuda(input_dict)

                if args.precision == "fp16":
                    input_dict["images"] = input_dict["images"].half()
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                elif args.precision == "bf16":
                    input_dict["images"] = input_dict["images"].bfloat16()
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                else:
                    input_dict["images"] = input_dict["images"].float()
                    input_dict["images_clip"] = input_dict["images_clip"].float()

                output_dict = model_engine(**input_dict)
                
                if global_step < 1:
                    for conv_i in input_dict['conversation_list']:
                        print('------------------')
                        print(conv_i)

                loss = output_dict["loss"]
                ce_loss = output_dict["ce_loss"]
                mask_bce_loss = output_dict["mask_bce_loss"]
                mask_dice_loss = output_dict["mask_dice_loss"]
                mask_loss = output_dict["mask_loss"]

                losses.update(loss.item(), input_dict["images"].size(0))
                ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
                mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                mask_losses.update(mask_loss.item(), input_dict["images"].size(0))

                model_engine.backward(loss)
                model_engine.step()
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % args.print_freq == 0:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

                if args.rank == 0:
                    progress.display(1 + global_step)
                    
                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()

        return train_iter

@torch.no_grad()
def validate(val_loader, model_engine, epoch, args, logger):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].long()
        output_list = (pred_masks[0] > 0).long()
        assert len(pred_masks) == 1

        device = pred_masks[0].device
        intersection, union, acc_iou = 0.0, 0.0, 0.0  
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    logger.info(f"[{epoch + 1:d}] On {val_loader.dataset.ds} giou: {giou:.4f}, ciou: {ciou:.4f}.")
    return giou, ciou



@torch.no_grad()
def eval_gres(val_loader, model_engine, epoch, args, logger, tokenizer):
    model_engine.eval()
    inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    g_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    nt_tp_meter = AverageMeter("NT_TP", ":6.3f", Summary.SUM)
    nt_tn_meter = AverageMeter("NT_TN", ":6.3f", Summary.SUM)
    nt_fp_meter = AverageMeter("NT_FP", ":6.3f", Summary.SUM)
    nt_fn_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)
    is_grefcoco = (val_loader.dataset.ds == 'grefcoco' or val_loader.dataset.ds == 'grefcoco_syn')
    for sample_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        
        # for generating text from llm.generate(), else they are None
        if sample_idx < 1:
            input_dict['query_ids'], input_dict['query_attention_masks'] = get_only_query_input_ids(input_dict, tokenizer)

        input_dict["reeval"] = True
        src_input_ids = input_dict["input_ids"].clone()
        output_dict = model_engine(**input_dict)
        pred_masks = output_dict["pred_masks"][0].ge(0).int()
        gt_masks = output_dict["gt_masks"][0].int()
        
        output_ids = output_dict["output_ids"][0]
        # clean_output_ids = []
        # for ot_idx in range(len(output_dict["output_ids"])):
        #     curr_inds = output_dict['output_ids'][ot_idx].clone()
        #     is_seg_or_rej = ((curr_inds == args.seg_token_idx) | (curr_inds == args.rej_token_idx))
        #     ignore_seg_or_rej_mask = ((is_seg_or_rej.cumsum(dim=0) == 1) | (is_seg_or_rej.cumsum(dim=0) == 2)) & is_seg_or_rej
        #     curr_inds[ignore_seg_or_rej_mask] = 0
        #     clean_output_ids.append(curr_inds)
        # output_ids = torch.cat([o for o in clean_output_ids], dim=0)
        seg_or_rej_index = ((output_ids == args.seg_token_idx) | (output_ids == args.rej_token_idx)).nonzero(as_tuple=True)[0]
        pred_nts = (output_ids[seg_or_rej_index] == args.rej_token_idx)
        assert len(seg_or_rej_index) == len(gt_masks)
        assert len(pred_masks) == len(gt_masks)

        if sample_idx < 1:
            for idx, input_id in enumerate(src_input_ids):
                input_answer_start_index = (input_id==13566).nonzero().item()+3
                query_answer_start_index = (output_dict['gen_output_ids'][idx]==13566).nonzero()[0]+2
                input_answer = tokenizer.decode(input_id[input_answer_start_index:])
                pred_answer = tokenizer.decode(output_dict['gen_output_ids'][idx][query_answer_start_index:])
                print(gt_masks[idx].sum())
                print(input_answer)
                print(pred_answer, '\n')
            print('------------------------------')
            # visualize_save_path = 'outputs/visualize/GSVA/faith_val'
            # if not os.path.exists(visualize_save_path):
            #     os.makedirs(visualize_save_path)
            # visualize_results_faith(visualize_save_path, input_dict, pred_masks, gt_masks, pred_nts)
            
        for b_idx, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            
            if gt.sum() < 1.0: # empty target
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                if pred_nts[b_idx]:
                    nt_tp_meter.update(1.0)     # gt is empty, pred is empty
                    g_iou_meter.update(1.0)
                else:
                    nt_fn_meter.update(1.0)     # gt is empty, pred is target
                    g_iou_meter.update(0.0)
                    if is_grefcoco:
                        union_meter.update(union_i)
            else:
                if pred_nts[b_idx]:
                    nt_fp_meter.update(1.0)     # gt is target, pred is empty
                else:
                    nt_tn_meter.update(1.0)     # gt is target, pred is target
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                this_giou = inter_i / (union_i + 1e-8)
                inter_meter.update(inter_i)
                union_meter.update(union_i)
                g_iou_meter.update(this_giou)

    inter_meter.all_reduce()
    union_meter.all_reduce()
    g_iou_meter.all_reduce()
    nt_tp_meter.all_reduce()
    nt_tn_meter.all_reduce()
    nt_fp_meter.all_reduce()
    nt_fn_meter.all_reduce()
    
    # total_masks = nt_tp_meter.sum + nt_tn_meter.sum + nt_fp_meter.sum + nt_fn_meter.sum
    # masks_have_targets = nt_tn_meter.sum + nt_fp_meter.sum
    logger.info(f"gt-0_pred-0: {nt_tp_meter.sum}, gt-0_pred-1: {nt_fn_meter.sum}, gt-1_pred-1: {nt_tn_meter.sum}, gt-1_pred-0: {nt_fp_meter.sum}.")
    N_acc = nt_tp_meter.sum / (nt_tp_meter.sum + nt_fn_meter.sum) # for gt is empty, pred is empty
    T_acc = nt_tn_meter.sum / (nt_tn_meter.sum + nt_fp_meter.sum) # for gt is target, pred is target
    g_iou = g_iou_meter.avg[1]
    c_iou = (inter_meter.sum / (union_meter.sum + 1e-10))[1]
    logger.info(f"[{epoch + 1:d}] {val_loader.dataset.ds} giou: {g_iou:.4f}, ciou: {c_iou:.4f}, N_acc: {N_acc:.4f}, T_acc: {T_acc:.4f}.")
    return g_iou, c_iou, N_acc, T_acc

@torch.no_grad()
def eval_faith(val_loader, model_engine, epoch, args, logger, tokenizer):
    model_engine.eval()
    inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    g_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    nt_tp_meter = AverageMeter("NT_TP", ":6.3f", Summary.SUM)
    nt_tn_meter = AverageMeter("NT_TN", ":6.3f", Summary.SUM)
    nt_fp_meter = AverageMeter("NT_FP", ":6.3f", Summary.SUM)
    nt_fn_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)
    is_grefcoco = (val_loader.dataset.ds == 'grefcoco' or val_loader.dataset.ds == 'grefcoco_syn' or val_loader.dataset.ds == 'grefcoco_hal')
    for sample_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        
        # for generating text from llm.generate(), else they are None
        if sample_idx < 0:
            input_dict['query_ids'], input_dict['query_attention_masks'] = get_only_query_input_ids(input_dict, tokenizer)

        input_dict["reeval"] = True
        src_input_ids = input_dict["input_ids"].clone()
        output_dict = model_engine(**input_dict)
        pred_masks = output_dict["pred_masks"][0].ge(0).int()
        gt_masks = output_dict["gt_masks"][0].int()
        
        # clean_output_ids = []
        # for ot_idx in range(len(output_dict["output_ids"])):
        #     curr_inds = output_dict['output_ids'][ot_idx].clone()
        #     is_seg_or_rej = ((curr_inds == args.seg_token_idx) | (curr_inds == args.rej_token_idx))
        #     ignore_seg_or_rej_mask = ((is_seg_or_rej.cumsum(dim=0) == 1) | (is_seg_or_rej.cumsum(dim=0) == 2)) & is_seg_or_rej
        #     curr_inds[ignore_seg_or_rej_mask] = 0
        #     clean_output_ids.append(curr_inds)
        # output_ids = torch.cat([o for o in clean_output_ids], dim=0)
        output_ids = torch.cat([o for o in output_dict["output_ids"]], dim=0)
        seg_or_rej_index = ((output_ids == args.seg_token_idx) | (output_ids == args.rej_token_idx)).nonzero(as_tuple=True)[0]
        pred_nts = (output_ids[seg_or_rej_index] == args.rej_token_idx)
        assert len(seg_or_rej_index) == len(gt_masks)
        assert len(pred_masks) == len(gt_masks)

        if sample_idx < 0:
            for idx, input_id in enumerate(src_input_ids):
                input_answer_start_index = (input_id==13566).nonzero().item()+3
                query_answer_start_index = (output_dict['gen_output_ids'][idx]==13566).nonzero()[0]+2
                input_answer = tokenizer.decode(input_id[input_answer_start_index:])
                pred_answer = tokenizer.decode(output_dict['gen_output_ids'][idx][query_answer_start_index:])
                print(gt_masks[idx].sum())
                print(input_answer)
                print(pred_answer, '\n')
            print('------------------------------')
            # visualize_save_path = 'outputs/visualize/GSVA/faith_val'
            # if not os.path.exists(visualize_save_path):
            #     os.makedirs(visualize_save_path)
            # visualize_results_faith(visualize_save_path, input_dict, pred_masks, gt_masks, pred_nts)
            
        for b_idx, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            
            if gt.sum() < 1.0: # empty target
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                if pred_nts[b_idx]:
                    nt_tp_meter.update(1.0)     # gt is empty, pred is empty
                    g_iou_meter.update(1.0)
                else:
                    nt_fn_meter.update(1.0)     # gt is empty, pred is target
                    g_iou_meter.update(0.0)
                    if is_grefcoco:
                        union_meter.update(union_i)
            else:
                if pred_nts[b_idx]:
                    nt_fp_meter.update(1.0)     # gt is target, pred is empty
                else:
                    nt_tn_meter.update(1.0)     # gt is target, pred is target
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                this_giou = inter_i / (union_i + 1e-8)
                inter_meter.update(inter_i)
                union_meter.update(union_i)
                g_iou_meter.update(this_giou)

    inter_meter.all_reduce()
    union_meter.all_reduce()
    g_iou_meter.all_reduce()
    nt_tp_meter.all_reduce()
    nt_tn_meter.all_reduce()
    nt_fp_meter.all_reduce()
    nt_fn_meter.all_reduce()

    # total_masks = nt_tp_meter.sum + nt_tn_meter.sum + nt_fp_meter.sum + nt_fn_meter.sum
    # masks_have_targets = nt_tn_meter.sum + nt_fp_meter.sum
    logger.info(f"gt-0_pred-0: {nt_tp_meter.sum}, gt-0_pred-1: {nt_fn_meter.sum}, gt-1_pred-1: {nt_tn_meter.sum}, gt-1_pred-0: {nt_fp_meter.sum}.")
    N_acc = nt_tp_meter.sum / (nt_tp_meter.sum + nt_fn_meter.sum) # for gt is empty, pred is empty
    T_acc = nt_tn_meter.sum / (nt_tn_meter.sum + nt_fp_meter.sum) # for gt is target, pred is target
    g_iou = g_iou_meter.avg[1]
    c_iou = (inter_meter.sum / (union_meter.sum + 1e-10))[1]
    logger.info(f"[{epoch + 1:d}] {val_loader.dataset.ds} giou: {g_iou:.4f}, ciou: {c_iou:.4f}, N_acc: {N_acc:.4f}, T_acc: {T_acc:.4f}.")
    return g_iou, c_iou, N_acc, T_acc

@torch.no_grad()
def eval_hal(val_loader, model_engine, epoch, args, logger, tokenizer):
    model_engine.eval()
    inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    g_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    nt_tp_meter = AverageMeter("NT_TP", ":6.3f", Summary.SUM)
    nt_tn_meter = AverageMeter("NT_TN", ":6.3f", Summary.SUM)
    nt_fp_meter = AverageMeter("NT_FP", ":6.3f", Summary.SUM)
    nt_fn_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)
    is_grefcoco = (val_loader.dataset.ds == 'grefcoco' or val_loader.dataset.ds == 'grefcoco_syn' or val_loader.dataset.ds == 'grefcoco_hal')
    gt_seg_list = []
    pred_seg_list = []
    obj_cap_list = []
    img_filename_list = []
    for sample_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        
        # for generating text from llm.generate(), else they are None
        if sample_idx < 100:
            input_dict['query_ids'], input_dict['query_attention_masks'] = get_only_query_input_ids(input_dict, tokenizer)

        input_dict["reeval"] = True
        src_input_ids = input_dict["input_ids"].clone()
        output_dict = model_engine(**input_dict)
        pred_masks = output_dict["pred_masks"][0].ge(0).int()
        gt_masks = output_dict["gt_masks"][0].int()
        
        # clean_output_ids = []
        # for ot_idx in range(len(output_dict["output_ids"])):
        #     curr_inds = output_dict['output_ids'][ot_idx].clone()
        #     is_seg_or_rej = ((curr_inds == args.seg_token_idx) | (curr_inds == args.rej_token_idx))
        #     ignore_seg_or_rej_mask = ((is_seg_or_rej.cumsum(dim=0) == 1) | (is_seg_or_rej.cumsum(dim=0) == 2)) & is_seg_or_rej
        #     curr_inds[ignore_seg_or_rej_mask] = 0
        #     clean_output_ids.append(curr_inds)
        # output_ids = torch.cat([o for o in clean_output_ids], dim=0)
        output_ids = torch.cat([o for o in output_dict["output_ids"]], dim=0)
        seg_or_rej_index = ((output_ids == args.seg_token_idx) | (output_ids == args.rej_token_idx)).nonzero(as_tuple=True)[0]
        pred_nts = (output_ids[seg_or_rej_index] == args.rej_token_idx)
        assert len(seg_or_rej_index) == len(gt_masks)
        assert len(pred_masks) == len(gt_masks)

        if sample_idx < 100:
            gt_ans_to_show = []
            pred_ans_to_show = []
            for idx, input_id in enumerate(src_input_ids):
                input_answer_start_index = (input_id==13566).nonzero().item()+2
                query_answer_start_index = (output_dict['gen_output_ids'][idx]==13566).nonzero()[0]+2
                input_answer = tokenizer.decode(input_id[input_answer_start_index:])
                pred_answer = tokenizer.decode(output_dict['gen_output_ids'][idx][query_answer_start_index:])
                print(gt_masks[idx].sum())
                print(input_answer)
                print(pred_answer, '\n')
                gt_ans_to_show.append(input_answer)
                pred_ans_to_show.append(pred_answer)

            visualize_save_path = 'outputs/visualize/Exp22.1_Hal'
            if not os.path.exists(visualize_save_path):
                os.makedirs(visualize_save_path)
            visualize_results_faith(visualize_save_path, input_dict, pred_masks, gt_masks, pred_nts, gt_ans_to_show, pred_ans_to_show)
            print('------------------------------')

        for b_idx, (pred, gt, prompt) in enumerate(zip(pred_masks, gt_masks, input_dict['conversation_list'])):
            
            obj_cap_list.append(get_objcap_from_prompt(prompt, tokenizer))
            img_filename_list.append(input_dict['image_paths'][0].split('/')[-1])

            if gt.sum() < 1.0: # empty target
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                if pred_nts[b_idx]:
                    nt_tp_meter.update(1.0)     # gt is empty, pred is empty
                    g_iou_meter.update(1.0)
                    gt_seg_list.append(0)
                    pred_seg_list.append(0)
                else:
                    nt_fn_meter.update(1.0)     # gt is empty, pred is target
                    g_iou_meter.update(0.0)
                    if is_grefcoco:
                        union_meter.update(union_i)
                    gt_seg_list.append(0)
                    pred_seg_list.append(1)
            else:
                if pred_nts[b_idx]:
                    nt_fp_meter.update(1.0)     # gt is target, pred is empty
                    gt_seg_list.append(1)
                    pred_seg_list.append(0)
                else:
                    nt_tn_meter.update(1.0)     # gt is target, pred is target
                    gt_seg_list.append(1)
                    pred_seg_list.append(1)
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                this_giou = inter_i / (union_i + 1e-8)
                inter_meter.update(inter_i)
                union_meter.update(union_i)
                g_iou_meter.update(this_giou)

    inter_meter.all_reduce()
    union_meter.all_reduce()
    g_iou_meter.all_reduce()
    nt_tp_meter.all_reduce()
    nt_tn_meter.all_reduce()
    nt_fp_meter.all_reduce()
    nt_fn_meter.all_reduce()

    # total_masks = nt_tp_meter.sum + nt_tn_meter.sum + nt_fp_meter.sum + nt_fn_meter.sum
    # masks_have_targets = nt_tn_meter.sum + nt_fp_meter.sum
    logger.info(f"gt-0_pred-0: {nt_tp_meter.sum}, gt-0_pred-1: {nt_fn_meter.sum}, gt-1_pred-1: {nt_tn_meter.sum}, gt-1_pred-0: {nt_fp_meter.sum}.")
    N_acc = nt_tp_meter.sum / (nt_tp_meter.sum + nt_fn_meter.sum) # for gt is empty, pred is empty
    T_acc = nt_tn_meter.sum / (nt_tn_meter.sum + nt_fp_meter.sum) # for gt is target, pred is target
    logger.info(f"[{epoch + 1:d}] {val_loader.dataset.ds} N_acc: {N_acc:.4f}, T_acc: {T_acc:.4f}, Acc: {(N_acc+T_acc)/2.0:.4f}.")

    return N_acc, T_acc, (N_acc+T_acc)/2.0, obj_cap_list, img_filename_list, gt_seg_list, pred_seg_list


def get_objcap_from_prompt(prompt, tokenizer):
    match = re.search(r'What is (.*?) Please output segmentation mask\.', prompt)
    # print(match.group(1))
    return match.group(1)


def get_only_query_input_ids(input_dict, tokenizer):

    only_query_ids_list = []        # [b, ids_len]
    only_query_attmask_list = []    
    for in_id in input_dict['input_ids']:
        # question_end_index = (in_id==13566).nonzero().item()+2
        before_segrej_index = (in_id==13566).nonzero().item()+7
        query_ids = in_id[:before_segrej_index].clone() # "What is green and gray umbrella in the leftin this image? Please output segmentation mask. ASSISTANT: The segmentation result of"
        only_query_ids_list.append(query_ids)
        only_query_attmask_list.append(torch.ones_like(query_ids))
    # # right padding
    # padded_only_query_ids = nn.utils.rnn.pad_sequence(only_query_ids_list, batch_first=True, padding_value=0)   # [b, max_ids_len]
    # padded_only_query_attmask = nn.utils.rnn.pad_sequence(only_query_attmask_list, batch_first=True, padding_value=0).bool()
    # left padding
    padded_only_query_ids = left_pad_sequence(only_query_ids_list, batch_first=True, padding_value=0)
    padded_only_query_attmask = left_pad_sequence(only_query_attmask_list, batch_first=True, padding_value=0).bool()

    return padded_only_query_ids, padded_only_query_attmask

def left_pad_sequence(sequences, batch_first=False, padding_value=0):
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) if batch_first else (max_len, len(sequences))
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, -length:] = tensor
        else:
            out_tensor[-length:, i] = tensor
    return out_tensor


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape, f"output_shape = {output.shape}, target_shape = {target.shape}"
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict



def visualize_results(save_path, input_dict, pred_masks, gt_masks, pred_nts):
    gt_label_color = torch.tensor([1, 0, 0], device=gt_masks.device, dtype=torch.float).unsqueeze(-1)       # red
    pred_label_color = torch.tensor([0, 1, 0], device=pred_masks.device, dtype=torch.float).unsqueeze(-1)   # green

    for b_i in range(len(input_dict['image_paths'])):
        # 读取图像并转换为张量
        image_path = input_dict['image_paths'][b_i]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float().to(pred_masks.device) / 255.0

        # 获取每个对象的描述
        input_prompt = input_dict['conversation_list'][b_i]
        assistant_part = re.search(r'ASSISTANT:(.*)</s>', input_prompt).group(1)
        obj_cap_list = re.findall(r'[^,]+:\[SEG\]|[^,]+:\[REJ\]', assistant_part)

        # 将 pred_nts 中的负样本掩码置为 0
        pred_masks[pred_nts] = 0

        # 创建一个大图来显示结果
        num_objects = len(obj_cap_list)
        fig, axes = plt.subplots(num_objects, 2, figsize=(10, 5 * num_objects))
        for i in range(num_objects):
            # 获取 GT-mask 和 pred-mask
            gt_mask = gt_masks[i]
            pred_mask = pred_masks[i]

            # 将掩码叠加到图像上
            gt_overlay = image.clone()
            pred_overlay = image.clone()
            gt_overlay[:, gt_mask > 0] = 0.5*gt_overlay[:, gt_mask > 0] + 0.5*gt_label_color
            pred_overlay[:, pred_mask > 0] = 0.5*pred_overlay[:, pred_mask > 0] + 0.5*pred_label_color

            # 将结果转换为 CPU 上的 NumPy 数组
            gt_overlay = gt_overlay.cpu().numpy().transpose(1, 2, 0)
            pred_overlay = pred_overlay.cpu().numpy().transpose(1, 2, 0)

            # 显示原图像叠加 GT-mask
            axes[i, 0].imshow(gt_overlay)
            axes[i, 0].set_title(f'GT: {obj_cap_list[i]}')
            axes[i, 0].axis('off')

            # 显示原图像叠加 pred-mask
            axes[i, 1].imshow(pred_overlay)
            axes[i, 1].set_title(f'Pred: {obj_cap_list[i]}')
            axes[i, 1].axis('off')

        # 保存结果图像
        plt.tight_layout()
        plt.savefig(f"{save_path}/{image_path.split('/')[-1][:-4]}.png")
        plt.close()

# def visualize_results_faith(save_path, input_dict, pred_masks, gt_masks, pred_nts):
#     gt_label_color = torch.tensor([1, 0, 0], device=gt_masks.device, dtype=torch.float).unsqueeze(-1)       # red
#     pred_label_color = torch.tensor([0, 1, 0], device=pred_masks.device, dtype=torch.float).unsqueeze(-1)   # green

#     for b_i in range(len(input_dict['image_paths'])):
#         # 读取图像并转换为张量
#         image_path = input_dict['image_paths'][b_i]
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = torch.from_numpy(image).permute(2, 0, 1).float().to(pred_masks.device) / 255.0
        
#         # 将 pred_nts 中的负样本掩码置为 0
#         pred_masks[pred_nts] = 0

#         # 创建一个大图来显示结果
#         num_objects = len(input_dict['conversation_list'])
#         fig, axes = plt.subplots(num_objects, 2, figsize=(10, 5 * num_objects))

#         for i, input_prompt in enumerate(input_dict['conversation_list']):
#             assistant_part = re.search(r'ASSISTANT:(.*)</s>', input_prompt).group(1)
#             # obj_cap = re.search(r'Sure,(.*:\[(SEG|REJ)\])', assistant_part).group(1)
#             obj_cap = assistant_part

#             gt_mask = gt_masks[i]
#             pred_mask = pred_masks[i]

#             # 将掩码叠加到图像上
#             gt_overlay = image.clone()
#             pred_overlay = image.clone()
#             gt_overlay[:, gt_mask > 0] = 0.3 * gt_overlay[:, gt_mask > 0] + 0.7 * gt_label_color
#             pred_overlay[:, pred_mask > 0] = 0.3 * pred_overlay[:, pred_mask > 0] + 0.7 * pred_label_color

#             # 将结果转换为 CPU 上的 NumPy 数组
#             gt_overlay = gt_overlay.cpu().numpy().transpose(1, 2, 0)
#             pred_overlay = pred_overlay.cpu().numpy().transpose(1, 2, 0)

#             # 显示原图像叠加 GT-mask
#             wrapped_title = "\n".join(textwrap.wrap(f'GT: {obj_cap}', width=40))  # 调整width以适应你的需求
#             if len(axes)==1:
#             # 当只有一行时
#                 axes[0].imshow(gt_overlay)
#                 axes[0].set_title(wrapped_title, fontsize=10)  # 调整fontsize以适应你的需求
#                 axes[0].axis('off')
#                 # 显示原图像叠加 pred-mask
#                 axes[1].imshow(pred_overlay)
#                 # axes[i, 1].set_title(f'Pred: {obj_cap_list[i]}')
#                 axes[1].axis('off')
#             else:
#             # 有多行时
#                 axes[i, 0].imshow(gt_overlay)
#                 axes[i, 0].set_title(wrapped_title, fontsize=10)  # 调整fontsize以适应你的需求
#                 axes[i, 0].axis('off')
#                 # 显示原图像叠加 pred-mask
#                 axes[i, 1].imshow(pred_overlay)
#                 # axes[i, 1].set_title(f'Pred: {obj_cap_list[i]}')
#                 axes[i, 1].axis('off')

#         # 保存结果图像
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/{image_path.split('/')[-1][:-4]}.png")
#         plt.close()


def visualize_results_faith(save_path, input_dict, pred_masks, gt_masks, pred_nts, gt_ans, pred_ans):
    gt_label_color = torch.tensor([1, 0, 0], device=gt_masks.device, dtype=torch.float).unsqueeze(-1)       # red
    pred_label_color = torch.tensor([0, 1, 0], device=pred_masks.device, dtype=torch.float).unsqueeze(-1)   # green

    for b_i in range(len(input_dict['image_paths'])):
        # 读取图像并转换为张量
        image_path = input_dict['image_paths'][b_i]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float().to(pred_masks.device) / 255.0
        
        # 将 pred_nts 中的负样本掩码置为 0
        pred_masks[pred_nts] = 0

        for i, input_prompt in enumerate(input_dict['conversation_list']):
            assistant_part = re.search(r'ASSISTANT:(.*)</s>', input_prompt).group(1)
            obj_cap = assistant_part

            gt_mask = gt_masks[i]
            pred_mask = pred_masks[i]

            # 将掩码叠加到图像上
            gt_overlay = image.clone()
            pred_overlay = image.clone()
            gt_overlay[:, gt_mask > 0] = 0.3 * gt_overlay[:, gt_mask > 0] + 0.7 * gt_label_color
            pred_overlay[:, pred_mask > 0] = 0.3 * pred_overlay[:, pred_mask > 0] + 0.7 * pred_label_color

            # 将结果转换为 CPU 上的 NumPy 数组
            gt_overlay = gt_overlay.cpu().numpy().transpose(1, 2, 0)
            pred_overlay = pred_overlay.cpu().numpy().transpose(1, 2, 0)

            # 创建一个大图来显示结果
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            # 显示原图像叠加 GT-mask
            wrapped_title = "\n".join(textwrap.wrap(f'GT: {obj_cap}', width=40))  # 调整width以适应你的需求
            axes[0].imshow(gt_overlay)
            axes[0].set_title(wrapped_title, fontsize=12)  # 调整fontsize以适应你的需求
            axes[0].axis('off')

            # 显示原图像叠加 pred-mask
            wrapped_title = "\n".join(textwrap.wrap(f'Pred: {pred_ans[i]}', width=40))  # 调整width以适应你的需求
            axes[1].imshow(pred_overlay)
            axes[1].set_title(wrapped_title, fontsize=12)  # 调整fontsize以适应你的需求
            axes[1].axis('off')

            # 保存结果图像
            plt.tight_layout()
            plt.savefig(f"{save_path}/{image_path.split('/')[-1][:-4]}_{i}.png")
            plt.close()

            # 保存对应的 gt_ans 和 pred_ans 到 JSON 文件
            json_filename = f"{save_path}/{image_path.split('/')[-1][:-4]}_{i}.json"
            with open(json_filename, 'w') as json_file:
                json.dump({'gt_ans': gt_ans[i], 'pred_ans': pred_ans[i]}, json_file, ensure_ascii=False, indent=4)