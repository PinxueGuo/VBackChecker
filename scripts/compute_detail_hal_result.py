import json
import os
from sentence_transformers import SentenceTransformer, util
import torch


src_pred_result_path = '../outputs/exp19_hal-detail'

with open(os.path.join(src_pred_result_path, 'hal_result_detail.json'), 'r') as f:
    pred_data = json.load(f)
with open('HalMetaBench_balance_checked_1.json', 'r') as f:
    ann_data = json.load(f)


# Group the data by image name
pred_data_grouped = {}
for cap, img, gt, pred in zip(pred_data['caption_list'], pred_data['imagename_list'], pred_data['gt_list'], pred_data['pred_list']):
    if img not in pred_data_grouped:
        pred_data_grouped[img] = {'captions':[], 'gts': [], 'preds': []}
    pred_data_grouped[img]['captions'].append(cap)
    pred_data_grouped[img]['gts'].append(gt)
    pred_data_grouped[img]['preds'].append(pred)

# match the pred-gt pairs to get the gt halucination type and model source
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
pred_data_grouped_with_types = {}
for img, data_item in pred_data_grouped.items():
    gt_this_img_group = [d for d in ann_data if d['image'] == img]
    ann_captions = [d['object caption'] for d in gt_this_img_group]
    pred_captions = data_item['captions']
    pred_embeddings = model.encode(pred_captions, convert_to_tensor=True)
    ann_embeddings = model.encode(ann_captions, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(pred_embeddings, ann_embeddings)
    match_indices = cosine_scores.argmax(dim=1)
    hal_types = [gt_this_img_group[map_idx]['hallucination type'] for map_idx in match_indices]
    model_source = [gt_this_img_group[map_idx]['model'] for map_idx in match_indices]
    data_item['hal_types'] = hal_types
    data_item['model_source'] = model_source
    pred_data_grouped_with_types[img] = data_item

# flat to list
caption_list = []
gt_list = []
pred_list = []
hal_types = []
model_sources = []
for img, data_item in pred_data_grouped_with_types.items():
    caption_list.extend(data_item['captions'])
    gt_list.extend(data_item['gts'])
    pred_list.extend(data_item['preds'])
    hal_types.extend(data_item['hal_types'])
    model_sources.extend(data_item['model_source'])

# compute the result for each halucination type and model source
gt_tensor = torch.tensor(gt_list, dtype=torch.bool)
pred_tensor = torch.tensor(pred_list, dtype=torch.bool)
hal_type_set = set(hal_types)
model_source_set = set(model_sources)
results = {
    "hal_type_accuracy": {},
    "model_source_accuracy": {}
}

for hal_type in hal_type_set:
    curr_group = [t==hal_type for t in hal_types]
    curr_gt = gt_tensor[curr_group]
    curr_pred = pred_tensor[curr_group]
    gt_seg_indices = curr_gt==True
    gt_hal_indices = curr_gt==False
    gt_seg_pred_seg_num = (curr_pred[gt_seg_indices]==True).sum()
    gt_seg_pred_hal_num = (curr_pred[gt_seg_indices]==False).sum()
    gt_hal_pred_hal_num = (curr_pred[gt_hal_indices]==False).sum()
    gt_hal_pred_real_num = (curr_pred[gt_hal_indices]==True).sum()
    Nacc = gt_hal_pred_hal_num / gt_hal_indices.sum()
    Tacc = gt_seg_pred_seg_num / gt_seg_indices.sum()
    accuracy = torch.mean((curr_pred == curr_gt).float())
    # print('#############################################')
    # print(f'Halucination type: {hal_type}. Item number: {len(curr_gt)}')
    # print(f'gt-0_pred-0:{gt_hal_pred_hal_num}, gt-0_pred-1:{gt_hal_pred_real_num}, gt-1_pred-1:{gt_seg_pred_seg_num}, gt-1_pred-0:{gt_seg_pred_hal_num}')
    # print(f"Nacc: {gt_hal_pred_hal_num/gt_hal_indices.sum()}")
    # print(f'Tacc: {gt_seg_pred_seg_num/gt_seg_indices.sum()}')
    # print(f"Accuracy: {torch.mean((curr_pred == curr_gt).float())}", '\n')
    results["hal_type_accuracy"][hal_type] = {
        "item_number": len(curr_gt),
        "gt_0_pred_0": gt_hal_pred_hal_num.item(),
        "gt_0_pred_1": gt_hal_pred_real_num.item(),
        "gt_1_pred_1": gt_seg_pred_seg_num.item(),
        "gt_1_pred_0": gt_seg_pred_hal_num.item(),
        "Nacc": Nacc.item(),
        "Tacc": Tacc.item(),
        "accuracy": accuracy.item()
    }

for model_source in model_source_set:
    curr_group = [s==model_source for s in model_sources]
    curr_gt = gt_tensor[curr_group]
    curr_pred = pred_tensor[curr_group]
    gt_seg_indices = curr_gt==True
    gt_hal_indices = curr_gt==False
    gt_seg_pred_seg_num = (curr_pred[gt_seg_indices]==True).sum()
    gt_seg_pred_hal_num = (curr_pred[gt_seg_indices]==False).sum()
    gt_hal_pred_hal_num = (curr_pred[gt_hal_indices]==False).sum()
    gt_hal_pred_real_num = (curr_pred[gt_hal_indices]==True).sum()
    Nacc = gt_hal_pred_hal_num / gt_hal_indices.sum()
    Tacc = gt_seg_pred_seg_num / gt_seg_indices.sum()
    accuracy = torch.mean((curr_pred == curr_gt).float())
    # print('#############################################')
    # print(f'Model Source: {hal_type}. Item number: {len(curr_gt)}')
    # print(f'gt-0_pred-0:{gt_hal_pred_hal_num}, gt-0_pred-1:{gt_hal_pred_real_num}, gt-1_pred-1:{gt_seg_pred_seg_num}, gt-1_pred-0:{gt_seg_pred_hal_num}')
    # print(f"Nacc: {gt_hal_pred_hal_num/gt_hal_indices.sum()}")
    # print(f'Tacc: {gt_seg_pred_seg_num/gt_seg_indices.sum()}')
    # print(f"Accuracy: {torch.mean((curr_pred == curr_gt).float())}", '\n')
    results["model_source_accuracy"][model_source] = {
        "item_number": len(curr_gt),
        "gt_0_pred_0": gt_hal_pred_hal_num.item(),
        "gt_0_pred_1": gt_hal_pred_real_num.item(),
        "gt_1_pred_1": gt_seg_pred_seg_num.item(),
        "gt_1_pred_0": gt_seg_pred_hal_num.item(),
        "Nacc": Nacc.item(),
        "Tacc": Tacc.item(),
        "accuracy": accuracy.item()
    }

with open(os.path.join(src_pred_result_path, 'grouped_hal_result.json'), 'w') as f:
    json.dump(results, f, indent=4)