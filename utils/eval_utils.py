import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_clam import CLAM_SB, CLAM_MB , CaPa_MoE_without_clinical, CaPa_MoE_clinical_MLP, CaPa_MoE_clinical_gate, CaPa_MoE
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'CaPa_MoE_without_clinical':
        model = CaPa_MoE_without_clinical(**model_dict)
    elif args.model_type == 'CaPa_MoE_clinical_MLP':
        model = CaPa_MoE_clinical_MLP(**model_dict)
    elif args.model_type == 'CaPa_MoE_clinical_gate':
        model = CaPa_MoE_clinical_gate(**model_dict)
    elif args.model_type == 'CaPa_MoE':
        model = CaPa_MoE(**model_dict)


    print_network(model)

    print('Load Checkpoint from {}'.format(ckpt_path))

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args, args.clinical_dir)
    print('test_error: ', test_error)
    print('auc: ', auc)

    # 假设有 args.fold
    csv_name = f"fold_{args.fold}.csv"
    df.to_csv(csv_name, index=False)

    return model, patient_results, test_error, auc, df

def summary(model, loader, args, clinical_dir=None):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    clinical_df = pd.read_csv(clinical_dir, sep=",")
    clinical_df = clinical_df.set_index("slide_id")
    clinical_cols = ["age", "ER", "PR", "HER2", "Ki67", "subtype"]
    def get_clinical_tensor(slide_id: str) -> torch.Tensor:
        """
        根据 slide_id 从 clinical_df 里取出对应行，
        然后把 age, ER, PR, HER2, Ki67, subtype, pCR 变成 float tensor
        """
        row = clinical_df.loc[slide_id, clinical_cols]  # Series
        # 如果怕有缺失，可以加 row = row.fillna(0)
        clinical_vec = torch.tensor(row.values, dtype=torch.float32)
        return clinical_vec

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))
    all_gate = np.full(len(loader), -1, dtype=int)


    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (dataV, dataU, label, slide_id) in enumerate(loader):
        # 1) 临床向量
        clinical_tensor = get_clinical_tensor(slide_id) 
        dataV, dataU, label, clinical = (
            dataV.to(device),
            dataU.to(device),
            label.to(device),
            clinical_tensor.to(device),
        )

        # 2) 第一次前向：带 instance_eval=True，拿 instance_dict 里的 gate 信息
        logits, Y_prob, Y_hat, _, instance_dict = model(
            dataV, dataU, clinical, label=label, instance_eval=True
        )

        # 从 instance_dict 里取出 active_expert（0/1/2）
        active_expert = -1
        if isinstance(instance_dict, dict) and 'active_expert' in instance_dict:
            active_expert = int(instance_dict['active_expert'])
            all_gate[batch_idx] = active_expert  # 记录下来

        # 如果你想再跑一次“纯推理”的前向，也可以保留这段
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(dataV, dataU, clinical)

        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        # 这里也可以把 gate 存进 patient_results，方便之后用
        patient_results.update({
            slide_id: {
                'slide_id': np.array(slide_id),
                'prob': probs,
                'label': label.item(),
                'gate': active_expert,   # 新增
            }
        })

        error = calculate_error(Y_hat, label)
        test_error += error

    # del dataV, dataU
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {
        'slide_id': slide_ids,
        'Y': all_labels,
        'Y_hat': all_preds,
        'gate': all_gate,   # 新增这一列
    }
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})

    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
