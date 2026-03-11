import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import csv

from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
from scipy.stats import pearsonr 

from utils import get_data, create_dataset_pt3
from model import GPS_prompt_mt_2roi

################## created by Jing Xia ###############################
roi_num = 264
hidden_channels = roi_num
hidden_channels2 = 256
hc = roi_num * roi_num
hc2 = 256

pt_dim = 8
pt_dim2 = 512
node_pt_dim = 8

epoch_num = 40
lr = 1e-4
weight_decay = 1e-3
batch_size = 10
kk = 66        
thres = 0.3
n_splits = 10

lambda_cls = 1.0
lambda_reg = 0.25  
lambda_con = 0.25  

tau = 0.1         

results_root = "/root/autodl-tmp/data/meta-prompt/results_FC"
os.makedirs(results_root, exist_ok=True)

exp_name = (
    f"COBRE_prompt_mt_2roi_"
    f"{hidden_channels}_{hidden_channels2}_"
    f"{lr}_{epoch_num}_{hc2}_"
    f"{pt_dim}_{pt_dim2}_"
    f"{kk}_{thres}_{tau}_{node_pt_dim}_"
    f"{lambda_cls}_{lambda_reg}_{lambda_con}_"
    f"{int(time.time())}"
)
timefile = os.path.join(results_root, exp_name)
os.makedirs(timefile, exist_ok=True)

print("Result dir:", timefile)

task='positive'
task2='cognition'
task5= 'soc'  #PS
task6= 'BACS'
task7= 'WM'
task8= 'VL'
connpath = '/root/autodl-tmp/data/meta-prompt/COBRE_power/X.npy'  ##need to modify
labelpath = '/root/autodl-tmp/data/meta-prompt/COBRE_power/Y_' + task +'.npy'  ##need to modify
scorepath5 = '/root/autodl-tmp/data/meta-prompt/COBRE_power/Y_' + task5 +'.npy'  ##need to modify
scorepath6 = '/root/autodl-tmp/data/meta-prompt/COBRE_power/Y_' + task6 +'.npy'  ##need to modify
scorepath7 = '/root/autodl-tmp/data/meta-prompt/COBRE_power/Y_' + task7 +'.npy'  ##need to modify
scorepath8 = '/root/autodl-tmp/data/meta-prompt/COBRE_power/Y_' + task8 +'.npy'

roi_pt_path   = '/root/autodl-tmp/data/meta-prompt/MT/language_models/power_cog_gpt.pt' #power_cog_clip.pt'
roi_pt2_path  = '/root/autodl-tmp/data/meta-prompt/MT/language_models/power_SZ_gpt.pt'
sub_pt_path   = '/root/autodl-tmp/data/meta-prompt/MT/language_models/Subject_prompt_SZ_clip.pt'
node_pt_path  = '/root/autodl-tmp/data/meta-prompt/MT/language_models/Power_dim_gpt.pt'



def write_attention(filename, train_attention, train_node_index, fold, task):
    filename1 = filename + '/train_attention' + str(fold) + task + ".npy"
    np.save(filename1, train_attention)
    print("save:", filename1, "shape:", train_attention.shape)
    mean_train = np.mean(train_attention, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + task + ".npy"
    np.save(filename2, mean_train)

    filename1 = filename + '/train_node_index' + str(fold) + task + ".npy"
    np.save(filename1, train_node_index)
    mean_train = np.mean(train_node_index, axis=0)
    filename2 = filename + '/train_mean_node_index' + str(fold) + task + ".npy"
    np.save(filename2, mean_train)


def write_attention2(filename, train_attention, fold, task):
    filename1 = filename + '/train_attention' + str(fold) + task + ".npy"
    print("save:", filename1, "shape:", train_attention.shape)
    np.save(filename1, train_attention)
    mean_train = np.mean(train_attention, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + task + ".npy"
    np.save(filename2, mean_train)


def write_attention3(filename, train_index_szf, train_weight_szf, fold, task):
    filename1 = filename + '/train_attention' + str(fold) + task + ".npy"
    np.save(filename1, train_weight_szf)
    mean_train = np.mean(train_weight_szf, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + task + ".npy"
    np.save(filename2, mean_train)

    filename1 = filename + '/train_node_index' + str(fold) + task + ".npy"
    np.save(filename1, train_index_szf)
    mean_train = np.mean(train_index_szf, axis=0)
    filename2 = filename + '/train_mean_node_index' + str(fold) + task + ".npy"
    np.save(filename2, mean_train)


def compute_class_weights(labels_np, device):
    unique, counts = np.unique(labels_np, return_counts=True)
    freq = counts / counts.sum()
    weights = 1.0 / freq
    w_tensor = torch.zeros(int(unique.max()) + 1, dtype=torch.float32)
    for u, w in zip(unique, weights):
        w_tensor[int(u)] = float(w)
    return w_tensor.to(device)

def contrastive_loss(g_embed, t_embed, tau=0.1):
    
    if g_embed.size(0) < 2:
        return torch.tensor(0.0, device=g_embed.device)

    g = F.normalize(g_embed, dim=1)
    t = F.normalize(t_embed, dim=1)

    logits = g @ t.t() / tau         # (B, B)
    labels = torch.arange(g.size(0), device=g.device)

    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_g2t + loss_t2g)

def _contrast_one_view(img_p, tex_p, img_n, tex_n, sub_all, tau=0.1, eps=1e-8):
    
    device = img_p.device

    if sub_all.size(0) == 0:
        return torch.tensor(0.0, device=device)

    def l2norm(x):
        return F.normalize(x, dim=1) if x.numel() > 0 else x

    img_p_n = l2norm(img_p)     # (Np, D)
    tex_p_n = l2norm(tex_p)     # (Np, D)
    img_n_n = l2norm(img_n)     # (Nn, D)
    tex_n_n = l2norm(tex_n)     # (Nn, D)
    sub_n   = l2norm(sub_all)   # (B,  D)

    if img_p_n.size(0) > 0 and tex_p_n.size(0) > 0:
        sim_neg_p = img_p_n @ sub_n.T                      # (Np, B)
        sum_low_p = torch.exp(sim_neg_p / tau).sum(dim=1, keepdim=True)  # (Np, 1)

       
        sim_pos_p = img_p_n @ tex_p_n.T                    # (Np, Np)
        prob_p = torch.exp(sim_pos_p / tau) / (sum_low_p + eps)         # (Np, Np)
        log_prob_p = -torch.log(prob_p + eps)                              

        loss_p_each = log_prob_p.mean(dim=1)               # (Np,)
        loss_p = loss_p_each.sum() / (img_p_n.size(0) + 1.0)
    else:
        loss_p = torch.tensor(0.0, device=device)

    if img_n_n.size(0) > 0 and tex_n_n.size(0) > 0:
        sim_neg_n = img_n_n @ sub_n.T                      # (Nn, B)
        sum_low_n = torch.exp(sim_neg_n / tau).sum(dim=1, keepdim=True)  # (Nn, 1)

        sim_pos_n = img_n_n @ tex_n_n.T                    # (Nn, Nn)
        prob_n = torch.exp(sim_pos_n / tau) / (sum_low_n + eps)
        log_prob_n = -torch.log(prob_n + eps)

        loss_n_each = log_prob_n.mean(dim=1)               # (Nn,)
        loss_n = loss_n_each.sum() / (img_n_n.size(0) + 1.0)
    else:
        loss_n = torch.tensor(0.0, device=device)

    return loss_p + loss_n

def contrastive_loss_two_views(x_out, sub_out, x_out2, sub_out2, labels, tau=0.1, eps=1e-8):
   
    idx_pos = (labels == 1)
    idx_neg = (labels == 0)

    # view1
    img_p  = x_out[idx_pos]
    tex_p  = sub_out[idx_pos]
    img_n  = x_out[idx_neg]
    tex_n  = sub_out[idx_neg]

    # view2
    img_p2 = x_out2[idx_pos]
    tex_p2 = sub_out2[idx_pos]
    img_n2 = x_out2[idx_neg]
    tex_n2 = sub_out2[idx_neg]

    loss_view1 = _contrast_one_view(
        img_p=img_p,
        tex_p=tex_p,
        img_n=img_n,
        tex_n=tex_n,
        sub_all=sub_out,
        tau=tau,
        eps=eps,
    )

    loss_view2 = _contrast_one_view(
        img_p=img_p2,
        tex_p=tex_p2,
        img_n=img_n2,
        tex_n=tex_n2,
        sub_all=sub_out2,
        tau=tau,
        eps=eps,
    )

   
    loss = loss_view1  
    return loss


def train_mt(model, loader, cls_labels_fold, fi_fold,
             optimizer, device,
             roi_pt_dim, sub_pt_dim, node_pt_dim,
             class_weights):
    model.train()
    total_loss = 0.0
    total_num = 0
    correct = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(
            data.x, data.edge_index, data.edge_weight, data.batch,
            device, roi_pt_dim, sub_pt_dim, node_pt_dim
        )
        # 0: x1 
        # 1: sub_pt1 
        # 2: x2 
        # 3: sub_pt2 
        # 4: x_out1
        # 5: sub_out1
        # 6: x_out2
        # 7: sub_out2
        # 8: perm1
        # 9: score1
        # 10: perm2
        # 11: score2
        # 12: att  (cross-attention)
        fi_pred_global = output[0].squeeze(-1)   # (B,)
        fi_pred_subj   = output[1].squeeze(-1)   # (B,)
        logits         = output[2]               # (B, 2)
        logits2        = output[3]               # (B, 2)


        y_cls = cls_labels_fold[data.y].long()   # ADHD 
        y_fi  = fi_fold[data.y].float()          # FI 

        
        loss_cls = F.cross_entropy(logits, y_cls, weight=class_weights)
        loss_cls2 = F.cross_entropy(logits2, y_cls, weight=class_weights)

        
        fi_pred = 0.5 * (fi_pred_global + fi_pred_subj)
        loss_reg = F.mse_loss(fi_pred, y_fi)

        if len(output) > 7:
            x_out1 = output[4]   # (B, D)
            sub_out1 = output[5]
            x_out2 = output[6]
            sub_out2 = output[7]

            loss_con = contrastive_loss_two_views(
                x_out1, sub_out1,
                x_out2, sub_out2,
                labels=y_cls,   
                tau=tau
            )
        else:
            loss_con = torch.tensor(0.0, device=device)


        loss = lambda_cls * loss_cls + lambda_cls * loss_cls2 + lambda_reg * loss_reg + lambda_con * loss_con
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_cls.size(0)
        total_num  += y_cls.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y_cls).sum().item()

    return total_loss / total_num, correct / total_num


@torch.no_grad()
def eval_mt(model, loader, cls_labels_fold, fi_fold,
            device, roi_pt_dim, sub_pt_dim, node_pt_dim,
            class_weights=None):
   
    model.eval()
    total_loss = 0.0
    total_num  = 0

    all_preds_cls = []
    all_labels_cls = []

    all_preds_fi = []
    all_labels_fi = []

    for data in loader:
        data = data.to(device)

        output = model(
            data.x, data.edge_index, data.edge_weight, data.batch,
            device, roi_pt_dim, sub_pt_dim, node_pt_dim
        )
        fi_pred_global = output[0].squeeze(-1)
        fi_pred_subj   = output[1].squeeze(-1)
        logits         = output[2]

        y_cls = cls_labels_fold[data.y].long()
        y_fi  = fi_fold[data.y].float()

       
        if class_weights is not None:
            loss_cls = F.cross_entropy(logits, y_cls, weight=class_weights)
        else:
            loss_cls = F.cross_entropy(logits, y_cls)


        fi_pred = 0.5 * (fi_pred_global + fi_pred_subj)
        loss_reg = F.mse_loss(fi_pred, y_fi)

        loss = lambda_cls * loss_cls + lambda_reg * loss_reg

        total_loss += loss.item() * y_cls.size(0)
        total_num  += y_cls.size(0)


        preds_cls = logits.argmax(dim=1)
        all_preds_cls.append(preds_cls.cpu())
        all_labels_cls.append(y_cls.cpu())


        all_preds_fi.append(fi_pred.cpu())
        all_labels_fi.append(y_fi.cpu())

    all_preds_cls = torch.cat(all_preds_cls).numpy()
    all_labels_cls = torch.cat(all_labels_cls).numpy()

    all_preds_fi = torch.cat(all_preds_fi).numpy()
    all_labels_fi = torch.cat(all_labels_fi).numpy()

    acc = accuracy_score(all_labels_cls, all_preds_cls)
    avg_loss = total_loss / total_num

    mse = np.mean((all_preds_fi - all_labels_fi) ** 2)
    mae = np.mean(np.abs(all_preds_fi - all_labels_fi))
    try:
        corr, _ = pearsonr(all_preds_fi, all_labels_fi)
    except Exception:
        corr = np.nan

    return avg_loss, acc, mse, mae, corr, all_labels_cls, all_preds_cls


@torch.no_grad()
def collect_attentions(model, loader, device,
                       roi_pt_dim, sub_pt_dim, node_pt_dim):
    
    model.eval()

    all_att = []
    all_perm1 = []
    all_score1 = []
    all_perm2 = []
    all_score2 = []

    for data in loader:
        data = data.to(device)
        output = model(
            data.x, data.edge_index, data.edge_weight, data.batch,
            device, roi_pt_dim, sub_pt_dim, node_pt_dim
        )

        perm1 = score1 = perm2 = score2 = att = None

        if len(output) > 8:
            perm1 = output[8]
            score1 = output[9]
        if len(output) > 10:
            perm2 = output[10]
            score2 = output[11]
        if len(output) > 12:
            att = output[12]

        if att is not None:
            all_att.append(att.detach().cpu().numpy())
        if perm1 is not None and score1 is not None:
            all_perm1.append(perm1.detach().cpu().numpy())
            all_score1.append(score1.detach().cpu().numpy())
        if perm2 is not None and score2 is not None:
            all_perm2.append(perm2.detach().cpu().numpy())
            all_score2.append(score2.detach().cpu().numpy())

    def cat_or_none(lst):
        if len(lst) == 0:
            return None
        return np.concatenate(lst, axis=0)

    return {
        "att":   cat_or_none(all_att),
        "perm1": cat_or_none(all_perm1),
        "score1": cat_or_none(all_score1),
        "perm2": cat_or_none(all_perm2),
        "score2": cat_or_none(all_score2),
    }


def main():
    print("Loading data ...")

    all_data, all_labels = get_data(connpath, labelpath)
    all_labels[all_labels>0] = 1
    all_labels = all_labels.astype(int)

    _, all_score5 = get_data(connpath, scorepath5)
    _, all_score6 = get_data(connpath, scorepath6)
    _, all_score7 = get_data(connpath, scorepath7)
    _, all_score8 = get_data(connpath, scorepath8)
    all_FI = (all_score5 + all_score6 + all_score8)/3 # 
    all_FI = all_FI.astype(float)

    roi_pt  = torch.stack(torch.load(roi_pt_path,  map_location="cpu"))
    roi_pt2 = torch.stack(torch.load(roi_pt2_path, map_location="cpu"))
    sub_pt  = torch.stack(torch.load(sub_pt_path,  map_location="cpu"))
    node_pt = torch.stack(torch.load(node_pt_path, map_location="cpu"))

    roi_pt_dim  = roi_pt.shape[1]
    roi_pt_dim2 = roi_pt2.shape[1]
    sub_pt_dim  = sub_pt.shape[1]

    print("ROI prompt dim:", roi_pt_dim, roi_pt_dim2)
    print("Subject prompt dim:", sub_pt_dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold = 0
    fold_accs = []
    fold_corrs = []
    fold_mses = []
    fold_maes = []

    for train_idx, test_idx in skf.split(all_data, all_labels):
        fold += 1
        print(f"\n==== Fold {fold}/{n_splits} ====")

        train_data = all_data[train_idx]
        test_data  = all_data[test_idx]

        train_labels_np = all_labels[train_idx]
        test_labels_np  = all_labels[test_idx]
        train_FI_raw = all_FI[train_idx]
        test_FI_raw  = all_FI[test_idx]

        scaler = preprocessing.StandardScaler().fit(train_FI_raw.reshape(-1, 1))
        train_FI = scaler.transform(train_FI_raw.reshape(-1, 1))[:, 0]
        test_FI  = scaler.transform(test_FI_raw.reshape(-1, 1))[:, 0]

        train_sub_pt = sub_pt[train_idx]
        test_sub_pt  = sub_pt[test_idx]

        index_train = np.arange(len(train_idx)).reshape(-1, 1)
        index_test  = np.arange(len(test_idx)).reshape(-1, 1)

        training_dataset = create_dataset_pt3(train_data, index_train, kk,
                                              roi_pt, roi_pt2, node_pt, train_sub_pt, thres)
        testing_dataset  = create_dataset_pt3(test_data, index_test, kk,
                                              roi_pt, roi_pt2, node_pt, test_sub_pt, thres)

        train_loader = DataLoader(training_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=False)
        test_loader  = DataLoader(testing_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=False)

        train_labels_fold = torch.tensor(train_labels_np, dtype=torch.long).to(device)
        test_labels_fold  = torch.tensor(test_labels_np, dtype=torch.long).to(device)

        train_FI_fold = torch.tensor(train_FI, dtype=torch.float32).to(device)
        test_FI_fold  = torch.tensor(test_FI, dtype=torch.float32).to(device)

        class_weights = compute_class_weights(train_labels_np, device)
        print("Class weights:", class_weights)

        model = GPS_prompt_mt_2roi(hidden_channels, hidden_channels2,
                                   hc, hc2, pt_dim, pt_dim2,
                                   roi_pt_dim, sub_pt_dim, node_pt_dim)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr, weight_decay=weight_decay)

        best_epoch = 0
        best_acc = 0.0
        best_corr = -1.0  

        best_y_true = None
        best_y_pred = None

        best_mse = None
        best_mae = None

        for epoch in range(1, epoch_num + 1):
            train_loss, train_acc = train_mt(
                model, train_loader,
                train_labels_fold, train_FI_fold,
                optimizer, device,
                roi_pt_dim, sub_pt_dim, node_pt_dim,
                class_weights
            )

            val_loss, val_acc, val_mse, val_mae, val_corr, y_true, y_pred = eval_mt(
                model, test_loader,
                test_labels_fold, test_FI_fold,
                device,
                roi_pt_dim, sub_pt_dim, node_pt_dim,
                class_weights
            )

            if val_acc + 0.5*val_corr > best_acc + 0.5*best_corr :
                best_corr = val_corr
                best_acc = val_acc
                best_epoch = epoch
                best_y_true = y_true
                best_y_pred = y_pred
                best_mse = val_mse
                best_mae = val_mae

        print(f"Fold {fold} ")
        print(f"  acc:  {best_acc:.3f}")
        print(f"  corr: {best_corr:.3f}")
        print(f"  MSE:  {best_mse:.4f}, MAE: {best_mae:.4f}")
        print("Classification report:")
        print(classification_report(best_y_true, best_y_pred, digits=4))

        fold_accs.append(best_acc)
        fold_corrs.append(best_corr)
        fold_mses.append(best_mse)
        fold_maes.append(best_mae)

       
        att_train = collect_attentions(
            model, train_loader, device,
            roi_pt_dim, sub_pt_dim, node_pt_dim
        )
        
        att_test = collect_attentions(
            model, test_loader, device,
            roi_pt_dim, sub_pt_dim, node_pt_dim
        )

        if att_train["att"] is not None:
            write_attention2(timefile, att_train["att"], fold, task="_att_train")
        if att_test["att"] is not None:
            write_attention2(timefile, att_test["att"], fold, task="_att_test")

        if att_train["perm1"] is not None and att_train["score1"] is not None:
            write_attention3(timefile,
                             att_train["perm1"],
                             att_train["score1"],
                             fold,
                             task="_pool1_train")
        if att_test["perm1"] is not None and att_test["score1"] is not None:
            write_attention3(timefile,
                             att_test["perm1"],
                             att_test["score1"],
                             fold,
                             task="_pool1_test")

        if att_train["perm2"] is not None and att_train["score2"] is not None:
            write_attention3(timefile,
                             att_train["perm2"],
                             att_train["score2"],
                             fold,
                             task="_pool2_train")
        if att_test["perm2"] is not None and att_test["score2"] is not None:
            write_attention3(timefile,
                             att_test["perm2"],
                             att_test["score2"],
                             fold,
                             task="_pool2_test")

    #print("\n==== 10-fold summary (multi-task + contrastive) ====")
    print("Fold accuracies:", fold_accs)
    print("Mean acc:  {:.3f} ± {:.3f}".format(np.mean(fold_accs), np.std(fold_accs)))

    print("Fold correlations:", fold_corrs)
    print("Mean corr: {:.3f} ± {:.3f}".format(np.nanmean(fold_corrs), np.nanstd(fold_corrs)))

    print("Fold MSEs:", fold_mses)
    print("Mean MSE:  {:.4f} ± {:.4f}".format(np.mean(fold_mses), np.std(fold_mses)))

    print("Fold MAEs:", fold_maes)
    print("Mean MAE:  {:.4f} ± {:.4f}".format(np.mean(fold_maes), np.std(fold_maes)))

    

if __name__ == "__main__":
    main()
