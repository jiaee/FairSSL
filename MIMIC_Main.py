#!/usr/bin/env python3
import os
import random
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import wandb
from DataSets.MIMIC import load_split, drop_unnecessary, median_impute, SameLabelBatchSampler
from Models.Basic_ConvNet import ConvEncoder, LinearHead
from Utils.metric_calculate import calculate_metrics
from SSL.VICReg import VICReg_M1, VICReg_M234
from Utils.config import is_configured, load_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = load_yaml_config(PROJECT_ROOT / "configs" / "mimic.yaml")
PATH_CONFIG = CONFIG.get("paths", {})
DEFAULT_CONFIG = CONFIG.get("defaults", {})

WANDB_API_KEY = CONFIG.get("wandb", {}).get("api_key")
if is_configured(WANDB_API_KEY):
    wandb.login(key=WANDB_API_KEY)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

TRAIN_PATH = PATH_CONFIG.get("train_path")
VAL_PATH = PATH_CONFIG.get("val_path")
TEST_PATH = PATH_CONFIG.get("test_path")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, default=TRAIN_PATH)
    parser.add_argument('--val-path',   type=str, default=VAL_PATH)
    parser.add_argument('--test-path',  type=str, default=TEST_PATH)
    parser.add_argument('--seed',       type=int, default=DEFAULT_CONFIG.get("seed", 42))
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG.get("batch_size", 32))
    parser.add_argument('--mlp', type=str, default=DEFAULT_CONFIG.get("mlp", "512-256-1024"))
    parser.add_argument('--epochs-ssl', type=int, default=DEFAULT_CONFIG.get("epochs_ssl", 10))
    parser.add_argument('--lr-ssl',     type=float, default=DEFAULT_CONFIG.get("lr_ssl", 1e-4))
    parser.add_argument('--sim-coeff',  type=float, default=DEFAULT_CONFIG.get("sim_coeff", 25.0))
    parser.add_argument('--std-coeff',  type=float, default=DEFAULT_CONFIG.get("std_coeff", 25.0))
    parser.add_argument('--cov-coeff',  type=float, default=DEFAULT_CONFIG.get("cov_coeff", 1.0))
    parser.add_argument('--epochs-ft',  type=int, default=DEFAULT_CONFIG.get("epochs_ft", 10))
    parser.add_argument('--lr-ft',      type=float, default=DEFAULT_CONFIG.get("lr_ft", 0.03))
    parser.add_argument('--log-every',  type=int, default=DEFAULT_CONFIG.get("log_every", 20))
    parser.add_argument('--loss-family', type=str,
                        choices=['vicreg','fairwell'],
                        default=DEFAULT_CONFIG.get("loss_family", "vicreg"))
    parser.add_argument('--loss-type', type=str,
                        choices=['M1','M2','M3','M4'],
                        default=DEFAULT_CONFIG.get("loss_type", "M1"))
    args = parser.parse_args()
    
    wandb.init(config=vars(args), reinit=True)

    config = wandb.config
    args.lr_ssl    = config.lr_ssl
    args.lr_ft     = config.lr_ft
    args.sim_coeff = config.sim_coeff
    args.std_coeff = config.std_coeff
    args.cov_coeff = config.cov_coeff
    args.epochs_ssl = config.epochs_ssl
    args.epochs_ft  = config.epochs_ft
    args.batch_size = config.batch_size
    args.mlp        = config.mlp 
    args.loss_family = str(wandb.config.loss_family).lower()
    args.loss_type   = wandb.config.loss_type
    args.seed = int(config.seed)

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Data loading & preprocessing
    train_ds = load_split(args.train_path)
    val_ds   = load_split(args.val_path)
    test_ds  = load_split(args.test_path)
    train_ds, val_ds, test_ds = drop_unnecessary(train_ds, val_ds, test_ds)
    train_ds, med = median_impute(train_ds)
    val_ds, _   = median_impute(val_ds, med)
    test_ds, _  = median_impute(test_ds, med)
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        X, _, _, _ = ds.tensors
        assert not torch.isnan(X).any(), f"{name} still has NaNs!"
    
    train_loader_sl = DataLoader(                       # homogeneous batches only
        train_ds,
        batch_sampler=SameLabelBatchSampler(train_ds, args.batch_size)
        )
    train_loader_ml = DataLoader(
        train_ds, batch_size=args.batch_size, 
        shuffle=True,  
        drop_last=True
        )
    
    # validation / test unchanged
    val_loader  = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Infer input channels from training data
    in_channels = train_ds.tensors[0].shape[1]
    
    # Initialize models and assign them to the device (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = ConvEncoder(in_channels=in_channels).to(device)
    opt_ssl = None

    # Infer embedding dim once
    with torch.no_grad():
        dummy = torch.zeros(args.batch_size, *train_ds.tensors[0].shape[1:], device=device)[:1]
        h_dummy = ConvEncoder(in_channels=train_ds.tensors[0].shape[1]).to(device)(dummy)
        emb_dim = h_dummy.shape[-1]

    family  = args.loss_family.lower()
    if family == 'fairwell':
        family = 'vicreg'
    if family != 'vicreg':
        raise ValueError(f"Unsupported FairWell loss family: {args.loss_family}")
    variant = args.loss_type.upper()

    if variant == 'M1':
        loss_fn = VICReg_M1(args)
    elif variant in {'M2', 'M3', 'M4'}:
        loss_fn = VICReg_M234(emb_dim, emb_dim, args)
    else:
        raise ValueError(f"Unknown FairWell loss_type: {variant}")

    loss_fn = loss_fn.to(device)

    with torch.no_grad():
        B0 = 4
        sid0 = list(range(B0))
        _ = loss_fn(
            torch.randn(B0, emb_dim, device=device),
            torch.randn(B0, emb_dim, device=device),
            sid0,
            sid0
        )

    opt_ssl = torch.optim.Adam(
        list(encoder.parameters()) + [p for p in loss_fn.parameters() if p.requires_grad],
        lr=args.lr_ssl
    )

    # Self-Supervised Training
    for epoch in range(1, args.epochs_ssl + 1):
        if variant in {'M1','M2'}:
            cur_loader = train_loader_ml
        elif variant == 'M3':
            cur_loader = train_loader_sl
        elif variant == 'M4':
            cur_loader = train_loader_sl if epoch % 2 == 1 else train_loader_ml
        else:
            raise ValueError(f"Unknown loss_type: {variant}")
        
        encoder.train()
        total_loss = 0.0
        for X, _, _, subj_ids in cur_loader:
            X = X.to(device)
            x1 = X.clone()
            x1[:, 8:, :] = 0
            x2 = X.clone()
            x2[:, :8, :] = 0
            h1 = encoder(x1)
            h2 = encoder(x2)
            subject_ids = subj_ids.detach().cpu().tolist() if torch.is_tensor(subj_ids) else list(subj_ids)

            loss = loss_fn(h1, h2, subject_ids, subject_ids)

            opt_ssl.zero_grad()
            loss.backward()
            opt_ssl.step()
            total_loss += loss.item()
        if epoch % args.log_every == 0:
            avg_loss = total_loss / len(cur_loader)
            print(f"[SSL] Epoch {epoch:03d} VICReg loss: {avg_loss:.4f}")
            wandb.log({"ssl_epoch": epoch, "ssl_loss": avg_loss})
    
    # Linear Probing
    # Freeze encoder & train only projection head
    for p in encoder.parameters():
        p.requires_grad = False

    linear_head = LinearHead(dim_in=emb_dim).to(device)
    opt_ft = torch.optim.Adadelta(linear_head.parameters(), lr=args.lr_ft)
    
    for epoch in range(1, args.epochs_ft + 1):
        linear_head.train()
        total_ce = 0.0
        for X, y, _, _ in train_loader_ml:
            X, y = X.to(device), y.to(device)
            h = encoder(X)
            logits = linear_head(h)
            loss = F.cross_entropy(logits, y)
            opt_ft.zero_grad()
            loss.backward()
            opt_ft.step()
            total_ce += loss.item()
        if epoch % args.log_every == 0:
            ce_loss = total_ce / len(train_loader_ml)
            print(f"[FT] Epoch {epoch:03d} CE loss: {ce_loss:.4f}")
            wandb.log({"ft_epoch": epoch, "ft_loss": ce_loss})
        
    encoder.eval()
    linear_head.eval()

    # on validation set (for tuning)
    all_probs_val, all_y_val, all_gender_val = [], [], []
    with torch.no_grad():
        for X, y, g, _ in val_loader:
            X = X.to(device)
            h = encoder(X)
            logits = linear_head(h).cpu()
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs_val.append(probs)
            all_y_val.append(y)
            all_gender_val.append(g)

    probs_val  = torch.cat(all_probs_val).numpy()
    y_true_val = torch.cat(all_y_val).numpy()
    gender_val = torch.cat(all_gender_val).numpy()
    mask_val   = ~np.isnan(probs_val)
    probs_val, y_true_val, gender_val = \
        probs_val[mask_val], y_true_val[mask_val], gender_val[mask_val]
    y_pred_val = (probs_val >= 0.5).astype(int)

    # on test set (for eval)
    all_probs_test, all_y_test, all_gender_test = [], [], []
    with torch.no_grad():
        for X, y, g, _ in test_loader:
            X = X.to(device)
            h = encoder(X)
            logits = linear_head(h).cpu()
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs_test.append(probs)
            all_y_test.append(y)
            all_gender_test.append(g)

    probs_test  = torch.cat(all_probs_test).numpy()
    y_true_test = torch.cat(all_y_test).numpy()
    gender_test = torch.cat(all_gender_test).numpy()
    mask_test   = ~np.isnan(probs_test)
    probs_test, y_true_test, gender_test = \
        probs_test[mask_test], y_true_test[mask_test], gender_test[mask_test]
    y_pred_test = (probs_test >= 0.5).astype(int)

    # compute metrics
    _, _, _, _, _, _, _, _, _, _, val_auc = calculate_metrics(
        y_pred_val, y_true_val, gender_val,
        y_scores=probs_val
    )
    SP, EOpp, EOdd, EAcc, Acc, Prec, Sens, Spec, F1, uar, auc = calculate_metrics(
        y_pred_test, y_true_test, gender_test,
        y_scores=probs_test
    )

    print(f"Validation AUC:     {val_auc:.4f}")
    print(f"Accuracy:           {Acc:.4f}")
    print(f"F1-score:           {F1:.4f}")
    print(f"Precision:          {Prec:.4f}")
    print(f"Sensitivity:        {Sens:.4f}")
    print(f"Specificity:        {Spec:.4f}")
    print(f"Statistical Parity: {SP:.4f}")
    print(f"Equal Opportunity:  {EOpp:.4f}")
    print(f"Equal Odds:         {EOdd:.4f}")
    print(f"Equal Accuracy:     {EAcc:.4f}")
    print(f"UAR:                {uar:.4f}")
    print(f"AUC:                {auc:.4f}")

    wandb.log({
        "val_auc": val_auc,
        "Accuracy": Acc,
        "F1 Score": F1,
        "Precision": Prec,
        "Sensitivity": Sens,
        "Specificity": Spec,
        "Statistical Parity": SP,
        "Equal Opportunity": EOpp,
        "Equal Odds": EOdd,
        "Equal Accuracy": EAcc,
        "UAR": uar,
        "AUC": auc,

    })

    encoder.eval()
    feat_dir = os.path.join(wandb.run.dir, "encoder_features")
    os.makedirs(feat_dir, exist_ok=True)
    feat_path = os.path.join(feat_dir, "test_encoder_features.npz")

    feat_m1, feat_m2 = [], []
    ys, gs, sids = [], [], []

    with torch.no_grad():
        for X, y, gender, sid in test_loader:
            X = X.to(device)
            X1 = X.clone(); X1[:, 8:] = 0.0
            X2 = X.clone(); X2[:, :8] = 0.0

            h1 = encoder(X1).detach().cpu().numpy()
            h2 = encoder(X2).detach().cpu().numpy()

            feat_m1.append(h1)
            feat_m2.append(h2)
            ys.append(y.numpy())
            gs.append(gender.numpy())

            if torch.is_tensor(sid):
                sids.extend(sid.cpu().numpy().tolist())
            else:
                sids.extend(list(sid))

    np.savez_compressed(
        feat_path,
        feat_mod1=np.vstack(feat_m1),
        feat_mod2=np.vstack(feat_m2),
        y_true=np.concatenate(ys),
        gender=np.concatenate(gs),
        subject_id=np.asarray(sids, dtype=object),
    )
    wandb.save(feat_path)

    wandb.finish()



if __name__ == "__main__":
    sweep_config = CONFIG.get("sweep", {})
    sweep_id = wandb.sweep(sweep_config, project=CONFIG.get("wandb", {}).get("project", "mimic_sweep"))
    wandb.agent(sweep_id, function=run)
