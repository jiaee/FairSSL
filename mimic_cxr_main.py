import os
import random
import csv
import json
import copy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from SSL.VICReg import VICReg_M1, VICReg_M234
from Utils.metric_calculate import calculate_metrics

from DataSets.MIMIC_CXR import build_mimic_cxr_loaders
from DataSets.MIMIC_CXR import RACE_GROUPS
from Utils.config import is_configured, load_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = load_yaml_config(PROJECT_ROOT / "configs" / "mimic_cxr.yaml")
SELECTED_CONFIG = CONFIG.get("selected", {})
PATH_CONFIG = CONFIG.get("paths", {})
WANDB_CONFIG = CONFIG.get("wandb", {})

SELECTED_DATASET = SELECTED_CONFIG.get("dataset", "mimic_cxr")
SELECTED_LOSS_FAMILY = SELECTED_CONFIG.get("loss_family", "vicreg")
SELECTED_LOSS_TYPE = SELECTED_CONFIG.get("loss_type", "M3")
SELECTED_POOL_MODALITY = SELECTED_CONFIG.get("pool_modality", "image")
RUN_AS_SWEEP = bool(SELECTED_CONFIG.get("run_as_sweep", True))

WANDB_API_KEY = WANDB_CONFIG.get("api_key")
WANDB_ENTITY = WANDB_CONFIG.get("entity") if is_configured(WANDB_CONFIG.get("entity")) else None
if is_configured(WANDB_API_KEY):
    wandb.login(key=WANDB_API_KEY, relogin=True)

MIMIC_DATA_DIR = PATH_CONFIG.get("mimic_root")
LOCAL_RESULTS_DIR = PATH_CONFIG.get("local_results_dir") if is_configured(PATH_CONFIG.get("local_results_dir")) else "mimic_cxr_results"


def _project_name_for(dataset_name: str, loss_family: str, loss_type: str) -> str:
    dataset_name = str(dataset_name).lower()
    loss_family = str(loss_family).lower()
    loss_type = str(loss_type).upper()
    projects = CONFIG.get("projects", {})
    return projects.get(loss_type, f"{loss_family}_{loss_type.lower()}_{dataset_name}")


def _default_config_for(dataset_name: str, loss_family: str, loss_type: str, pool_modality: str) -> dict:
    cfg = copy.deepcopy(CONFIG.get("defaults", {}))
    cfg.update(copy.deepcopy(PATH_CONFIG))
    cfg.update(
        dataset_name=str(dataset_name).lower(),
        loss_family=str(loss_family).lower(),
        loss_type=str(loss_type).upper(),
        pool_modality=str(pool_modality),
    )
    if cfg.get("pathology_labels_csv") == "":
        cfg["pathology_labels_csv"] = None
    return cfg


def _sweep_config_for(dataset_name: str, loss_family: str, loss_type: str, pool_modality: str) -> dict:
    dataset_name = str(dataset_name).lower()
    loss_family = str(loss_family).lower()
    loss_type = str(loss_type).upper()
    pool_modality = str(pool_modality)

    if dataset_name != "mimic_cxr":
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    if loss_family == "fairwell":
        loss_family = "vicreg"
    if loss_family != "vicreg":
        raise ValueError(f"Unsupported FairWell loss_family: {loss_family}")
    if loss_type not in {"M1", "M2", "M3", "M4"}:
        raise ValueError(f"Unsupported FairWell loss_type: {loss_type}")

    cfg = copy.deepcopy(CONFIG.get("sweep", {}))
    cfg.setdefault("parameters", {})
    cfg["parameters"]["loss_family"] = {"value": loss_family}
    cfg["parameters"]["loss_type"] = {"value": loss_type}
    cfg["parameters"]["pool_modality"] = {"value": pool_modality}
    return cfg


WANDB_PROJECT = _project_name_for(SELECTED_DATASET, SELECTED_LOSS_FAMILY, SELECTED_LOSS_TYPE)
DEFAULT_CONFIG_TEMPLATE = _default_config_for(SELECTED_DATASET, SELECTED_LOSS_FAMILY, SELECTED_LOSS_TYPE, SELECTED_POOL_MODALITY)
SWEEP_CONFIG = _sweep_config_for(SELECTED_DATASET, SELECTED_LOSS_FAMILY, SELECTED_LOSS_TYPE, SELECTED_POOL_MODALITY)


def _first_existing_path(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _resolve_text_cache_path(config) -> str:
    cfg_path = getattr(config, "text_cache_path", None)
    if not is_configured(cfg_path):
        raise FileNotFoundError("text_cache_path is not configured")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"text_cache_path does not exist: {cfg_path}")
    return cfg_path


def _default_mimic_path(config, key: str, leaf: str) -> str:
    value = getattr(config, key, None)
    if value:
        return value
    return os.path.join(getattr(config, "mimic_root", MIMIC_DATA_DIR), leaf)


def _csv_safe_value(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return json.dumps(v.detach().cpu().tolist())
    if isinstance(v, (list, tuple, dict, set)):
        try:
            return json.dumps(v, sort_keys=True)
        except TypeError:
            return str(v)
    if v is None:
        return ""
    return v


def _results_csv_path(config) -> str:
    dataset_name = str(getattr(config, "dataset_name", SELECTED_DATASET)).lower()
    loss_family = str(getattr(config, "loss_family", SELECTED_LOSS_FAMILY)).lower()
    loss_type = str(getattr(config, "loss_type", SELECTED_LOSS_TYPE)).upper()

    Path(LOCAL_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    filename = f"{dataset_name}__{loss_family}__{loss_type}__results.csv"
    return os.path.join(LOCAL_RESULTS_DIR, filename)


def _append_row_to_csv(csv_path: str, row: dict):
    row = {str(k): _csv_safe_value(v) for k, v in row.items()}
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_rows = list(reader)
        old_fields = reader.fieldnames or []

    all_fields = list(old_fields)
    for k in row.keys():
        if k not in all_fields:
            all_fields.append(k)

    old_rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for r in old_rows:
            writer.writerow({field: r.get(field, "") for field in all_fields})


def _save_final_results_row(config, final_metrics: dict):
    csv_path = _results_csv_path(config)

    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "dataset_name": str(getattr(config, "dataset_name", SELECTED_DATASET)),
        "loss_family": str(getattr(config, "loss_family", "")),
        "loss_type": str(getattr(config, "loss_type", "")),
        "wandb_project": WANDB_PROJECT,
        "wandb_run_id": getattr(wandb.run, "id", ""),
        "wandb_run_name": getattr(wandb.run, "name", ""),
        "wandb_run_url": getattr(wandb.run, "url", ""),
    }

    for k, v in vars(config).items():
        row[f"cfg__{k}"] = v

    for k, v in final_metrics.items():
        row[f"metric__{k}"] = v

    _append_row_to_csv(csv_path, row)
    print(f"Saved local results row to: {csv_path}")

# -----------------------------
# Seeding
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Metric Helpers
# -----------------------------

def _safe_mean(vals):
    vals = [float(v) for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _compute_binary_fairness_metrics(y_pred, y_labels, sensitive, y_scores=None):
    SP, EOpp, EOdd, EAcc, *_ = calculate_metrics(
        y_pred=y_pred,
        y_labels=y_labels,
        gender_list=sensitive,
        y_scores=y_scores,
        linearprobing=True,
    )
    return {
        "SP": float(SP),
        "EOpp": float(EOpp),
        "EOdd": float(EOdd),
        "EAcc": float(EAcc),
    }


def _compute_pairwise_race_fairness(y_pred, y_labels, race_list, y_scores=None, race_groups=None):
    from itertools import combinations

    race_groups = list(race_groups) if race_groups is not None else list(RACE_GROUPS)
    race_arr = np.asarray(race_list).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    y_labels = np.asarray(y_labels).astype(np.int64)
    y_scores = None if y_scores is None else np.asarray(y_scores, dtype=np.float64)

    pairwise = {}
    agg = {"SP": [], "EOpp": [], "EOdd": [], "EAcc": []}

    for a, b in combinations(range(len(race_groups)), 2):
        mask = np.logical_or(race_arr == a, race_arr == b)
        if mask.sum() == 0:
            continue

        pair_sensitive = np.where(race_arr[mask] == a, 0, 1).astype(np.int64)
        if np.unique(pair_sensitive).size < 2:
            continue

        pair_scores = None if y_scores is None else y_scores[mask]
        metrics = _compute_binary_fairness_metrics(
            y_pred=y_pred[mask].tolist(),
            y_labels=y_labels[mask].tolist(),
            sensitive=pair_sensitive.tolist(),
            y_scores=None if pair_scores is None else pair_scores.tolist(),
        )

        pair_name = f"{race_groups[a]}_vs_{race_groups[b]}"
        pairwise[pair_name] = metrics
        for k, v in metrics.items():
            agg[k].append(v)

    aggregated = {f"{k}_race_aggregated": _safe_mean(v) for k, v in agg.items()}
    return pairwise, aggregated


def _print_group_fairness(prefix, metrics_dict):
    print(f"\n[{prefix}] fairness metrics")
    for k, v in metrics_dict.items():
        print(f"{k}: {v}")


def _print_pairwise_race_fairness(pairwise_metrics, aggregated_metrics):
    print("\n[RACE] pairwise fairness metrics")
    if not pairwise_metrics:
        print("No valid race pairs found.")
    for pair_name, metrics in pairwise_metrics.items():
        pretty = ", ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"{pair_name}: {pretty}")

    print("\n[RACE] aggregated fairness metrics")
    for k, v in aggregated_metrics.items():
        print(f"{k}: {v}")

# -----------------------------
class CXRTextBackbone(nn.Module):
    """A minimal dual-encoder.

    - Image encoder: ResNet18 adapted to 1-channel.
    - Text encoder: token embedding + transformer encoder.

    It exposes: forward(images, input_ids, attn_mask, return_features=True)
      -> (img_emb, txt_emb)
    """

    def __init__(
        self,
        proj_dim: int = 256,
        txt_vocab_size: int = 49408,
        txt_embed_dim: int = 256,
        txt_num_layers: int = 2,
        txt_num_heads: int = 4,
        txt_ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Image encoder
        import torchvision.models as tvm

        resnet = tvm.resnet18(weights=None)
        # adapt first conv to grayscale
        old = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            1,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        self.img_backbone = resnet
        img_out_dim = resnet.fc.in_features
        self.img_backbone.fc = nn.Identity()

        self.img_proj = nn.Sequential(
            nn.LayerNorm(img_out_dim),
            nn.Linear(img_out_dim, proj_dim),
        )

        # Text encoder
        self.tok_emb = nn.Embedding(txt_vocab_size, txt_embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=txt_embed_dim,
            nhead=txt_num_heads,
            dim_feedforward=txt_ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.txt_encoder = nn.TransformerEncoder(enc_layer, num_layers=txt_num_layers)
        self.txt_proj = nn.Sequential(
            nn.LayerNorm(txt_embed_dim),
            nn.Linear(txt_embed_dim, proj_dim),
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B,1,224,224) float
        feat = self.img_backbone(images)
        return self.img_proj(feat)

    def encode_text(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: (B,L), attn_mask: (B,L)
        x = self.tok_emb(input_ids)
        # transformer expects src_key_padding_mask with True on PAD positions
        key_padding = (attn_mask == 0)
        h = self.txt_encoder(x, src_key_padding_mask=key_padding)
        # mean pool on non-pad
        mask = attn_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.txt_proj(pooled)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        return_features: bool = False,
    ):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attn_mask)
        if return_features:
            return img_emb, txt_emb
        return img_emb, txt_emb


# -----------------------------
# Training utilities
# -----------------------------

@torch.no_grad()
def _normalize_img_u8_to_float(img_u8: torch.Tensor) -> torch.Tensor:
    # img_u8: uint8 in [0,255]
    return (img_u8.float() / 255.0)


def train_one_epoch_mimic_cxr(
    model,
    dataloader,
    optimizer,
    loss_fn,
    loss_family: str,
    loss_type: str,
    device,
    pool_modality: str = "image",
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    family = str(loss_family).lower()
    if family == "fairwell":
        family = "vicreg"
    variant = str(loss_type).upper()
    if family != "vicreg" or variant not in {"M1", "M2", "M3", "M4"}:
        raise ValueError(f"Unsupported FairWell loss: {loss_family}/{loss_type}")

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        if isinstance(batch, dict):
            if family not in {"vicreg", "fairwell"}:
                raise ValueError(
                    f"Got BAG (dict) batches, but loss_family='{loss_family}'. "
                    "BAG mode is only supported here for VICReg/FairWell."
                )

            img_u8 = batch["img"]            # (N_img,1,H,W) uint8
            img_sid = batch["img_sid"]       # (N_img,) long
            txt_ids = batch["txt_ids"]       # (N_txt,L) long
            txt_mask = batch["txt_mask"]     # (N_txt,L) long
            txt_sid = batch["txt_sid"]       # (N_txt,) long

            # Empty-batch guard (robustness)
            if img_u8.numel() == 0 or txt_ids.numel() == 0:
                continue

            images = _normalize_img_u8_to_float(img_u8).to(device, non_blocking=True)
            img_sid = img_sid.to(device, non_blocking=True).long()

            txt_ids = txt_ids.to(device, non_blocking=True)
            txt_mask = txt_mask.to(device, non_blocking=True)
            txt_sid = txt_sid.to(device, non_blocking=True).long()

            # Encode independently
            z_img = base_model.encode_image(images)
            z_txt = base_model.encode_text(txt_ids, txt_mask)

            if pool_modality not in {"image", "img", "text", "txt"}:
                raise ValueError(
                    f"pool_modality must be one of {{'image','img','text','txt'}}, got: {pool_modality}"
                )

            if pool_modality in {"text", "txt"}:
                z_a, z_b = z_txt, z_img
                sid_a, sid_b = txt_sid, img_sid
            else:
                z_a, z_b = z_img, z_txt
                sid_a, sid_b = img_sid, txt_sid

            loss = loss_fn(z_a, z_b, sid_a, sid_b)

        else:
            (
                img_u8,
                ids1,
                mask1,
                _len1,
                ids2,
                mask2,
                _len2,
                _gender,
                _race,
                _age_bin,
                _label,
                subj_ids,
            ) = batch

            # Empty-batch guard (collate can return empties in rare cases)
            if isinstance(img_u8, torch.Tensor) and img_u8.numel() == 0:
                continue

            images = _normalize_img_u8_to_float(img_u8).to(device, non_blocking=True)
            ids1 = ids1.to(device, non_blocking=True)
            mask1 = mask1.to(device, non_blocking=True)
            ids2 = ids2.to(device, non_blocking=True)
            mask2 = mask2.to(device, non_blocking=True)

            z_img, z_txt = model(images, ids1, mask1, return_features=True)

            if pool_modality not in {"image", "img", "text", "txt"}:
                raise ValueError(
                    f"pool_modality must be one of {{'image','img','text','txt'}}, got: {pool_modality}"
                )
            if pool_modality in {"text", "txt"}:
                z_img, z_txt = z_txt, z_img

            sid = torch.as_tensor(subj_ids, dtype=torch.long, device=device)
            loss = loss_fn(z_img, z_txt, sid, sid)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def setup_linear_classifier(img_dim: int, txt_dim: int, device: torch.device, config):
    clf = nn.Sequential(nn.Linear(img_dim + txt_dim, 1)).to(device)

    probe_lr = float(getattr(config, "probe_lr", 1e-3))
    probe_wd = float(getattr(config, "probe_wd", 0.0))

    opt = optim.Adam(clf.parameters(), lr=probe_lr, weight_decay=probe_wd)
    return clf, opt


def linear_probe_mimic_cxr(backbone, train_loader, val_loader, device, config):
    base_model = backbone.module if isinstance(backbone, nn.DataParallel) else backbone

    def _pool_mean_by_sid(z: torch.Tensor, sid: torch.Tensor):
        # z: (N,D), sid: (N,)
        if z.numel() == 0:
            return z, sid
        uniq, inv = torch.unique(sid, sorted=False, return_inverse=True)
        sums = torch.zeros((uniq.numel(), z.size(1)), device=z.device, dtype=z.dtype)
        sums.index_add_(0, inv, z)
        cnt = torch.zeros((uniq.numel(),), device=z.device, dtype=z.dtype)
        cnt.index_add_(0, inv, torch.ones_like(inv, dtype=z.dtype))
        return sums / cnt.unsqueeze(1).clamp_min(1.0), uniq

    def _align_subject_rows(pooled: torch.Tensor, pooled_sids: torch.Tensor, subject_ids: torch.Tensor):
        # Return rows in the same order as subject_ids
        sid_to_idx = {int(s): i for i, s in enumerate(pooled_sids.detach().cpu().tolist())}
        idxs = []
        for s in subject_ids.detach().cpu().tolist():
            idxs.append(sid_to_idx.get(int(s), None))
        # If something is missing (shouldn't happen), fill zeros
        out = []
        for j in idxs:
            if j is None:
                out.append(torch.zeros((pooled.size(1),), device=pooled.device, dtype=pooled.dtype))
            else:
                out.append(pooled[j])
        return torch.stack(out, dim=0) if out else torch.empty(0, pooled.size(1), device=pooled.device, dtype=pooled.dtype)

    # Determine embedding dims
    for batch in train_loader:
        if isinstance(batch, dict):
            images = _normalize_img_u8_to_float(batch["img"]).to(device)
            txt_ids = batch["txt_ids"].to(device)
            txt_mask = batch["txt_mask"].to(device)

            with torch.no_grad():
                z_img = base_model.encode_image(images)
                z_txt = base_model.encode_text(txt_ids, txt_mask)
            img_dim = z_img.shape[1]
            txt_dim = z_txt.shape[1]
        else:
            img_u8, ids1, mask1, *_rest = batch
            images = _normalize_img_u8_to_float(img_u8).to(device)
            ids1 = ids1.to(device)
            mask1 = mask1.to(device)
            with torch.no_grad():
                z_img, z_txt = backbone(images, ids1, mask1, return_features=True)
            img_dim = z_img.shape[1]
            txt_dim = z_txt.shape[1]
        break

    clf, opt = setup_linear_classifier(img_dim, txt_dim, device, config)
    crit = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None

    probe_epochs = int(getattr(config, "probe_epochs", 5))
    for epoch in range(probe_epochs):
        clf.train()
        total_loss = 0.0

        for batch in train_loader:
            if isinstance(batch, dict):
                images = _normalize_img_u8_to_float(batch["img"]).to(device)
                img_sid = batch["img_sid"].to(device)

                txt_ids = batch["txt_ids"].to(device)
                txt_mask = batch["txt_mask"].to(device)
                txt_sid = batch["txt_sid"].to(device)

                subject_ids = batch["subject_ids"].to(device)
                labels = batch["label"].to(device).float()

                with torch.no_grad():
                    z_img = base_model.encode_image(images)
                    z_txt = base_model.encode_text(txt_ids, txt_mask)

                # subject-level features for probe (this is evaluation, not SSL training)
                img_pooled, img_sids = _pool_mean_by_sid(z_img, img_sid)
                txt_pooled, txt_sids = _pool_mean_by_sid(z_txt, txt_sid)

                img_feat = _align_subject_rows(img_pooled, img_sids, subject_ids)
                txt_feat = _align_subject_rows(txt_pooled, txt_sids, subject_ids)

                feats = torch.cat([img_feat, txt_feat], dim=1)
                logits = clf(feats).squeeze(1)
                loss = crit(logits, labels)

            else:
                img_u8, ids1, mask1, *_mid, labels, _subj_ids = batch
                images = _normalize_img_u8_to_float(img_u8).to(device)
                ids1 = ids1.to(device)
                mask1 = mask1.to(device)
                labels = labels.to(device).float()

                with torch.no_grad():
                    z_img, z_txt = backbone(images, ids1, mask1, return_features=True)
                feats = torch.cat([z_img, z_txt], dim=1)
                logits = clf(feats).squeeze(1)
                loss = crit(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().item())

        # Validation AUC
        clf.eval()
        y_scores, y_true, y_gender = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    images = _normalize_img_u8_to_float(batch["img"]).to(device)
                    img_sid = batch["img_sid"].to(device)

                    txt_ids = batch["txt_ids"].to(device)
                    txt_mask = batch["txt_mask"].to(device)
                    txt_sid = batch["txt_sid"].to(device)

                    subject_ids = batch["subject_ids"].to(device)
                    labels = batch["label"].to(device).float()
                    gender = batch["gender"].to(device)

                    z_img = base_model.encode_image(images)
                    z_txt = base_model.encode_text(txt_ids, txt_mask)

                    img_pooled, img_sids = _pool_mean_by_sid(z_img, img_sid)
                    txt_pooled, txt_sids = _pool_mean_by_sid(z_txt, txt_sid)

                    img_feat = _align_subject_rows(img_pooled, img_sids, subject_ids)
                    txt_feat = _align_subject_rows(txt_pooled, txt_sids, subject_ids)

                    feats = torch.cat([img_feat, txt_feat], dim=1)
                    logits = clf(feats).squeeze(1)

                    y_scores.extend(torch.sigmoid(logits).cpu().tolist())
                    y_true.extend(labels.cpu().tolist())
                    y_gender.extend(gender.cpu().tolist())

                else:
                    img_u8, ids1, mask1, *_mid, gender, _race, _age_bin, labels, _subj_ids = batch
                    images = _normalize_img_u8_to_float(img_u8).to(device)
                    ids1 = ids1.to(device)
                    mask1 = mask1.to(device)
                    labels = labels.to(device).float()

                    z_img, z_txt = backbone(images, ids1, mask1, return_features=True)
                    feats = torch.cat([z_img, z_txt], dim=1)
                    logits = clf(feats).squeeze(1)

                    y_scores.extend(torch.sigmoid(logits).cpu().tolist())
                    y_true.extend(labels.cpu().tolist())
                    y_gender.extend(gender.cpu().tolist())

        *_, auc = calculate_metrics(
            y_pred=[1 if s >= 0.5 else 0 for s in y_scores],
            y_labels=[int(t) for t in y_true],
            gender_list=y_gender,
            y_scores=y_scores,
            linearprobing=True,
        )

        wandb.log({
            "linear_probe_epoch": epoch + 1,
            "probe_loss": total_loss / max(len(train_loader), 1),
            "val_auc_probe": auc,
        })

        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)

    return best_val_auc, clf


@torch.no_grad()
def linear_probe_test_mimic_cxr(backbone, clf, test_loader, device):
    backbone.eval()
    clf.eval()

    base_model = backbone.module if isinstance(backbone, nn.DataParallel) else backbone

    def _pool_mean_by_sid(z: torch.Tensor, sid: torch.Tensor):
        # z: (N,D), sid: (N,)
        if z.numel() == 0:
            return z, sid
        uniq, inv = torch.unique(sid, sorted=False, return_inverse=True)
        sums = torch.zeros((uniq.numel(), z.size(1)), device=z.device, dtype=z.dtype)
        sums.index_add_(0, inv, z)
        cnt = torch.zeros((uniq.numel(),), device=z.device, dtype=z.dtype)
        cnt.index_add_(0, inv, torch.ones_like(inv, dtype=z.dtype))
        return sums / cnt.unsqueeze(1).clamp_min(1.0), uniq

    def _align_subject_rows(pooled: torch.Tensor, pooled_sids: torch.Tensor, subject_ids: torch.Tensor):
        sid_to_idx = {int(s): i for i, s in enumerate(pooled_sids.detach().cpu().tolist())}
        out = []
        for s in subject_ids.detach().cpu().tolist():
            j = sid_to_idx.get(int(s), None)
            if j is None:
                out.append(torch.zeros((pooled.size(1),), device=pooled.device, dtype=pooled.dtype))
            else:
                out.append(pooled[j])
        if not out:
            return torch.empty(0, pooled.size(1), device=pooled.device, dtype=pooled.dtype)
        return torch.stack(out, dim=0)

    preds = []
    labels = []
    probs = []
    genders = []
    races = []
    age_bins = []
    subject_ids_out = []

    for batch in test_loader:
        # -------------------------
        # BAG MODE: dict batch
        # -------------------------
        if isinstance(batch, dict):
            images = _normalize_img_u8_to_float(batch["img"]).to(device)
            img_sid = batch["img_sid"].to(device)

            txt_ids = batch["txt_ids"].to(device)
            txt_mask = batch["txt_mask"].to(device)
            txt_sid = batch["txt_sid"].to(device)

            subject_ids = batch["subject_ids"].to(device)
            y = batch["label"].to(device).float()
            gender = batch["gender"].to(device)
            race = batch["race"].to(device)
            age_bin = batch["age_bin"].to(device)

            z_img = base_model.encode_image(images) if images.numel() > 0 else torch.empty(0, 0, device=device)
            z_txt = base_model.encode_text(txt_ids, txt_mask) if txt_ids.numel() > 0 else torch.empty(0, 0, device=device)

            img_pooled, img_sids = _pool_mean_by_sid(z_img, img_sid)
            txt_pooled, txt_sids = _pool_mean_by_sid(z_txt, txt_sid)

            img_feat = _align_subject_rows(img_pooled, img_sids, subject_ids)
            txt_feat = _align_subject_rows(txt_pooled, txt_sids, subject_ids)

            feats = torch.cat([img_feat, txt_feat], dim=1)
            logits = clf(feats).squeeze(1)

            p = torch.sigmoid(logits).detach().cpu().numpy()
            y_hat = (p >= 0.5).astype(np.int64)

            preds.extend(y_hat.tolist())
            probs.extend(p.tolist())
            labels.extend(y.detach().cpu().numpy().astype(np.int64).tolist())
            genders.extend(gender.detach().cpu().numpy().astype(np.int64).tolist())
            races.extend(race.detach().cpu().numpy().astype(np.int64).tolist())
            age_bins.extend(age_bin.detach().cpu().numpy().astype(np.int64).tolist())
            subject_ids_out.extend(subject_ids.detach().cpu().numpy().astype(np.int64).tolist())

        # -------------------------
        # PAIRED MODE: tuple batch
        # -------------------------
        else:
            img_u8, ids1, mask1, *_mid, gender, race, age_bin, y, subj_ids = batch
            images = _normalize_img_u8_to_float(img_u8).to(device)
            ids1 = ids1.to(device)
            mask1 = mask1.to(device)

            img_z, txt_z = backbone(images, ids1, mask1, return_features=True)
            feats = torch.cat([img_z, txt_z], dim=1)
            logits = clf(feats).squeeze(1)

            p = torch.sigmoid(logits).cpu().numpy()
            y_hat = (p >= 0.5).astype(np.int64)

            preds.extend(y_hat.tolist())
            probs.extend(p.tolist())
            labels.extend(y.cpu().numpy().astype(np.int64).tolist())
            genders.extend(gender.cpu().numpy().astype(np.int64).tolist())
            races.extend(race.cpu().numpy().astype(np.int64).tolist())
            age_bins.extend(age_bin.cpu().numpy().astype(np.int64).tolist())
            subject_ids_out.extend(list(subj_ids))

    acc = float(np.mean((np.array(preds) == np.array(labels)).astype(np.float32))) if len(labels) else 0.0
    return acc, preds, labels, probs, genders, races, age_bins, subject_ids_out


# -----------------------------
# Main (wandb sweep-compatible)
# -----------------------------

def main_sweep():
    default_config = dict(DEFAULT_CONFIG_TEMPLATE)

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
    )

    print("run entity:", run.entity)
    print("run project:", run.project)
    print("run url:", run.url)

    merged_config = dict(default_config)
    merged_config.update(dict(wandb.config))
    config = SimpleNamespace(**merged_config)


    set_seed(int(getattr(config, "seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    family = str(getattr(config, "loss_family", "vicreg")).lower()
    if family == "fairwell":
        family = "vicreg"
    variant = str(getattr(config, "loss_type", "M1")).upper()
    if family != "vicreg" or variant not in {"M1", "M2", "M3", "M4"}:
        raise ValueError(f"Unsupported FairWell loss: {family}/{variant}")

    text_cache_path = _resolve_text_cache_path(config)

    train_loader_sl, val_loader_sl, train_loader_ml, val_loader_ml, test_loader, meta = build_mimic_cxr_loaders(
        loss_type=variant,
        loss_family=family,
        mimic_root=getattr(config, "mimic_root", MIMIC_DATA_DIR),
        official_subdir=getattr(config, "official_subdir", "official_data_iccv_final"),
        cxr_record_list_csv=_default_mimic_path(config, "cxr_record_list_csv", "cxr-record-list.csv"),
        train_aug_csv=_default_mimic_path(config, "train_aug_csv", "mimic_cxr_aug_train_70.csv"),
        val_aug_csv=_default_mimic_path(config, "val_aug_csv", "mimic_cxr_aug_val_15.csv"),
        test_aug_csv=_default_mimic_path(config, "test_aug_csv", "mimic_cxr_aug_test_15.csv"),
        admissions_csv=_default_mimic_path(config, "admissions_csv", "admissions.csv"),
        patients_csv=_default_mimic_path(config, "patients_csv", "patients.csv"),
        task=getattr(config, "task", "no_finding"),
        pathology_labels_csv=getattr(config, "pathology_labels_csv", None),
        uncertain_policy=getattr(config, "uncertain_policy", "zero"),
        age_threshold=float(getattr(config, "age_threshold", 60.0)),
        max_items_per_batch=int(getattr(config, "batch_size", 128)),
        max_items_per_subject=int(getattr(config, "max_items_per_subject", 0)),
        num_workers=int(getattr(config, "num_workers", 4)),
        pin_memory=bool(getattr(config, "pin_memory", torch.cuda.is_available())),
        text_cache_path=text_cache_path,
        cap_max_len=int(getattr(config, "cap_max_len", 77)),
        pairs_per_subject=getattr(config, "pairs_per_subject", "all"),
        max_pairs_per_subject=getattr(config, "max_pairs_per_subject", None),
    )

    # Backbone
    backbone = CXRTextBackbone(
        proj_dim=int(getattr(config, "proj_dim", 256)),
        txt_vocab_size=int(getattr(config, "txt_vocab_size", meta.get("vocab_size", 49408))),
        txt_embed_dim=int(getattr(config, "txt_embed_dim", 256)),
        txt_num_layers=int(getattr(config, "txt_num_layers", 2)),
        txt_num_heads=int(getattr(config, "txt_num_heads", 4)),
        txt_ff_dim=int(getattr(config, "txt_ff_dim", 512)),
        dropout=float(getattr(config, "dropout", 0.1)),
    ).to(device)

    if torch.cuda.device_count() > 1:
        backbone = nn.DataParallel(backbone)

    # Determine embedding dims via a dummy forward.
    with torch.no_grad():
        dummy_img = torch.zeros(2, 1, 224, 224, device=device)
        dummy_ids = torch.zeros(2, meta.get("cap_max_len", 77), dtype=torch.long, device=device)
        dummy_mask = torch.ones_like(dummy_ids)
        dim_img, dim_txt = backbone(dummy_img, dummy_ids, dummy_mask, return_features=True)
        dim_mod1 = dim_img.shape[-1]
        dim_mod2 = dim_txt.shape[-1]
        config.emb = int(dim_mod1)

    family = str(getattr(config, "loss_family", "vicreg")).lower()
    if family == "fairwell":
        family = "vicreg"
    variant = str(getattr(config, "loss_type", "M1")).upper()
    if family != "vicreg":
        raise ValueError(f"Unsupported FairWell loss_family: {family}")
    if variant == "M1":
        loss_fn = VICReg_M1(config)
    elif variant in {"M2", "M3", "M4"}:
        loss_fn = VICReg_M234(dim_mod1, dim_mod2, config)
    else:
        raise ValueError(f"Unsupported FairWell loss_type: {variant}")

    loss_fn = loss_fn.to(device)

    with torch.no_grad():
        B0 = 4
        sid0 = torch.arange(B0, dtype=torch.long, device=device)
        _ = loss_fn(
            torch.randn(B0, dim_mod1, device=device),
            torch.randn(B0, dim_mod2, device=device),
            sid0,
            sid0
        )

    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        list(backbone.parameters()) + loss_params,
        lr=float(getattr(config, "lr", 1e-4)),
        weight_decay=float(getattr(config, "wd", 1e-4)),
    )

    ssl_epochs = int(getattr(config, "ssl_epochs", getattr(config, "vicreg_epochs", 10)))

    for epoch in range(ssl_epochs):
        if variant in {"M1", "M2"}:
            cur_loader = train_loader_ml
            cur_val = val_loader_ml
        elif variant == "M3":
            cur_loader = train_loader_sl
            cur_val = val_loader_sl
        else:
            if (epoch + 1) % 2 == 1:
                cur_loader = train_loader_sl
                cur_val = val_loader_sl
            else:
                cur_loader = train_loader_ml
                cur_val = val_loader_ml
            

        # before iterating cur_loader
        if hasattr(cur_loader.dataset, "set_epoch"):
            cur_loader.dataset.set_epoch(epoch)

        # If using batch_sampler:
        bs = getattr(cur_loader, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch"):
            bs.set_epoch(epoch)

        loss_val = train_one_epoch_mimic_cxr(
            backbone,
            cur_loader,
            optimizer,
            loss_fn,
            family,
            variant,
            device,
            pool_modality=str(getattr(config, "pool_modality", "image")),
        )

        if variant in {"M1", "M2"}:
            loader_tag = "ML"
        elif variant == "M3":
            loader_tag = "SL"
        else:
            loader_tag = "SL" if (epoch + 1) % 2 == 1 else "ML"

        wandb.log({
            "ssl_epoch": epoch + 1,
            "ssl_loss": loss_val,
            "loader_type": loader_tag,
        })

        # (optional) track probe AUC
        if bool(getattr(config, "probe_every_epoch", False)):
            val_auc, _ = linear_probe_mimic_cxr(backbone, train_loader_ml, val_loader_ml, device, config)
            wandb.log({"val_auc_probe": val_auc})

    # Final linear probe + test
    val_auc, clf = linear_probe_mimic_cxr(backbone, train_loader_ml, val_loader_ml, device, config)
    wandb.log({"val_auc": val_auc})

    test_acc, preds, labels, probs, genders, races, age_bins, subject_ids = linear_probe_test_mimic_cxr(backbone, clf, test_loader, device)
    wandb.log({"test_acc": test_acc})

    SP, EOpp, EOdd, EAcc, Acc, Prec, Sens, Spec, F1, uar, auc = calculate_metrics(
        y_pred=preds,
        y_labels=labels,
        gender_list=genders,
        y_scores=probs,
    )

    age_metrics = _compute_binary_fairness_metrics(
        y_pred=preds,
        y_labels=labels,
        sensitive=age_bins,
        y_scores=probs,
    )

    race_pairwise_metrics, race_aggregated_metrics = _compute_pairwise_race_fairness(
        y_pred=preds,
        y_labels=labels,
        race_list=races,
        y_scores=probs,
        race_groups=RACE_GROUPS,
    )

    wandb.log({
        "test_accuracy": Acc,
        "test_precision": Prec,
        "test_sensitivity": Sens,
        "test_specificity": Spec,
        "test_f1": F1,
        "test_uar": uar,
        "test_auc": auc,
        "statistical_parity": SP,
        "equal_opportunity": EOpp,
        "equalized_odds": EOdd,
        "equal_accuracy": EAcc,
        "sp_age_bin": age_metrics["SP"],
        "equal_opportunity_age_bin": age_metrics["EOpp"],
        "equalized_odds_age_bin": age_metrics["EOdd"],
        "equal_accuracy_age_bin": age_metrics["EAcc"],
        **race_aggregated_metrics,
        **{
            f"race_pairwise/{pair}/{metric}": value
            for pair, vals in race_pairwise_metrics.items()
            for metric, value in vals.items()
        },
    })


    final_metrics = {
        "val_auc": val_auc,
        "test_acc_raw": test_acc,
        "test_accuracy": Acc,
        "test_precision": Prec,
        "test_sensitivity": Sens,
        "test_specificity": Spec,
        "test_f1": F1,
        "test_uar": uar,
        "test_auc": auc,
        "statistical_parity": SP,
        "equal_opportunity": EOpp,
        "equalized_odds": EOdd,
        "equal_accuracy": EAcc,
        "sp_age_bin": age_metrics["SP"],
        "equal_opportunity_age_bin": age_metrics["EOpp"],
        "equalized_odds_age_bin": age_metrics["EOdd"],
        "equal_accuracy_age_bin": age_metrics["EAcc"],
        **race_aggregated_metrics,
        **{
            f"race_pairwise/{pair}/{metric}": value
            for pair, vals in race_pairwise_metrics.items()
            for metric, value in vals.items()
        },
    }

    _save_final_results_row(config, final_metrics)



if __name__ == "__main__":
    if RUN_AS_SWEEP:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
        wandb.agent(sweep_id, function=main_sweep)
    else:
        main_sweep()
