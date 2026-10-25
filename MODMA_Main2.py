import os
import re
import csv
import json
import time
import hashlib
import random
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from Models.FeatNet import EEGFeatNet, AudioFeatNet, MultiModalCNN
from DataSets.MODMA import prepare_subject_dataloader, prepare_folds_and_test_data
from Utils.Eval import setup_linear_classifier, linear_probing, initialize_weights, train_final_model, evaluate_model
from Utils.metric_calculate import calculate_metrics
from SSL.VICReg import VICReg_M1, VICReg_M234
from Utils.config import is_configured, load_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = load_yaml_config(PROJECT_ROOT / "configs" / "modma.yaml")

WANDB_API_KEY = CONFIG.get("wandb", {}).get("api_key")
if is_configured(WANDB_API_KEY):
    wandb.login(key=WANDB_API_KEY)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

MODEL_SAVE_DIR = CONFIG.get("paths", {}).get("model_save_dir", "saved_models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# -------------------- helpers --------------------
def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sigmoid_np(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _get_save_root() -> str:
    save_root = CONFIG.get("paths", {}).get("save_root")
    return save_root if is_configured(save_root) else "modma_results"


def _merge_subject_data(*folds):
    merged = {}
    for fold in folds:
        for sid, values in fold.items():
            if sid not in merged:
                merged[sid] = {
                    "audio": list(values.get("audio", [])),
                    "eeg": list(values.get("eeg", [])),
                    "labels": list(values.get("labels", [])),
                    "genders": list(values.get("genders", [])),
                    "subjects": list(values.get("subjects", [])),
                }
            else:
                merged[sid]["audio"].extend(values.get("audio", []))
                merged[sid]["eeg"].extend(values.get("eeg", []))
                if not merged[sid]["labels"]:
                    merged[sid]["labels"].extend(values.get("labels", []))
                if not merged[sid]["genders"]:
                    merged[sid]["genders"].extend(values.get("genders", []))
                if not merged[sid]["subjects"]:
                    merged[sid]["subjects"].extend(values.get("subjects", []))
    return merged


def _sanitize(s: str, max_len: int = 160) -> str:
    s = re.sub(r"[^\w\.\-+=,]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) <= max_len:
        return s
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    return s[: max_len - 11] + "_" + h


def _append_row_csv(path, fieldnames, row_dict):
    _ensure_dir(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row_dict.get(k, None) for k in fieldnames})


def save_test_outputs(run_dir: str,
                      seed: int,
                      y_pred, y_true, y_gender, y_logits, y_ids,
                      config_dict: dict,
                      metrics: dict) -> str:
    out_dir = Path(run_dir) / "test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    logits = np.asarray(y_logits).reshape(-1)
    probs = _sigmoid_np(logits)

    npz_path = out_dir / f"seed_{seed}_test_outputs.npz"
    np.savez_compressed(
        npz_path,
        y_pred=np.asarray(y_pred),
        y_true=np.asarray(y_true),
        y_gender=np.asarray(y_gender),
        y_logits=logits,
        y_prob=probs,
        y_ids=np.asarray(list(y_ids), dtype=object),
    )

    meta_path = out_dir / f"seed_{seed}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"seed": int(seed), "config": config_dict, "metrics": metrics}, f, indent=2)

    return str(out_dir)


# -------------------- centroid tracker (male vs female) --------------------
class GenderCentroidTrackerMODMA:
    """
    Tracks two gender group centroids per epoch in 2D PCA space (basis computed once and reused).
    Saves:
      - csv per epoch
      - pca basis file (npz)
    """
    def __init__(self, out_dir: str, csv_name: str, max_batches=None):
        self.out_dir = _ensure_dir(out_dir)
        self.csv_path = os.path.join(out_dir, csv_name)
        self.basis_path = os.path.join(out_dir, "pca_basis.npz")
        self.max_batches = max_batches

        self.fields = [
            "epoch",
            "centroid_distance",
            "group0_label", "group1_label",
            "group0_n", "group1_n",
            "group0_cx", "group0_cy",
            "group1_cx", "group1_cy",
        ]

        self.mean_ = None
        self.components_ = None  # 2 x D

        if os.path.exists(self.basis_path):
            d = np.load(self.basis_path)
            self.mean_ = d["mean"]
            self.components_ = d["components"]

    @staticmethod
    def _pca_fit(X: np.ndarray):
        mean = X.mean(axis=0, keepdims=True)
        Xc = X - mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        components = Vt[:2, :]  # 2 x D
        return mean.reshape(-1), components

    def _pca_transform(self, X: np.ndarray):
        Xc = X - self.mean_[None, :]
        return Xc @ self.components_.T

    def compute_and_append(self, epoch: int, model, val_dataloader, device):
        model.eval()
        feats = []
        genders = []
        n_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if len(batch) == 6:
                    audio, eeg, _y, g, audio_ids, eeg_ids = batch
                else:
                    audio, eeg, _y, g, audio_ids = batch
                    eeg_ids = audio_ids
                audio = audio.to(device).float()
                eeg   = eeg.to(device).float()

                z_a, z_b = model(audio, eeg)

                if torch.is_tensor(audio_ids):
                    audio_ids = audio_ids.detach().cpu().tolist()
                else:
                    audio_ids = list(audio_ids)
                if torch.is_tensor(eeg_ids):
                    eeg_ids = eeg_ids.detach().cpu().tolist()
                else:
                    eeg_ids = list(eeg_ids)
                if torch.is_tensor(g):
                    g_values = g.detach().cpu().numpy().reshape(-1).tolist()
                else:
                    g_values = np.asarray(g).reshape(-1).tolist()

                audio_groups = {}
                eeg_groups = {}
                gender_by_subject = {}
                for idx, sid in enumerate(audio_ids):
                    audio_groups.setdefault(sid, []).append(idx)
                    if idx < len(g_values):
                        gender_by_subject.setdefault(sid, g_values[idx])
                for idx, sid in enumerate(eeg_ids):
                    eeg_groups.setdefault(sid, []).append(idx)

                common_subjects = [
                    sid for sid in audio_groups.keys()
                    if sid in eeg_groups and sid in gender_by_subject
                ]
                if not common_subjects:
                    continue

                batch_feats = []
                batch_genders = []
                for sid in common_subjects:
                    a = z_a[audio_groups[sid]].mean(dim=0)
                    b = z_b[eeg_groups[sid]].mean(dim=0)
                    batch_feats.append(torch.cat([a, b], dim=0).detach().cpu().numpy())
                    batch_genders.append(gender_by_subject[sid])

                feats.append(np.stack(batch_feats, axis=0))
                genders.append(np.asarray(batch_genders))


                n_batches += 1
                if self.max_batches is not None and n_batches >= int(self.max_batches):
                    break

        if len(feats) == 0:
            return {}

        X = np.concatenate(feats, axis=0)
        G = np.concatenate(genders, axis=0).reshape(-1)

        if self.mean_ is None or self.components_ is None:
            mean, comps = self._pca_fit(X)
            self.mean_ = mean
            self.components_ = comps
            np.savez_compressed(self.basis_path, mean=self.mean_, components=self.components_)

        X2 = self._pca_transform(X)  # N x 2

        uniq = list(pd.unique(G)) if "pd" in globals() else list(dict.fromkeys(G.tolist()))
        if len(uniq) < 2:
            return {}

        g0, g1 = uniq[0], uniq[1]
        m0 = X2[G == g0]
        m1 = X2[G == g1]
        if len(m0) == 0 or len(m1) == 0:
            return {}

        c0 = m0.mean(axis=0)
        c1 = m1.mean(axis=0)
        dist = float(np.linalg.norm(c0 - c1))

        row = {
            "epoch": int(epoch),
            "centroid_distance": dist,
            "group0_label": str(g0),
            "group1_label": str(g1),
            "group0_n": int(len(m0)),
            "group1_n": int(len(m1)),
            "group0_cx": float(c0[0]),
            "group0_cy": float(c0[1]),
            "group1_cx": float(c1[0]),
            "group1_cy": float(c1[1]),
        }
        _append_row_csv(self.csv_path, self.fields, row)
        return row


# -------------------- robust imports for losses --------------------
def _import_loss_classes():
    return {
        "VICReg_M1": VICReg_M1,
        "VICReg_M234": VICReg_M234,
    }


LOSS = _import_loss_classes()


def build_loss_fn(loss_family: str, loss_type: str, dim_mod1: int, dim_mod2: int, config):
    family = str(loss_family).lower()
    if family == "fairwell":
        family = "vicreg"
    variant = str(loss_type).upper()

    if family != "vicreg":
        raise ValueError(f"Unsupported FairWell loss_family: {loss_family}")
    if variant == "M1":
        return LOSS["VICReg_M1"](config)
    if variant in {"M2", "M3", "M4"}:
        return LOSS["VICReg_M234"](dim_mod1, dim_mod2, config)
    raise ValueError(f"Unsupported FairWell loss_type: {loss_type}")


def train_one_epoch_modma(model, dataloader, optimizer, loss_fn,
                          loss_family: str, loss_type: str, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    family = str(loss_family).lower()
    if family == "fairwell":
        family = "vicreg"
    variant = str(loss_type).upper()
    if family != "vicreg" or variant not in {"M1", "M2", "M3", "M4"}:
        raise ValueError(f"Unsupported FairWell loss: {loss_family}/{loss_type}")

    for batch in dataloader:
        if len(batch) == 6:
            audio, eeg, _y_label, _gender, audio_ids, eeg_ids = batch
        else:
            audio, eeg, _y_label, _gender, audio_ids = batch
            eeg_ids = audio_ids
        audio = audio.to(device).float()
        eeg   = eeg.to(device).float()
        if torch.is_tensor(audio_ids):
            audio_ids = audio_ids.detach().cpu().tolist()
        else:
            audio_ids = list(audio_ids)
        if torch.is_tensor(eeg_ids):
            eeg_ids = eeg_ids.detach().cpu().tolist()
        else:
            eeg_ids = list(eeg_ids)

        optimizer.zero_grad()

        z_a, z_b = model(audio, eeg)
        loss = loss_fn(z_a, z_b, audio_ids, eeg_ids)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


# -------------------- val metrics --------------------
def compute_val_auc(multimodal_model, linear_classifier, val_dataloader, device):
    y_pred, y_true, y_gender, y_logits, _ = evaluate_model(
        val_dataloader, multimodal_model, linear_classifier
    )

    logits = np.array(y_logits).reshape(-1)
    probs  = _sigmoid_np(logits)

    _, _, _, _, val_acc, _, _, _, _, _, val_auc = calculate_metrics(
        y_pred=y_pred,
        y_labels=y_true,
        gender_list=y_gender,
        y_scores=probs,
        linearprobing=True
    )
    return float(val_acc), float(val_auc)


# -------------------- final test --------------------
def run_final_test_for_config(config_dict: dict,
                              run_dir: str,
                              seed_values=(42,)) -> dict:
    per_seed_metrics = []

    for seed in seed_values:
        set_global_seed(seed)
        print(f"\n=== Final train+test for seed {seed} ===")

        y_pred, y_true, y_gender, y_logits, y_ids = train_final_model(
            dict(config_dict),
            all_train_data_by_subject,
            test_data_by_subject,
            seed,
        )

        logits = np.asarray(y_logits).reshape(-1)
        probs = _sigmoid_np(logits)

        SP, EOpp, EOdd, EAcc, Acc, Prec, Sens, Spec, F1, uar, auc = calculate_metrics(
            y_pred=y_pred,
            y_labels=y_true,
            gender_list=y_gender,
            y_scores=probs,
        )

        metrics_named = {
            "Accuracy": float(Acc),
            "F1": float(F1),
            "Precision": float(Prec),
            "Sensitivity": float(Sens),
            "Specificity": float(Spec),
            "Statistical_Parity": float(SP),
            "Equal_Opportunity": float(EOpp),
            "Equal_Odds": float(EOdd),
            "Equal_Accuracy": float(EAcc),
            "UAR": float(uar),
            "AUC": float(auc),
        }

        wandb.log({f"test/seed_{seed}/{k}": v for k, v in metrics_named.items()})

        save_test_outputs(
            run_dir=run_dir,
            seed=seed,
            y_pred=y_pred,
            y_true=y_true,
            y_gender=y_gender,
            y_logits=y_logits,
            y_ids=y_ids,
            config_dict=dict(config_dict),
            metrics=metrics_named,
        )

        per_seed_metrics.append(metrics_named)

    keys = list(per_seed_metrics[0].keys())
    mean = {f"test/{k}_mean": float(np.mean([m[k] for m in per_seed_metrics])) for k in keys}

    if len(per_seed_metrics) > 1:
        std = {f"test/{k}_std": float(np.std([m[k] for m in per_seed_metrics], ddof=1)) for k in keys}
    else:
        std = {f"test/{k}_std": 0.0 for k in keys}

    wandb.log({**mean, **std})
    return {**mean, **std}


PATH_CONFIG = CONFIG.get("paths", {})
DATA_CONFIG = CONFIG.get("data", {})

data = prepare_folds_and_test_data(
    base_dir_eeg=PATH_CONFIG.get("train_base_dir_eeg"),
    base_dir_audio=PATH_CONFIG.get("train_base_dir_audio"),
    eeg_aug_list=DATA_CONFIG.get("train_eeg_aug_list", []),
    audio_aug_list=DATA_CONFIG.get("train_audio_aug_list", []),
)

data_val = prepare_folds_and_test_data(
    base_dir_eeg=PATH_CONFIG.get("val_base_dir_eeg"),
    base_dir_audio=PATH_CONFIG.get("val_base_dir_audio"),
    eeg_aug_list=DATA_CONFIG.get("val_eeg_aug_list", []),
    audio_aug_list=DATA_CONFIG.get("val_audio_aug_list", []),
)

all_folds = [data[f"fold{i}"] for i in range(1, 6)]
all_folds_val = [data_val[f"fold{i}"] for i in range(1, 6)]
test_data_by_subject = data_val["test"]
all_train_data_by_subject = _merge_subject_data(*all_folds)


# ======================
# MAIN TRAIN FUNCTION
# ======================
def train_model():
    wandb.init()
    config = wandb.config

    SAVE_ROOT = _ensure_dir(_get_save_root())

    family = str(config.loss_family).lower()
    variant = str(config.loss_type).upper()

    htag = (
        f"lr={float(config.learning_rate):.3g}_linlr={float(config.linear_lr):.3g}_"
        f"spb={int(config.subjects_per_batch)}_wd={float(config.weight_decay):.3g}_"
        f"sslE={int(config.epoch)}_probeE={int(config.probe_epochs)}"
    )

    run_id = wandb.run.id if wandb.run is not None else f"norun_{int(time.time())}"
    run_dir_name = _sanitize(f"{family}__{variant}__{htag}__{run_id}")
    RUN_DIR = _ensure_dir(os.path.join(SAVE_ROOT, run_dir_name))

    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump(dict(config), f, indent=2)

    metrics_dir = _ensure_dir(os.path.join(RUN_DIR, "metrics"))
    metrics_csv = os.path.join(metrics_dir, "ssl_epoch_metrics.csv")
    metrics_fields = [
        "fold", "epoch",
        "ssl_loss",
        "loader_type",
        "val_auc",
        "val_acc",
        "centroid_distance",
        "group0_label", "group1_label",
        "group0_n", "group1_n",
        "group0_cx", "group0_cy",
        "group1_cx", "group1_cy",
    ]

    seed_values = CONFIG.get("final", {}).get("seed_values", [42])
    best_val_aucs = []

    for seed in seed_values:
        set_global_seed(seed)

        for fold_idx in range(5):
            print(f"\n=== Starting Fold {fold_idx + 1} ===")
            wandb.log({"current_fold": fold_idx + 1})

            val_fold = all_folds_val[fold_idx]
            train_folds = [f for i, f in enumerate(all_folds) if i != fold_idx]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            train_data_by_subject = _merge_subject_data(*train_folds)
            val_data_by_subject = val_fold

            # Probe loader (always mixed-label for probing)
            probe_train_loader = prepare_subject_dataloader(
                train_data_by_subject,
                config.subjects_per_batch,
                device,
                single_label=False
            )

            train_loader_ml = prepare_subject_dataloader(train_data_by_subject, config.subjects_per_batch, device, single_label=False)
            train_loader_sl = prepare_subject_dataloader(train_data_by_subject, config.subjects_per_batch, device, single_label=True)
            val_loader_ml   = prepare_subject_dataloader(val_data_by_subject,   config.subjects_per_batch, device, single_label=False)
            val_loader_sl   = prepare_subject_dataloader(val_data_by_subject,   config.subjects_per_batch, device, single_label=True)

            # Init model
            cnn_audio_encoder = AudioFeatNet(input_shape=(1024, 128)).to(device)
            cnn_eeg_encoder   = EEGFeatNet(input_shape=(128, 500)).to(device)
            initialize_weights(cnn_audio_encoder)
            initialize_weights(cnn_eeg_encoder)

            multimodal_model = MultiModalCNN(cnn_audio_encoder, cnn_eeg_encoder).to(device)
            if torch.cuda.device_count() > 1:
                multimodal_model = nn.DataParallel(multimodal_model)

            optimizer = optim.Adam(
                multimodal_model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

            # Infer embedding dims
            with torch.no_grad():
                dummy_audio = torch.zeros(1, 1, 1024, 128, device=device)
                dummy_eeg   = torch.zeros(1, 1, 128, 500, device=device)
                dim_mod1 = cnn_audio_encoder(dummy_audio).shape[1]
                dim_mod2 = cnn_eeg_encoder(dummy_eeg).shape[1]

            loss_fn = build_loss_fn(config.loss_family, config.loss_type, dim_mod1, dim_mod2, config).to(device)

            with torch.no_grad():
                B0 = 4
                sid0 = list(range(B0))
                _ = loss_fn(
                    torch.randn(B0, dim_mod1, device=device),
                    torch.randn(B0, dim_mod2, device=device),
                    sid0,
                    sid0
                )

            loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
            if len(loss_params) > 0:
                optimizer.add_param_group({"params": loss_params})

            # Centroid tracker per fold
            centroid_dir = _ensure_dir(os.path.join(RUN_DIR, "centroid_tracking", f"fold_{fold_idx + 1}"))
            centroid_csv_name = f"gender_centroids__{family}__{variant}__{htag}__fold{fold_idx + 1}.csv"
            centroid_tracker = GenderCentroidTrackerMODMA(
                out_dir=centroid_dir,
                csv_name=centroid_csv_name,
                max_batches=getattr(config, "centroid_max_batches", None)
            )

            # Early stopping on AUC
            best_val_auc = -1.0
            patience = int(config.patience)
            patience_counter = 0
            best_multimodal_model = deepcopy(multimodal_model)
            best_linear_classifier = None

            num_epochs = int(config.epoch)

            for ep in range(num_epochs):
                if variant in {"M1", "M2"}:
                    cur_loader = train_loader_ml
                    val_loader = val_loader_ml
                    loader_tag = "ML"
                elif variant == "M3":
                    cur_loader = train_loader_sl
                    val_loader = val_loader_sl
                    loader_tag = "SL"
                else:  # M4
                    if (ep + 1) % 2 == 1:
                        cur_loader = train_loader_sl
                        val_loader = val_loader_sl
                        loader_tag = "SL"
                    else:
                        cur_loader = train_loader_ml
                        val_loader = val_loader_ml
                        loader_tag = "ML"

                avg_loss = train_one_epoch_modma(
                    multimodal_model,
                    cur_loader,
                    optimizer,
                    loss_fn,
                    config.loss_family,
                    config.loss_type,
                    device
                )

                wandb.log({
                    "fold": fold_idx + 1,
                    "epoch": ep + 1,
                    "training_loss": float(avg_loss),
                    "loader_type": loader_tag
                })

                # Centroids each epoch on fixed val ML for comparability
                model_for_cent = multimodal_model.module if isinstance(multimodal_model, nn.DataParallel) else multimodal_model
                cent = centroid_tracker.compute_and_append(
                    epoch=ep + 1,
                    model=model_for_cent,
                    val_dataloader=val_loader_ml,
                    device=device
                ) or {}

                # Linear probing + val AUC
                _, (multimodal_model, linear_classifier) = linear_probing(
                    multimodal_model=multimodal_model,
                    train_dataloader=probe_train_loader,
                    val_dataloader=val_loader_ml,
                    setup_linear_classifier=setup_linear_classifier,
                    device=device,
                    MODEL_SAVE_DIR=os.path.join(RUN_DIR, f"fold_{fold_idx + 1}"),
                    config=config
                )

                val_acc_eval, val_auc = compute_val_auc(multimodal_model, linear_classifier, val_loader_ml, device)

                wandb.log({
                    "fold": fold_idx + 1,
                    "epoch": ep + 1,
                    "val_acc": float(val_acc_eval),
                    "val_auc": float(val_auc),
                })

                # per-epoch CSV
                row = {
                    "fold": fold_idx + 1,
                    "epoch": ep + 1,
                    "ssl_loss": float(avg_loss),
                    "loader_type": loader_tag,
                    "val_auc": float(val_auc),
                    "val_acc": float(val_acc_eval),

                    "centroid_distance": float(cent.get("centroid_distance", float("nan"))),
                    "group0_label": cent.get("group0_label", None),
                    "group1_label": cent.get("group1_label", None),
                    "group0_n": cent.get("group0_n", None),
                    "group1_n": cent.get("group1_n", None),
                    "group0_cx": cent.get("group0_cx", None),
                    "group0_cy": cent.get("group0_cy", None),
                    "group1_cx": cent.get("group1_cx", None),
                    "group1_cy": cent.get("group1_cy", None),
                }
                _append_row_csv(metrics_csv, metrics_fields, row)

                # early stopping
                if float(val_auc) > best_val_auc:
                    best_val_auc = float(val_auc)
                    best_multimodal_model = deepcopy(multimodal_model)
                    best_linear_classifier = deepcopy(linear_classifier)
                    patience_counter = 0

                    fold_dir = _ensure_dir(os.path.join(RUN_DIR, f"fold_{fold_idx + 1}"))
                    torch.save({
                        "model_state_dict": (best_multimodal_model.module.state_dict()
                                             if isinstance(best_multimodal_model, nn.DataParallel)
                                             else best_multimodal_model.state_dict()),
                        "linear_classifier_state_dict": best_linear_classifier.state_dict(),
                        "config": dict(config)
                    }, os.path.join(fold_dir, "best_model.pth"))

                    wandb.log({"best_val_auc_so_far": best_val_auc})
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Fold {fold_idx + 1}: Early stopping at epoch {ep + 1}.")
                        break

            best_val_aucs.append(best_val_auc)

    av_val_auc = float(np.mean(best_val_aucs)) if len(best_val_aucs) else 0.0
    std_val_auc = float(np.std(best_val_aucs, ddof=1)) if len(best_val_aucs) > 1 else 0.0

    wandb.log({"av_val_auc": av_val_auc, "std_val_auc": std_val_auc})
    wandb.run.summary["val_auc"] = av_val_auc

    test_summary = run_final_test_for_config(dict(config), RUN_DIR, seed_values=seed_values)
    for k, v in test_summary.items():
        wandb.run.summary[k] = v

    wandb.finish()
    return av_val_auc, std_val_auc


# ======================
# SWEEP CONFIGS
# ======================
def make_sweep_config(loss_family: str, loss_type: str):
    loss_family = str(loss_family).lower()
    if loss_family == "fairwell":
        loss_family = "vicreg"
    loss_type   = str(loss_type).upper()
    if loss_family != "vicreg":
        raise ValueError(f"Unsupported FairWell loss_family: {loss_family}")
    if loss_type not in {"M1", "M2", "M3", "M4"}:
        raise ValueError(f"Unsupported FairWell loss_type: {loss_type}")

    cfg = deepcopy(CONFIG.get("sweep", {}))
    cfg.setdefault("parameters", {})
    cfg["parameters"]["loss_family"] = {"value": loss_family}
    cfg["parameters"]["loss_type"] = {"value": loss_type}
    return cfg


def run_all_sweeps():
    run_config = CONFIG.get("runs", {})
    loss_family = str(run_config.get("loss_family", "vicreg"))
    loss_types = run_config.get("loss_types", ["M1", "M2", "M3", "M4"])
    jobs = [(loss_family, v) for v in loss_types]
    project_prefix = CONFIG.get("wandb", {}).get("project_prefix", "modma_fairwell")
    agent_count = int(CONFIG.get("wandb", {}).get("agent_count", 1))

    for fam, typ in jobs:
        project_name = f"{project_prefix}_{typ}"
        sweep_config = make_sweep_config(fam, typ)
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, function=train_model, count=agent_count)


if __name__ == "__main__":
    # One call runs everything in a single loop.
    run_all_sweeps()
