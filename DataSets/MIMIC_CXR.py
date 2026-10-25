"""MIMIC-CXR subject-level dataloaders for FairWell."""

import ast
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms as T


EXCLUDE_RACES = {"UNKNOWN", "OTHER", "UNABLE TO OBTAIN"}
RACE_GROUPS = ["WHITE", "BLACK", "ASIAN", "HISPANIC_LATINO"]
RACE_TO_ID = {g: i for i, g in enumerate(RACE_GROUPS)}


def _sha1_text_id(s: str) -> str:
    return hashlib.sha1(str(s).encode("utf-8", errors="ignore")).hexdigest()


def _parse_list_cell(x) -> List[str]:
    """Parse a CSV cell that may contain a Python list string ("['a','b']") or a scalar."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, float) and np.isnan(x):
        return []
    if not isinstance(x, str):
        s = str(x).strip()
        return [] if (not s or s.lower() == "nan") else [s]

    s = x.strip()
    if not s or s.lower() == "nan":
        return []
    try:
        v = ast.literal_eval(s)
    except Exception:
        return [s]

    if isinstance(v, list):
        out: List[str] = []
        for it in v:
            t = str(it).strip()
            if t and t.lower() != "nan":
                out.append(t)
        return out

    t = str(v).strip()
    return [] if (not t or t.lower() == "nan") else [t]


def _clean_unique_texts(texts: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in texts:
        tt = str(t).strip()
        if not tt:
            continue
        if tt not in seen:
            seen.add(tt)
            out.append(tt)
    return out


def _race_to_group_optionA(race_raw: str) -> Optional[str]:
    r = str(race_raw).strip().upper()
    if r in EXCLUDE_RACES:
        return None
    if r.startswith("WHITE") or r == "PORTUGUESE":
        return "WHITE"
    if r.startswith("BLACK"):
        return "BLACK"
    if r.startswith("ASIAN"):
        return "ASIAN"
    if r.startswith("HISPANIC/LATINO") or r in {"HISPANIC OR LATINO", "HISPANIC/LATINO"}:
        return "HISPANIC_LATINO"
    return None


@dataclass
class SubjectRecord:
    subject_id: int
    images_rel: List[str]         # relative .jpg paths (under img_root)
    text_keys: List[str]          # sha1 keys
    label: int
    race_id: int
    gender_id: int
    age: float
    age_bin: int


class _SameLabelSubjectBatchSampler(Sampler[List[int]]):
    """Batch sampler that yields batches containing only one label (homogeneous batches).

    The sampler uses precomputed subject labels and does not call dataset indexing during grouping.
    """

    def __init__(self, subject_labels: List[int], batch_size: int, seed: int = 42):
        self.subject_labels = list(map(int, subject_labels))
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        label_to_indices: Dict[int, List[int]] = {}
        for i, y in enumerate(self.subject_labels):
            label_to_indices.setdefault(int(y), []).append(i)
        self.label_to_indices = label_to_indices
        self.labels = sorted(label_to_indices.keys())

    def __iter__(self):
        rng = random.Random(self.seed)
        labels = self.labels[:]
        rng.shuffle(labels)
        for lab in labels:
            idxs = self.label_to_indices[lab][:]
            rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                yield idxs[i : i + self.batch_size]

    def __len__(self):
        import math

        total = 0
        for idxs in self.label_to_indices.values():
            total += math.ceil(len(idxs) / self.batch_size)
        return total


class MIMICCXRSubjectDataset(Dataset):
    """Returns one item per subject.

    Per-subject sampling is deterministic for a given seed, subject, and epoch.
    """

    def __init__(
        self,
        subject_records: List[SubjectRecord],
        base_seed: int = 12345,
        max_items_per_subject: int = 5,
    ):
        self.records = subject_records
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.subject_labels = [int(r.label) for r in subject_records]
        self.max_items_per_subject = int(max_items_per_subject)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]

        # Deterministic RNG per subject per epoch
        seed = self.base_seed + int(r.subject_id) + 1000003 * int(self.epoch)
        rng = random.Random(seed)

        imgs = list(r.images_rel)
        txts = list(r.text_keys)

        rng.shuffle(imgs)
        rng.shuffle(txts)

        # Cap per modality (independently)
        m = self.max_items_per_subject
        if m > 0:
            if len(imgs) > m:
                imgs = imgs[:m]
            if len(txts) > m:
                txts = txts[:m]

        # Second text view: independently shuffled + capped
        txts_u = list(txts)
        rng.shuffle(txts_u)
        if m > 0 and len(txts_u) > m:
            txts_u = txts_u[:m]

        return {
            "subject_id": int(r.subject_id),
            "images_rel": imgs,
            "text_keys": txts,
            "text_keys_u": txts_u,
            "label": int(r.label),
            "race": int(r.race_id),
            "gender": int(r.gender_id),
            "age": float(r.age),
            "age_bin": int(r.age_bin),
        }


def _make_collate_fn(
    *,
    img_root: Path,
    cache_input_ids: torch.Tensor,
    cache_attn_mask: torch.Tensor,
    cache_lengths: torch.Tensor,
    key_to_row: Dict[str, int],
    output_mode: str,
    pairs_per_subject: Union[int, str],
    max_pairs_per_subject: Optional[int],
    image_size: int,
    base_seed: int,
    pad_pairs: bool = False,
):
    output_mode = str(output_mode).strip().lower()
    if output_mode not in {"paired", "bag"}:
        raise ValueError("output_mode must be 'paired' or 'bag'")

    if isinstance(pairs_per_subject, str):
        pairs_per_subject = pairs_per_subject.strip().lower()
        if pairs_per_subject not in {"all"}:
            raise ValueError("pairs_per_subject string must be 'all'")

    to_uint8_gray = T.Compose([
        T.Resize((image_size, image_size)),
        T.PILToTensor(),
    ])

    pad_img = torch.zeros(1, image_size, image_size, dtype=torch.uint8)

    # Prefer using cached tokens for the empty string if present.
    empty_key = _sha1_text_id("")
    empty_row = key_to_row.get(empty_key, None)
    if empty_row is not None:
        pad_ids = cache_input_ids[empty_row].to(torch.long).clone()
        pad_mask = cache_attn_mask[empty_row].to(torch.long).clone()
        pad_len = cache_lengths[empty_row].to(torch.long).clone()
    else:
        # Fallback: deterministic "zero text" row
        pad_ids = torch.zeros(int(cache_input_ids.shape[1]), dtype=torch.long)
        pad_mask = torch.zeros(int(cache_attn_mask.shape[1]), dtype=torch.long)
        # Avoid all-masked edge cases in some text encoders
        if pad_mask.numel() > 0:
            pad_mask[0] = 1
        pad_len = torch.ones((), dtype=torch.long)

    def _load_image(rel_path: str) -> torch.Tensor:
        p = img_root / rel_path
        img = Image.open(p).convert("L")
        return to_uint8_gray(img)  # [1,H,W] uint8

    def _texts_to_tokens(text_keys: List[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map cached text sha1 keys -> (input_ids, attention_mask, lengths).

        Supports padding entries (None): None is mapped to the "empty text" padded row.
        """
        n = len(text_keys)
        L = int(cache_input_ids.shape[1])

        if n == 0:
            return (
                torch.empty(0, L, dtype=torch.long),
                torch.empty(0, L, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        # Fill all rows with padding; overwrite rows that have real keys.
        out_ids = pad_ids.unsqueeze(0).expand(n, -1).clone()
        out_mask = pad_mask.unsqueeze(0).expand(n, -1).clone()
        out_len = pad_len.reshape(1).expand(n).clone()

        pos: List[int] = []
        rows: List[int] = []
        for i, k in enumerate(text_keys):
            if k is None:
                continue
            j = key_to_row.get(k)
            if j is None:
                raise KeyError(
                    "Found a text key not present in the precomputed cache. "
                    "Either rebuild the cache or filter the input CSVs so all texts are cached. "
                    f"Missing sha1={k}"
                )
            pos.append(int(i))
            rows.append(int(j))

        if pos:
            jj = torch.tensor(rows, dtype=torch.long)
            out_ids[pos] = cache_input_ids.index_select(0, jj)
            out_mask[pos] = cache_attn_mask.index_select(0, jj)
            out_len[pos] = cache_lengths.index_select(0, jj)

        return out_ids, out_mask, out_len

    def collate(batch: List[Dict[str, Any]]):
        # ------------------------------------
        # BAG MODE (no pairing, no padding)
        # ------------------------------------
        if output_mode == "bag":
            all_imgs: List[torch.Tensor] = []
            all_img_sids: List[int] = []
            all_txt_keys: List[str] = []
            all_txt_sids: List[int] = []
            all_txt_keys_u: List[str] = []
            all_txt_sids_u: List[int] = []

            subj_sids: List[int] = []
            subj_labels: List[int] = []
            subj_gender: List[int] = []
            subj_race: List[int] = []
            subj_age: List[float] = []
            subj_age_bin: List[int] = []

            for s in batch:
                sid = int(s["subject_id"])
                subj_sids.append(sid)
                subj_labels.append(int(s["label"]))
                subj_gender.append(int(s["gender"]))
                subj_race.append(int(s["race"]))
                subj_age.append(float(s["age"]))
                subj_age_bin.append(int(s["age_bin"]))

                for rp in s["images_rel"]:
                    all_imgs.append(_load_image(rp))
                    all_img_sids.append(sid)

                for k in s["text_keys"]:
                    all_txt_keys.append(k)
                    all_txt_sids.append(sid)
                for k in s["text_keys_u"]:
                    all_txt_keys_u.append(k)
                    all_txt_sids_u.append(sid)

            img_tensor = (
                torch.stack(all_imgs, dim=0)
                if all_imgs
                else torch.empty(0, 1, image_size, image_size, dtype=torch.uint8)
            )

            txt_ids, txt_mask, txt_len = _texts_to_tokens(all_txt_keys) if all_txt_keys else (
                torch.empty(0, cache_input_ids.shape[1], dtype=torch.long),
                torch.empty(0, cache_attn_mask.shape[1], dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
            txt_ids_u, txt_mask_u, txt_len_u = _texts_to_tokens(all_txt_keys_u) if all_txt_keys_u else (
                torch.empty(0, cache_input_ids.shape[1], dtype=torch.long),
                torch.empty(0, cache_attn_mask.shape[1], dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

            return {
                "img": img_tensor,
                "img_sid": torch.tensor(all_img_sids, dtype=torch.long),
                "txt_ids": txt_ids,
                "txt_mask": txt_mask,
                "txt_len": txt_len,
                "txt_sid": torch.tensor(all_txt_sids, dtype=torch.long),
                "txt_ids_u": txt_ids_u,
                "txt_mask_u": txt_mask_u,
                "txt_len_u": txt_len_u,
                "txt_sid_u": torch.tensor(all_txt_sids_u, dtype=torch.long),
                "subject_ids": torch.tensor(subj_sids, dtype=torch.long),
                "label": torch.tensor(subj_labels, dtype=torch.long),
                "gender": torch.tensor(subj_gender, dtype=torch.long),
                "race": torch.tensor(subj_race, dtype=torch.long),
                "age": torch.tensor(subj_age, dtype=torch.float32),
                "age_bin": torch.tensor(subj_age_bin, dtype=torch.long),
            }

        pair_imgs: List[torch.Tensor] = []
        pair_txt_keys: List[Optional[str]] = []
        pair_txt_keys_u: List[Optional[str]] = []

        pair_sids: List[int] = []
        pair_labels: List[int] = []
        pair_gender: List[int] = []
        pair_race: List[int] = []
        pair_age: List[float] = []
        pair_age_bin: List[int] = []

        for s in batch:
            sid = int(s["subject_id"])
            imgs: List[str] = list(s["images_rel"])
            txts: List[str] = list(s["text_keys"])
            txts_u: List[str] = list(s["text_keys_u"])

            # If both modalities are empty, skip.
            if len(imgs) == 0 and len(txts) == 0:
                continue

            if pairs_per_subject == "all":
                k = max(len(imgs), len(txts))
            else:
                k = int(pairs_per_subject)

            if max_pairs_per_subject is not None:
                k = min(k, int(max_pairs_per_subject))

            if k <= 0:
                continue

            if pad_pairs:
                # STRICT padding: never repeat real segments
                imgs_k: List[Optional[str]] = imgs[:k]
                txts_k: List[Optional[str]] = txts[:k]
                txts_u_k: List[Optional[str]] = txts_u[:k]

                if len(imgs_k) < k:
                    imgs_k.extend([None] * (k - len(imgs_k)))
                if len(txts_k) < k:
                    txts_k.extend([None] * (k - len(txts_k)))
                if len(txts_u_k) < k:
                    txts_u_k.extend([None] * (k - len(txts_u_k)))

                for i in range(k):
                    rp = imgs_k[i]
                    t1 = txts_k[i]
                    t2 = txts_u_k[i]

                    pair_imgs.append(pad_img if rp is None else _load_image(rp))

                    if t1 is None:
                        pair_txt_keys.append(None)
                        pair_txt_keys_u.append(None)
                    else:
                        if t2 is None:
                            t2 = t1
                        pair_txt_keys.append(t1)
                        pair_txt_keys_u.append(t2)

                    pair_sids.append(sid)
                    pair_labels.append(int(s["label"]))
                    pair_gender.append(int(s["gender"]))
                    pair_race.append(int(s["race"]))
                    pair_age.append(float(s["age"]))
                    pair_age_bin.append(int(s["age_bin"]))

            else:
                # LEGACY cycling: NO dummy padding. Require both modalities present.
                if len(imgs) == 0 or len(txts) == 0:
                    continue

                for i in range(k):
                    rp = imgs[i % len(imgs)]
                    t1 = txts[i % len(txts)]
                    if len(txts_u) > 0:
                        t2 = txts_u[i % len(txts_u)]
                    else:
                        t2 = t1

                    pair_imgs.append(_load_image(rp))
                    pair_txt_keys.append(t1)
                    pair_txt_keys_u.append(t2)

                    pair_sids.append(sid)
                    pair_labels.append(int(s["label"]))
                    pair_gender.append(int(s["gender"]))
                    pair_race.append(int(s["race"]))
                    pair_age.append(float(s["age"]))
                    pair_age_bin.append(int(s["age_bin"]))

        if len(pair_imgs) == 0:
            # Keep the training loop from crashing.
            empty_img = torch.empty(0, 1, image_size, image_size, dtype=torch.uint8)
            empty_tok = torch.empty(0, cache_input_ids.shape[1], dtype=torch.long)
            empty_msk = torch.empty(0, cache_attn_mask.shape[1], dtype=torch.long)
            empty_len = torch.empty(0, dtype=torch.long)
            empty_i64 = torch.empty(0, dtype=torch.long)
            return (
                empty_img,
                empty_tok,
                empty_msk,
                empty_len,
                empty_tok,
                empty_msk,
                empty_len,
                empty_i64,  # gender
                empty_i64,  # race
                empty_i64,  # age_bin
                empty_i64,  # label
                [],         # subj_ids
            )

        img_tensor = torch.stack(pair_imgs, dim=0)
        ids1, m1, l1 = _texts_to_tokens(pair_txt_keys)
        ids2, m2, l2 = _texts_to_tokens(pair_txt_keys_u)

        return (
            img_tensor,
            ids1,
            m1,
            l1,
            ids2,
            m2,
            l2,
            torch.tensor(pair_gender, dtype=torch.long),
            torch.tensor(pair_race, dtype=torch.long),
            torch.tensor(pair_age_bin, dtype=torch.long),
            torch.tensor(pair_labels, dtype=torch.long),
            pair_sids,
        )

    return collate


def _normalize_task_name(task: str) -> str:
    t = str(task).strip().lower()
    if t in {"pleural_effusion", "pleural effusion", "effusion"}:
        return "pleural_effusion"
    if t in {"no_finding", "no finding", "nofinding", "no_findings", "no findings"}:
        return "no_finding"
    if t in {"mortality", "mortality_prediction", "expire", "death"}:
        return "mortality"
    raise ValueError(f"Unknown task='{task}'. Use 'pleural_effusion', 'no_finding', or 'mortality'.")


def _pick_default_pathology_labels(mimic_root: Path) -> str:
    # Accept both .csv and .csv.gz
    candidates = [
        mimic_root / "mimic-cxr-2.0.0-chexpert.csv.gz",
        mimic_root / "mimic-cxr-2.0.0-chexpert.csv",
        mimic_root / "mimic-cxr-2.0.0-negbio.csv.gz",
        mimic_root / "mimic-cxr-2.0.0-negbio.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "Pleural effusion / No Finding tasks require a label file under mimic_root. "
        "Expected one of: mimic-cxr-2.0.0-chexpert.csv(.gz) or mimic-cxr-2.0.0-negbio.csv(.gz)."
    )


def _make_label_fns(
    *,
    task: str,
    pathology_labels_csv: Optional[str],
    mimic_root: Path,
    uncertain_policy: str,
):
    """Return callables rel_path -> label (or None to ignore sample)."""

    task = _normalize_task_name(task)
    uncertain_policy = str(uncertain_policy).strip().lower()
    if uncertain_policy not in {"zero", "one", "ignore"}:
        raise ValueError("uncertain_policy must be 'zero', 'one', or 'ignore'")

    if task == "mortality":
        return None, None

    if pathology_labels_csv is None:
        pathology_labels_csv = _pick_default_pathology_labels(mimic_root)

    lab_df = pd.read_csv(pathology_labels_csv, usecols=["study_id", "Pleural Effusion", "No Finding"])
    lab_df["study_id"] = lab_df["study_id"].astype(int)

    effusion_by_study = dict(zip(lab_df["study_id"].to_numpy(), lab_df["Pleural Effusion"].to_numpy()))
    nofinding_by_study = dict(zip(lab_df["study_id"].to_numpy(), lab_df["No Finding"].to_numpy()))

    def _study_id_from_rel_path(rel_path: str) -> Optional[int]:
        rp = str(rel_path).replace("\\", "/")
        m = re.search(r"/s(\d+)/", rp)
        if m is None:
            return None
        return int(m.group(1))

    def _to_bin(v: Any) -> Optional[int]:
        if pd.isna(v):
            return 0
        vv = float(v)
        if vv == 1.0:
            return 1
        if vv == 0.0:
            return 0
        # uncertain (-1)
        if uncertain_policy == "one":
            return 1
        if uncertain_policy == "ignore":
            return None
        return 0

    def effusion_label(rel_path: str) -> Optional[int]:
        sid = _study_id_from_rel_path(rel_path)
        if sid is None:
            return None
        return _to_bin(effusion_by_study.get(sid, np.nan))

    def nofinding_label(rel_path: str) -> Optional[int]:
        sid = _study_id_from_rel_path(rel_path)
        if sid is None:
            return None
        return _to_bin(nofinding_by_study.get(sid, np.nan))

    return effusion_label, nofinding_label


def _load_aug_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["subject_id", "PA"]:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column '{c}'")
    if ("text_augment" not in df.columns) and ("text" not in df.columns):
        raise ValueError(f"{path} missing both text_augment and text columns")
    df["subject_id"] = df["subject_id"].astype(int)
    return df


def _extract_texts_from_row(row: Any) -> List[str]:
    texts: List[str] = []
    if hasattr(row, "text_augment"):
        texts = _parse_list_cell(getattr(row, "text_augment"))
    if len(texts) == 0 and hasattr(row, "text"):
        texts = _parse_list_cell(getattr(row, "text"))
    return _clean_unique_texts(texts)


def _build_subject_records(
    *,
    df: pd.DataFrame,
    split_name: str,
    img_root: Path,
    keep_subject_info: Dict[int, Dict[str, Any]],
    task: str,
    effusion_label_fn,
    nofinding_label_fn,
    max_missing_images: int,
) -> List[SubjectRecord]:
    """Build subject-level records from an augmented split CSV.

    NOTE: The CSV is expected to have 1+ rows per subject. We aggregate across
    rows so we truly keep ALL images/texts available for each subject.
    """

    task = _normalize_task_name(task)

    # Aggregate per subject
    by_subject: Dict[int, Dict[str, Any]] = {}

    # Filter to subjects we can fully annotate (race/gender/age, and label for mortality)
    df = df[df["subject_id"].isin(keep_subject_info.keys())].copy()

    missing_images = 0

    for row in df.itertuples(index=False):
        sid = int(getattr(row, "subject_id"))
        info = keep_subject_info.get(sid)
        if info is None:
            continue

        texts = _extract_texts_from_row(row)
        if len(texts) < 2:
            continue
        pa_list = _parse_list_cell(getattr(row, "PA", None))

        if sid not in by_subject:
            by_subject[sid] = {
                "images": [],
                "texts": [],
            }

        # Texts
        by_subject[sid]["texts"].extend(texts)

        # Images (keep only existing .jpg)
        for rel_path in pa_list:
            rel_path = str(rel_path).strip()
            if not rel_path or (not rel_path.lower().endswith(".jpg")):
                continue
            full = img_root / rel_path
            if not full.exists():
                missing_images += 1
                if missing_images > int(max_missing_images):
                    raise FileNotFoundError(
                        f"[MIMIC-CXR] Too many missing images while building '{split_name}'. "
                        f"Found {missing_images} missing, limit={max_missing_images}. "
                        f"Check mimic_root/official_subdir. Example missing: {rel_path}"
                    )
                continue
            by_subject[sid]["images"].append(rel_path)

    # Convert to records
    out: List[SubjectRecord] = []
    for sid, blob in by_subject.items():
        images_rel = list(dict.fromkeys(blob["images"]))  # unique preserve order
        texts = _clean_unique_texts(blob["texts"])         # unique texts

        if len(images_rel) == 0 or len(texts) == 0:
            continue

        # Map to cache keys
        text_keys = [_sha1_text_id(t) for t in texts]

        # Subject label
        if task == "mortality":
            y = int(keep_subject_info[sid]["label"])

        elif task == "pleural_effusion":
            ys: List[int] = []
            for rp in images_rel:
                v = effusion_label_fn(rp)
                if v is None:
                    continue
                ys.append(int(v))
            if len(ys) == 0:
                continue
            y = int(max(ys))

        elif task == "no_finding":
            ys = []
            for rp in images_rel:
                v = nofinding_label_fn(rp)
                if v is None:
                    continue
                ys.append(int(v))
            if len(ys) == 0:
                continue
            y = int(max(ys))

        else:
            raise RuntimeError("Unhandled task")

        out.append(
            SubjectRecord(
                subject_id=int(sid),
                images_rel=images_rel,
                text_keys=text_keys,
                label=y,
                race_id=int(keep_subject_info[sid]["race_id"]),
                gender_id=int(keep_subject_info[sid]["gender_id"]),
                age=float(keep_subject_info[sid]["age"]),
                age_bin=int(keep_subject_info[sid]["age_bin"]),
            )
        )

    return out

class _DynamicSubjectBatchSampler(Sampler[List[int]]):
    """Greedy pack subjects into batches so that neither modality exceeds max_items_per_batch.

    - Each yielded batch is a list of subject indices.
    - Subjects are never split across batches.
    - If shuffle=True, subject order is reshuffled each epoch (seed+epoch).
    """

    def __init__(
        self,
        *,
        per_subject_img_items: List[int],
        per_subject_txt_items: List[int],
        max_items_per_batch: int,
        shuffle: bool,
        seed: int = 42,
        drop_last: bool = False,
    ):
        assert len(per_subject_img_items) == len(per_subject_txt_items)
        self.img_items = list(map(int, per_subject_img_items))
        self.txt_items = list(map(int, per_subject_txt_items))
        self.max_items = int(max_items_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.img_items)
        indices = list(range(n))

        if self.shuffle:
            rng = random.Random(self.seed + 1000003 * self.epoch)
            rng.shuffle(indices)

        batch = []
        cur_img = 0
        cur_txt = 0

        for idx in indices:
            add_img = self.img_items[idx]
            add_txt = self.txt_items[idx]

            # Retain oversized subjects as single-subject batches.
            if add_img > self.max_items or add_txt > self.max_items:
                if len(batch) > 0:
                    yield batch
                    batch = []
                    cur_img = 0
                    cur_txt = 0
                yield [idx]
                continue

            # Start a new batch when either modality reaches the item limit.
            if (cur_img + add_img > self.max_items) or (cur_txt + add_txt > self.max_items):
                if len(batch) > 0:
                    yield batch
                batch = [idx]
                cur_img = add_img
                cur_txt = add_txt
            else:
                batch.append(idx)
                cur_img += add_img
                cur_txt += add_txt

        if len(batch) > 0 and (not self.drop_last):
            yield batch

    def __len__(self):
        # This is not exact without simulating packing; return a safe approximation.
        # DataLoader doesn't require exact length for training.
        n = len(self.img_items)
        if n == 0:
            return 0
        avg = max(1.0, float(sum(self.img_items) + sum(self.txt_items)) / (2.0 * n))
        est = int((n * avg) / max(1, self.max_items)) + 1
        return max(1, est)


class _DynamicSameLabelSubjectBatchSampler(Sampler[List[int]]):
    """Same as _DynamicSubjectBatchSampler, but batches are homogeneous by label."""

    def __init__(
        self,
        *,
        subject_labels: List[int],
        per_subject_img_items: List[int],
        per_subject_txt_items: List[int],
        max_items_per_batch: int,
        shuffle: bool,
        seed: int = 42,
        drop_last: bool = False,
    ):
        assert len(subject_labels) == len(per_subject_img_items) == len(per_subject_txt_items)
        self.labels = list(map(int, subject_labels))
        self.img_items = list(map(int, per_subject_img_items))
        self.txt_items = list(map(int, per_subject_txt_items))
        self.max_items = int(max_items_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        label_to_indices: Dict[int, List[int]] = {}
        for i, y in enumerate(self.labels):
            label_to_indices.setdefault(int(y), []).append(i)
        self.label_to_indices = label_to_indices
        self.unique_labels = sorted(label_to_indices.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + 1000003 * self.epoch)

        labs = self.unique_labels[:]
        if self.shuffle:
            rng.shuffle(labs)

        for lab in labs:
            idxs = self.label_to_indices[lab][:]
            if self.shuffle:
                rng.shuffle(idxs)

            batch = []
            cur_img = 0
            cur_txt = 0

            for idx in idxs:
                add_img = self.img_items[idx]
                add_txt = self.txt_items[idx]

                if add_img > self.max_items or add_txt > self.max_items:
                    if len(batch) > 0:
                        yield batch
                        batch = []
                        cur_img = 0
                        cur_txt = 0
                    yield [idx]
                    continue

                if (cur_img + add_img > self.max_items) or (cur_txt + add_txt > self.max_items):
                    if len(batch) > 0:
                        yield batch
                    batch = [idx]
                    cur_img = add_img
                    cur_txt = add_txt
                else:
                    batch.append(idx)
                    cur_img += add_img
                    cur_txt += add_txt

            if len(batch) > 0 and (not self.drop_last):
                yield batch

    def __len__(self):
        # Approximation; see _DynamicSubjectBatchSampler.__len__
        n = len(self.labels)
        if n == 0:
            return 0
        avg = max(1.0, float(sum(self.img_items) + sum(self.txt_items)) / (2.0 * n))
        est = int((n * avg) / max(1, self.max_items)) + 1
        return max(1, est)

def build_mimic_cxr_loaders(
    mimic_root: str = "",
    official_subdir: str = "official_data_iccv_final",
    cxr_record_list_csv: str = "",

    train_aug_csv: str = "",
    val_aug_csv:   str = "",
    test_aug_csv:  str = "",

    admissions_csv: str = "",
    patients_csv:   str = "",

    task: str = "no_finding",
    pathology_labels_csv: Optional[str] = None,
    uncertain_policy: str = "zero",

    age_threshold: float = 60.0,

    max_items_per_batch: int = 128,

    num_workers: int = 4,
    pin_memory: bool = torch.cuda.is_available(),

    clip_model_name: str = "openai/clip-vit-base-patch32",
    cap_max_len: int = 77,
    text_cache_path: str = "",

    loss_type: str = "M1",
    loss_family: Optional[str] = None,

    pairs_per_subject: Union[int, str] = "all",
    max_pairs_per_subject: Optional[int] = None,

    image_size: int = 224,
    base_seed: int = 12345,
    max_missing_images: int = 40,

    max_items_per_subject: int = 0,
):
    task = _normalize_task_name(task)
    mimic_root_p = Path(mimic_root)
    img_root = mimic_root_p / official_subdir

    loss_type_u = str(loss_type).strip().upper()
    family_u = None if loss_family is None else str(loss_family).strip().lower()

    pad_pairs = False
    if family_u in {None, "vicreg", "fairwell"} and loss_type_u in {"M1", "M2", "M3", "M4"}:
        output_mode = "bag"
    else:
        output_mode = "paired"

    # -----------------------------
    # Load cache
    # -----------------------------
    cache = torch.load(text_cache_path, map_location="cpu")
    if cache.get("kind") != "mimic_text_sha1":
        raise ValueError(f"MIMIC cache kind mismatch: {cache.get('kind')}")
    if str(cache.get("clip_model_name")) != str(clip_model_name):
        raise ValueError(
            f"MIMIC cache clip_model_name mismatch: {cache.get('clip_model_name')} != {clip_model_name}"
        )
    if int(cache.get("max_len")) != int(cap_max_len):
        raise ValueError(f"MIMIC cache max_len mismatch: {cache.get('max_len')} != {cap_max_len}")

    key_to_row = {k: i for i, k in enumerate(cache["keys"])}
    cache_input_ids = cache["input_ids"].to(torch.long)
    cache_attn_mask = cache["attention_mask"].to(torch.long)
    cache_lengths = cache["lengths"].to(torch.long)

    # -----------------------------
    # Core subject inclusion filters
    # -----------------------------
    cxr_df = pd.read_csv(cxr_record_list_csv, usecols=["subject_id"])
    cxr_df["subject_id"] = cxr_df["subject_id"].astype(int)
    subjects_in_cxr = set(cxr_df["subject_id"].unique().tolist())

    adm_df = pd.read_csv(admissions_csv, usecols=["subject_id", "race", "hospital_expire_flag"])
    adm_df["subject_id"] = adm_df["subject_id"].astype(int)
    adm_df["hospital_expire_flag"] = adm_df["hospital_expire_flag"].astype(int)
    adm_df = adm_df.dropna(subset=["race"])

    subject_mort_label: Dict[int, int] = {}
    if task == "mortality":
        subject_mort_label = adm_df.groupby("subject_id")["hospital_expire_flag"].max().to_dict()

    races_by_subject = adm_df.groupby("subject_id")["race"].apply(list).to_dict()
    subject_race_raw: Dict[int, str] = {}
    excluded_subjects = set()

    for sid, races in races_by_subject.items():
        rs = [str(x).strip().upper() for x in races if str(x).strip()]
        if any(r in EXCLUDE_RACES for r in rs):
            excluded_subjects.add(int(sid))
            continue
        subject_race_raw[int(sid)] = str(races[0])

    pat_df = pd.read_csv(patients_csv, usecols=["subject_id", "gender", "anchor_age"])
    pat_df["subject_id"] = pat_df["subject_id"].astype(int)
    pat_df = pat_df.dropna(subset=["gender", "anchor_age"])
    pat_df = pat_df.drop_duplicates(subset=["subject_id"], keep="first")

    subject_gender_age: Dict[int, Tuple[int, float]] = {}
    for r in pat_df.itertuples(index=False):
        sid = int(r.subject_id)
        g = str(r.gender).strip().upper()
        gender_id = 1 if g == "M" else 0
        subject_gender_age[sid] = (gender_id, float(r.anchor_age))

    subjects_core = set(subject_race_raw.keys()) & set(subject_gender_age.keys()) & subjects_in_cxr
    if task == "mortality":
        subjects_core = subjects_core & set(subject_mort_label.keys())

    subjects_core = subjects_core - excluded_subjects

    keep_subject_info: Dict[int, Dict[str, Any]] = {}
    for sid in subjects_core:
        grp = _race_to_group_optionA(subject_race_raw[sid])
        if grp is None:
            continue
        gender_id, age = subject_gender_age[sid]
        keep_subject_info[sid] = {
            "race_id": int(RACE_TO_ID[grp]),
            "gender_id": int(gender_id),
            "age": float(age),
            "age_bin": int(float(age) > float(age_threshold)),
        }
        if task == "mortality":
            keep_subject_info[sid]["label"] = int(subject_mort_label[sid])

    # -----------------------------
    # Label mapping for pathology tasks
    # -----------------------------
    effusion_label_fn, nofinding_label_fn = _make_label_fns(
        task=task,
        pathology_labels_csv=pathology_labels_csv,
        mimic_root=mimic_root_p,
        uncertain_policy=uncertain_policy,
    )

    # -----------------------------
    # Load split CSVs and build records
    # -----------------------------
    train_df = _load_aug_df(train_aug_csv)
    val_df = _load_aug_df(val_aug_csv)
    test_df = _load_aug_df(test_aug_csv)

    train_records = _build_subject_records(
        df=train_df,
        split_name="train",
        img_root=img_root,
        keep_subject_info=keep_subject_info,
        task=task,
        effusion_label_fn=effusion_label_fn,
        nofinding_label_fn=nofinding_label_fn,
        max_missing_images=max_missing_images,
    )
    val_records = _build_subject_records(
        df=val_df,
        split_name="val",
        img_root=img_root,
        keep_subject_info=keep_subject_info,
        task=task,
        effusion_label_fn=effusion_label_fn,
        nofinding_label_fn=nofinding_label_fn,
        max_missing_images=max_missing_images,
    )
    test_records = _build_subject_records(
        df=test_df,
        split_name="test",
        img_root=img_root,
        keep_subject_info=keep_subject_info,
        task=task,
        effusion_label_fn=effusion_label_fn,
        nofinding_label_fn=nofinding_label_fn,
        max_missing_images=max_missing_images,
    )

    # Dataset (caps per-subject modality items)
    train_ds = MIMICCXRSubjectDataset(train_records, base_seed=base_seed, max_items_per_subject=max_items_per_subject)
    val_ds = MIMICCXRSubjectDataset(val_records, base_seed=base_seed, max_items_per_subject=max_items_per_subject)
    test_ds = MIMICCXRSubjectDataset(test_records, base_seed=base_seed, max_items_per_subject=max_items_per_subject)

    # Collate
    collate_fn = _make_collate_fn(
        img_root=img_root,
        cache_input_ids=cache_input_ids,
        cache_attn_mask=cache_attn_mask,
        cache_lengths=cache_lengths,
        key_to_row=key_to_row,
        output_mode=output_mode,
        pairs_per_subject=pairs_per_subject,
        max_pairs_per_subject=max_pairs_per_subject,
        pad_pairs=pad_pairs,
        image_size=image_size,
        base_seed=base_seed,
    )

    # -----------------------------
    # Compute per-subject "item contributions" for batching
    # -----------------------------
    def _per_subject_items(records: List[SubjectRecord]) -> Tuple[List[int], List[int]]:
        img_items = []
        txt_items = []
        m = int(max_items_per_subject)

        for r in records:
            n_img = min(len(r.images_rel), m) if m > 0 else len(r.images_rel)
            n_txt = min(len(r.text_keys), m) if m > 0 else len(r.text_keys)

            if str(output_mode).lower() == "bag":
                img_items.append(int(n_img))
                txt_items.append(int(n_txt))
            else:
                if isinstance(pairs_per_subject, str) and pairs_per_subject.strip().lower() == "all":
                    k = max(int(n_img), int(n_txt))
                else:
                    k = int(pairs_per_subject)

                if max_pairs_per_subject is not None:
                    k = min(k, int(max_pairs_per_subject))

                img_items.append(int(k))
                txt_items.append(int(k))

        return img_items, txt_items

    train_img_items, train_txt_items = _per_subject_items(train_records)
    val_img_items, val_txt_items = _per_subject_items(val_records)
    test_img_items, test_txt_items = _per_subject_items(test_records)

    # -----------------------------
    # Batch samplers
    # -----------------------------
    train_batch_sampler_ml = _DynamicSubjectBatchSampler(
        per_subject_img_items=train_img_items,
        per_subject_txt_items=train_txt_items,
        max_items_per_batch=max_items_per_batch,
        shuffle=True,
        seed=base_seed,
        drop_last=True,
    )
    val_batch_sampler_ml = _DynamicSubjectBatchSampler(
        per_subject_img_items=val_img_items,
        per_subject_txt_items=val_txt_items,
        max_items_per_batch=max_items_per_batch,
        shuffle=False,
        seed=base_seed,
        drop_last=False,
    )
    test_batch_sampler = _DynamicSubjectBatchSampler(
        per_subject_img_items=test_img_items,
        per_subject_txt_items=test_txt_items,
        max_items_per_batch=max_items_per_batch,
        shuffle=False,
        seed=base_seed,
        drop_last=False,
    )

    train_batch_sampler_sl = _DynamicSameLabelSubjectBatchSampler(
        subject_labels=train_ds.subject_labels,
        per_subject_img_items=train_img_items,
        per_subject_txt_items=train_txt_items,
        max_items_per_batch=max_items_per_batch,
        shuffle=True,
        seed=base_seed,
        drop_last=True,
    )
    val_batch_sampler_sl = _DynamicSameLabelSubjectBatchSampler(
        subject_labels=val_ds.subject_labels,
        per_subject_img_items=val_img_items,
        per_subject_txt_items=val_txt_items,
        max_items_per_batch=max_items_per_batch,
        shuffle=False,
        seed=base_seed,
        drop_last=False,
    )

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=False,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 1

    train_loader_ml = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler_ml, **dl_kwargs)
    val_loader_ml = DataLoader(dataset=val_ds, batch_sampler=val_batch_sampler_ml, **dl_kwargs)

    train_loader_sl = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler_sl, **dl_kwargs)
    val_loader_sl = DataLoader(dataset=val_ds, batch_sampler=val_batch_sampler_sl, **dl_kwargs)

    test_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, **dl_kwargs)

    meta = {
        "dataset": "mimic-cxr-optionA-subject",
        "task": task,
        "loss_type": str(loss_type_u),
        "loss_family": (None if family_u is None else str(family_u)),
        "race_groups": list(RACE_GROUPS),
        "race_cardinality": 4,
        "sensitive_attrs": ["race", "gender", "age_bin"],
        "cap_max_len": int(cap_max_len),
        "clip_model_name": str(clip_model_name),

        "output_mode": str(output_mode),
        "pairs_per_subject": pairs_per_subject,
        "max_pairs_per_subject": max_pairs_per_subject,
        "pad_pairs": bool(pad_pairs),

        "image_size": int(image_size),
        "max_items_per_batch": int(max_items_per_batch),
        "max_items_per_subject": int(max_items_per_subject),
        "paths": {
            "mimic_root": str(mimic_root_p),
            "official_subdir": str(official_subdir),
            "train_aug_csv": str(train_aug_csv),
            "val_aug_csv": str(val_aug_csv),
            "test_aug_csv": str(test_aug_csv),
            "admissions_csv": str(admissions_csv),
            "patients_csv": str(patients_csv),
            "cxr_record_list_csv": str(cxr_record_list_csv),
            "pathology_labels_csv": str(pathology_labels_csv) if pathology_labels_csv is not None else None,
            "text_cache_path": str(text_cache_path),
        },
        "counts": {
            "train_subjects": len(train_records),
            "val_subjects": len(val_records),
            "test_subjects": len(test_records),
        },
    }

    return train_loader_sl, val_loader_sl, train_loader_ml, val_loader_ml, test_loader, meta
