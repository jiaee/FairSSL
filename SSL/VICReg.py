import torch
import torch.nn as nn
import torch.nn.functional as F

### Utils

def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.InstanceNorm1d(f[i + 1])) 
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

### M1

def _to_id_list(ids):
    if ids is None:
        return None
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    return [str(x) for x in ids]

def _group_indices(ids_list):
    d = {}
    for i, sid in enumerate(ids_list):
        d.setdefault(sid, []).append(i)
    return d

class VICReg_M1(nn.Module):
    """
    FAIRWELL M1 (intra-subject):
      IFWreg: pooled(modality A subject i) vs segments(modality B subject i)
      Vreg/Creg: computed per-subject, per-modality (then averaged)
    Requires subject ids.
    """
    def __init__(self, args, eps: float = 1e-4):
        super().__init__()
        self.args = args
        self.eps = float(eps)
        self.a_projector = None
        self.b_projector = None

    def _init_projectors_if_needed(self, z_a, z_b):
        if self.a_projector is None:
            self.a_projector = Projector(self.args, z_a.size(-1)).to(z_a.device)
        if self.b_projector is None:
            self.b_projector = Projector(self.args, z_b.size(-1)).to(z_b.device)

    def _vreg(self, z: torch.Tensor) -> torch.Tensor:
        # z: [n, d] for ONE subject in ONE modality
        n = int(z.size(0))
        if n < 2:
            return z.new_tensor(0.0)
        zc = z - z.mean(dim=0)
        std = torch.sqrt(zc.var(dim=0, unbiased=False) + self.eps)
        return F.relu(1.0 - std).mean()

    def _creg(self, z: torch.Tensor) -> torch.Tensor:
        n = int(z.size(0))
        d = int(z.size(1))
        if n < 2:
            return z.new_tensor(0.0)
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / float(n - 1)
        return off_diagonal(cov).pow(2).sum() / float(d)

    def forward(self, z_a, z_b, subject_ids_a=None, subject_ids_b=None):
        """
        z_a: [N_a, D_a] (pooling modality)
        z_b: [N_b, D_b] (segment modality)
        subject_ids_*: length N_* (tensor or list)
        """
        ids_a = _to_id_list(subject_ids_a)
        ids_b = _to_id_list(subject_ids_b if subject_ids_b is not None else subject_ids_a)
        if ids_a is None or ids_b is None:
            raise ValueError("FAIRWELL M1 requires subject_ids_a/subject_ids_b.")

        self._init_projectors_if_needed(z_a, z_b)
        za = self.a_projector(z_a)
        zb = self.b_projector(z_b)

        subj2idx_a = _group_indices(ids_a)
        subj2idx_b = _group_indices(ids_b)
        common_sids = list(set(subj2idx_a.keys()) & set(subj2idx_b.keys()))
        if len(common_sids) == 0:
            return za.new_tensor(0.0)

        device = za.device

        # ---- IFWreg (intra-subject only) ----
        repr_total = za.new_tensor(0.0)
        pair_count = 0

        for sid in common_sids:
            ia = torch.as_tensor(subj2idx_a[sid], device=device, dtype=torch.long)
            ib = torch.as_tensor(subj2idx_b[sid], device=device, dtype=torch.long)

            if ia.numel() == 0 or ib.numel() == 0:
                continue

            pooled_a = za.index_select(0, ia).mean(dim=0, keepdim=True)  # [1, D]
            seg_b = zb.index_select(0, ib)                               # [n_b, D]

            # mean_j || seg_b[j] - pooled_a ||^2
            repr_total = repr_total + (seg_b - pooled_a).pow(2).sum(dim=1).mean()
            pair_count += 1

        repr_loss = repr_total / max(pair_count, 1)

        # ---- Vreg/Creg (per-subject, per-modality) ----
        v_sum = za.new_tensor(0.0)
        c_sum = za.new_tensor(0.0)
        v_cnt = 0

        for sid in common_sids:
            ia = torch.as_tensor(subj2idx_a[sid], device=device, dtype=torch.long)
            ib = torch.as_tensor(subj2idx_b[sid], device=device, dtype=torch.long)
            if ia.numel() > 0:
                v_sum = v_sum + self._vreg(za.index_select(0, ia))
                c_sum = c_sum + self._creg(za.index_select(0, ia))
            if ib.numel() > 0:
                v_sum = v_sum + self._vreg(zb.index_select(0, ib))
                c_sum = c_sum + self._creg(zb.index_select(0, ib))
            v_cnt += 1

        v_loss = v_sum / max(v_cnt, 1)
        c_loss = c_sum / max(v_cnt, 1)

        return (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * v_loss
            + self.args.cov_coeff * c_loss
        )


class VICReg_M234(nn.Module):
    """
    FAIRWELL M2/M3/M4-style (inter-subject):
      IFWreg: pooled(modality A subject i) vs segments(modality B subject k) for ALL i,k
      Vreg/Creg: mean per-subject per-modality (normalization matches paper after 1/|B|^2)
    Handles unequal segments & unequal subject sets across modalities safely.
    """
    def __init__(self, in_dim_a, in_dim_b, args, eps: float = 1e-4):
        super().__init__()
        self.args = args
        self.eps = float(eps)
        self.a_projector = Projector(args, int(in_dim_a))
        self.b_projector = Projector(args, int(in_dim_b))

    def _vreg(self, z: torch.Tensor) -> torch.Tensor:
        n = int(z.size(0))
        if n < 2:
            return z.new_tensor(0.0)
        zc = z - z.mean(dim=0)
        std = torch.sqrt(zc.var(dim=0, unbiased=False) + self.eps)
        return F.relu(1.0 - std).mean()

    def _creg(self, z: torch.Tensor) -> torch.Tensor:
        n = int(z.size(0))
        d = int(z.size(1))
        if n < 2:
            return z.new_tensor(0.0)
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / float(n - 1)
        return off_diagonal(cov).pow(2).sum() / float(d)

    def forward(self, z_a, z_b, subject_ids_a, subject_ids_b):
        ids_a = _to_id_list(subject_ids_a)
        ids_b = _to_id_list(subject_ids_b)
        if ids_a is None or ids_b is None:
            raise ValueError("FAIRWELL M234 requires subject_ids_a and subject_ids_b.")

        za = self.a_projector(z_a)
        zb = self.b_projector(z_b)

        subj2idx_a = _group_indices(ids_a)
        subj2idx_b = _group_indices(ids_b)

        sids_a = list(subj2idx_a.keys())
        sids_b = list(subj2idx_b.keys())
        if len(sids_a) == 0 or len(sids_b) == 0:
            return za.new_tensor(0.0)

        device = za.device

        # Precompute pooled A per subject and segment tensors per B subject
        pooled_a = {}
        for sid in sids_a:
            ia = torch.as_tensor(subj2idx_a[sid], device=device, dtype=torch.long)
            if ia.numel() == 0:
                continue
            pooled_a[sid] = za.index_select(0, ia).mean(dim=0, keepdim=True)  # [1,D]

        segs_b = {}
        for sid in sids_b:
            ib = torch.as_tensor(subj2idx_b[sid], device=device, dtype=torch.long)
            if ib.numel() == 0:
                continue
            segs_b[sid] = zb.index_select(0, ib)  # [n,D]

        if len(pooled_a) == 0 or len(segs_b) == 0:
            return za.new_tensor(0.0)

        # ---- IFWreg over ALL subject pairs (i,k) ----
        repr_total = za.new_tensor(0.0)
        pair_count = 0

        for sid_i, p_i in pooled_a.items():
            for sid_k, b_k in segs_b.items():
                repr_total = repr_total + (b_k - p_i).pow(2).sum(dim=1).mean()
                pair_count += 1

        repr_loss = repr_total / max(pair_count, 1)

        # ---- Vreg/Creg: mean over subjects in each modality ----
        v_a = za.new_tensor(0.0)
        c_a = za.new_tensor(0.0)
        cnt_a = 0
        for sid in sids_a:
            ia = torch.as_tensor(subj2idx_a[sid], device=device, dtype=torch.long)
            if ia.numel() == 0:
                continue
            z = za.index_select(0, ia)
            v_a = v_a + self._vreg(z)
            c_a = c_a + self._creg(z)
            cnt_a += 1

        v_b = za.new_tensor(0.0)
        c_b = za.new_tensor(0.0)
        cnt_b = 0
        for sid in sids_b:
            ib = torch.as_tensor(subj2idx_b[sid], device=device, dtype=torch.long)
            if ib.numel() == 0:
                continue
            z = zb.index_select(0, ib)
            v_b = v_b + self._vreg(z)
            c_b = c_b + self._creg(z)
            cnt_b += 1

        v_loss = (v_a / max(cnt_a, 1)) + (v_b / max(cnt_b, 1))
        c_loss = (c_a / max(cnt_a, 1)) + (c_b / max(cnt_b, 1))

        return (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * v_loss
            + self.args.cov_coeff * c_loss
        )
