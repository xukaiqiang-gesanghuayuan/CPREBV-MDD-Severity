# -*- coding: utf-8 -*-
"""
PREBV / CPREBV training script.

Changes in this version:
1) Removed gradient accumulation: one optimizer update per mini-batch.
2) Load demographic covariates from Excel: ID, Sex, Age, Education (years).
3) Use /home/xukaiqiang/004 as the default project/data root.
4) Match GM/WM/CSF .npy files with demographic rows by subject ID, e.g. S1-1-0001 or S1-2-0001.

Expected directory layout:
/home/xukaiqiang/004/
    2.xlsx or 2(2).xlsx
    mwc1npytrain/
    mwc2npytrain/
    mwc3npytrain/
    mwc1npyval/
    mwc2npyval/
    mwc3npyval/
"""

import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:  # compatible with older timm versions
    from timm.models.layers import DropPath, trunc_normal_


# ==================== Config ====================
SEED = 19
DEVICE_ID = 2
NUM_CLASSES = 2              # 二分类：0=MDD, 1=HC
POS_LABEL = 0                # 以 MDD 为阳性类计算 Precision/Recall/SEN/F1/AUC
BATCH_SIZE = 8
NUM_EPOCHS = 200
LR = 1e-4
GEO_REG_WEIGHT = 0.1
NUM_WORKERS = 4

DATA_BASE = Path("/home/xukaiqiang/004")

DEMO_EXCEL_CANDIDATES = [
    DATA_BASE / "2.xlsx",
    DATA_BASE / "2(2).xlsx",
    DATA_BASE / "MDDdata.xlsx",
    DATA_BASE / "MDDdata1.xlsx",
]

TRAIN_ROOTS = (
    DATA_BASE / "mwc1npytrain",  # GM
    DATA_BASE / "mwc2npytrain",  # WM
    DATA_BASE / "mwc3npytrain",  # CSF
)
VAL_ROOTS = (
    DATA_BASE / "mwc1npyval",
    DATA_BASE / "mwc2npyval",
    DATA_BASE / "mwc3npyval",
)

SAVE_PATH = DATA_BASE / "best_prebv_model.pth"

ID_PATTERN = re.compile(r"S\d+-[12]-\d+", re.IGNORECASE)
DEMO_COLUMNS = ["Sex", "Age", "Education (years)"]


def set_seed(seed: int = 19) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_existing_excel(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    candidate_text = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot find demographic Excel. Checked:\n{candidate_text}")


def extract_subject_id(path_or_name) -> str:
    """Extract subject ID such as S1-1-0001 from a file path/name."""
    name = Path(path_or_name).stem
    match = ID_PATTERN.search(name)
    if match:
        return match.group(0).upper()
    return name.upper()


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric and treat -9999 as missing."""
    out = pd.to_numeric(series, errors="coerce")
    out = out.replace(-9999, np.nan)
    return out


def _clean_sex(series: pd.Series) -> pd.Series:
    """
    Convert Sex to numeric. The uploaded sheet already uses 1/2.
    This also supports M/F just in case.
    """
    mapped = series.replace({
        "M": 1, "Male": 1, "male": 1, "m": 1,
        "F": 2, "Female": 2, "female": 2, "f": 2,
    })
    return _clean_numeric(mapped)


def load_demographic_table(excel_path: Path) -> pd.DataFrame:
    """
    Read MDD and Controls sheets.
    Output columns: ID, Sex, Age, Education (years), label.
    label: 0=MDD, 1=HC.
    """
    mdd_df = pd.read_excel(excel_path, sheet_name="MDD")
    hc_df = pd.read_excel(excel_path, sheet_name="Controls")

    required = ["ID", "Sex", "Age", "Education (years)"]
    for sheet_name, df in [("MDD", mdd_df), ("Controls", hc_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Sheet {sheet_name} is missing columns: {missing}. Existing columns: {list(df.columns)}")

    mdd_df = mdd_df[required].copy()
    hc_df = hc_df[required].copy()
    mdd_df["label"] = 0
    hc_df["label"] = 1

    demo_df = pd.concat([mdd_df, hc_df], ignore_index=True)
    demo_df["ID"] = demo_df["ID"].astype(str).str.strip().str.upper()
    demo_df["Sex"] = _clean_sex(demo_df["Sex"])
    demo_df["Age"] = _clean_numeric(demo_df["Age"])
    demo_df["Education (years)"] = _clean_numeric(demo_df["Education (years)"])
    demo_df = demo_df.dropna(subset=["ID", "label"])
    demo_df = demo_df.drop_duplicates(subset=["ID"], keep="first")
    demo_df = demo_df.set_index("ID", drop=False)
    return demo_df


def collect_npy_by_id(root: Path, valid_ids: set) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Npy directory does not exist: {root}")

    id_to_path: Dict[str, Path] = {}
    duplicate_ids = []
    for file_path in sorted(root.rglob("*.npy")):
        subject_id = extract_subject_id(file_path)
        if subject_id not in valid_ids:
            continue
        if subject_id in id_to_path:
            duplicate_ids.append(subject_id)
            continue
        id_to_path[subject_id] = file_path

    if duplicate_ids:
        print(f"Warning: found duplicate npy IDs in {root}; kept the first file. Examples: {duplicate_ids[:5]}")
    return id_to_path


def load_npy_as_tensor(path: Path) -> torch.Tensor:
    arr = np.load(path)
    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Expected shape: [D, H, W]. Convert to [C, D, H, W].
    if arr.ndim == 3:
        arr = arr[None, ...]
    elif arr.ndim == 4:
        # If channel is last, move it to first. If already channel-first, keep it.
        if arr.shape[-1] == 1 and arr.shape[0] != 1:
            arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Unsupported npy shape {arr.shape} in file: {path}")

    return torch.from_numpy(arr)


class TripleNpyDemographicDataset(Dataset):
    """
    Dataset for GM/WM/CSF .npy files plus demographic covariates.

    Returns:
        ((gm, wm, csf), label, demo_info)
    where demo_info = standardized [Sex, Age, Education].
    """

    def __init__(
        self,
        root_dir1: Path,
        root_dir2: Path,
        root_dir3: Path,
        demo_df: pd.DataFrame,
        demo_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.roots = (Path(root_dir1), Path(root_dir2), Path(root_dir3))
        self.demo_df = demo_df
        self.valid_ids = set(demo_df.index.tolist())

        files1 = collect_npy_by_id(self.roots[0], self.valid_ids)
        files2 = collect_npy_by_id(self.roots[1], self.valid_ids)
        files3 = collect_npy_by_id(self.roots[2], self.valid_ids)

        common_ids = sorted(set(files1) & set(files2) & set(files3))
        if not common_ids:
            raise RuntimeError(
                "No common subject IDs were found across GM/WM/CSF directories and demographic table.\n"
                f"GM: {self.roots[0]}\nWM: {self.roots[1]}\nCSF: {self.roots[2]}"
            )

        self.samples = []
        raw_demo_values = []
        for subject_id in common_ids:
            row = demo_df.loc[subject_id]
            label = int(row["label"])
            raw_demo = row[DEMO_COLUMNS].to_numpy(dtype=np.float32)
            raw_demo_values.append(raw_demo)
            self.samples.append((subject_id, files1[subject_id], files2[subject_id], files3[subject_id], label, raw_demo))

        raw_demo_values = np.stack(raw_demo_values, axis=0)
        if demo_stats is None:
            mean = np.nanmean(raw_demo_values, axis=0)
            std = np.nanstd(raw_demo_values, axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            self.demo_stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
        else:
            self.demo_stats = demo_stats

        labels = np.array([sample[4] for sample in self.samples])
        print(
            f"Loaded dataset from {self.roots[0].parent}: n={len(self.samples)}, "
            f"MDD(label=0)={(labels == 0).sum()}, HC(label=1)={(labels == 1).sum()}"
        )
        print(f"Demo mean used for standardization: {self.demo_stats['mean']}")
        print(f"Demo std used for standardization:  {self.demo_stats['std']}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        subject_id, gm_path, wm_path, csf_path, label, raw_demo = self.samples[index]
        gm = load_npy_as_tensor(gm_path)
        wm = load_npy_as_tensor(wm_path)
        csf = load_npy_as_tensor(csf_path)

        mean = self.demo_stats["mean"]
        std = self.demo_stats["std"]
        demo = np.where(np.isnan(raw_demo), mean, raw_demo)
        demo = (demo - mean) / std
        demo = torch.from_numpy(demo.astype(np.float32))

        return (gm, wm, csf), torch.tensor(label, dtype=torch.long), demo


# ==================== ConvNeXt 3D Backbone ====================
class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first 3D tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data_format: {data_format}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


class Block(nn.Module):
    """ConvNeXt block adapted for 3D inputs."""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)
        return shortcut + self.drop_path(x)


class ConvNeXt3D(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        depths: Tuple[int, int, int, int] = (1, 1, 3, 1),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
                nn.BatchNorm3d(dims[0]),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.BatchNorm3d(dims[i]),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            self.stages.append(
                nn.Sequential(
                    *[
                        Block(dims[i], dp_rates[cur + j], layer_scale_init_value)
                        for j in range(depths[i])
                    ]
                )
            )
            cur += depths[i]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        outputs = []
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)
            outputs.append(x)
        return outputs


# ==================== PRF: Procrustes Orthogonal Fusion ====================
class ProcrustesOrthogonalFusion(nn.Module):
    """
    Learnable orthogonal feature alignment for WM/CSF to the GM feature space.
    The rotation matrices are projected to the orthogonal manifold in each forward pass.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        eye = torch.eye(feature_dim)
        self.rot_wm = nn.Parameter(eye + 1e-4 * torch.randn(feature_dim, feature_dim))
        self.rot_csf = nn.Parameter(eye + 1e-4 * torch.randn(feature_dim, feature_dim))

    @staticmethod
    def _orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u @ vh

    @staticmethod
    def _rotate_channels(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W], rotation: [C, C]
        return torch.einsum("bcdhw,co->bodhw", x, rotation)

    def forward(self, f_gm: torch.Tensor, f_wm: torch.Tensor, f_csf: torch.Tensor) -> torch.Tensor:
        mean_gm = f_gm.mean(dim=(2, 3, 4), keepdim=True)
        mean_wm = f_wm.mean(dim=(2, 3, 4), keepdim=True)
        mean_csf = f_csf.mean(dim=(2, 3, 4), keepdim=True)

        f_wm_centered = f_wm - mean_wm
        f_csf_centered = f_csf - mean_csf

        rot_wm = self._orthogonalize(self.rot_wm)
        rot_csf = self._orthogonalize(self.rot_csf)

        f_wm_aligned = self._rotate_channels(f_wm_centered, rot_wm) + mean_gm
        f_csf_aligned = self._rotate_channels(f_csf_centered, rot_csf) + mean_gm

        return f_gm + f_wm_aligned + f_csf_aligned


# ==================== D-GG: Demographic-aware Gating ====================
class DemographicGating(nn.Module):
    def __init__(self, feature_dim: int, demo_dim: int):
        super().__init__()
        self.demo_proj = nn.Linear(demo_dim, feature_dim)
        self.w_t = nn.Linear(feature_dim * 2, feature_dim)
        self.w_s = nn.Linear(feature_dim * 2, feature_dim)
        self.w_o = nn.Linear(feature_dim, feature_dim)
        self.b_t = nn.Parameter(torch.zeros(feature_dim))
        self.b_s = nn.Parameter(torch.zeros(feature_dim))
        self.b_o = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, fused_feature: torch.Tensor, demo_info: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = fused_feature.shape
        pooled_feature = F.adaptive_avg_pool3d(fused_feature, 1).view(b, c)
        demo_feature = self.demo_proj(demo_info.float())
        fused_demo = torch.cat([pooled_feature, demo_feature], dim=1)

        u_tanh = torch.tanh(self.w_t(fused_demo) + self.b_t)
        u_sigma = torch.sigmoid(self.w_s(fused_demo) + self.b_s)
        gate = self.w_o(u_tanh * u_sigma) + self.b_o
        return fused_feature * gate.view(b, c, 1, 1, 1)


# ==================== RMP-EBV: Residual Multi-Prototype EBV Head ====================
class ResidualMultiPrototypeEBV(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        num_prototypes_per_class: int = 3,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.temperature = temperature

        self.register_buffer("ebv_anchors", self._init_equiangular_frame())
        self.class_residuals = nn.Parameter(torch.zeros(num_classes, feature_dim))
        self.prototype_residuals = nn.Parameter(
            torch.zeros(num_classes, num_prototypes_per_class, feature_dim)
        )

    def _init_equiangular_frame(self) -> torch.Tensor:
        if self.feature_dim < self.num_classes:
            raise ValueError("feature_dim should be >= num_classes for simplex EBV anchors.")

        anchors = torch.eye(self.num_classes)
        anchors = anchors - anchors.mean(dim=0, keepdim=True)
        anchors = F.pad(anchors, (0, self.feature_dim - self.num_classes))
        return F.normalize(anchors, p=2, dim=1)

    def _class_directions(self) -> torch.Tensor:
        return F.normalize(self.ebv_anchors + self.class_residuals, p=2, dim=1)

    def forward(self, x: torch.Tensor):
        v_hat = F.normalize(x, p=2, dim=1)
        class_directions = self._class_directions()                              # [K, D]
        prototype_directions = class_directions[:, None, :] + self.prototype_residuals
        prototype_directions = F.normalize(prototype_directions, p=2, dim=2)     # [K, M, D]

        similarities = torch.einsum("bd,kmd->bkm", v_hat, prototype_directions)
        similarities = similarities / self.temperature
        logits = torch.logsumexp(similarities, dim=2) - np.log(self.num_prototypes_per_class)
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities

    def get_geometry_regularization(self) -> torch.Tensor:
        lambda_class = 0.01
        lambda_proto = 0.01
        lambda_angle = 0.01
        alpha = -1.0 / (self.num_classes - 1) if self.num_classes > 1 else 0.0

        class_residual_loss = lambda_class * self.class_residuals.pow(2).sum()
        prototype_residual_loss = lambda_proto * self.prototype_residuals.pow(2).sum()

        class_directions = self._class_directions()
        gram = class_directions @ class_directions.t()
        off_diag = ~torch.eye(self.num_classes, dtype=torch.bool, device=gram.device)
        angle_loss = (gram[off_diag] - alpha).pow(2).mean()
        angle_regularization = lambda_angle * angle_loss

        return class_residual_loss + prototype_residual_loss + angle_regularization


# ==================== Full Model ====================
class TripleInputConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        num_classes: int = 2,
        depths: Tuple[int, int, int, int] = (1, 1, 3, 1),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        demo_dim: int = 3,
    ):
        super().__init__()
        self.features1 = ConvNeXt3D(in_chans, depths, dims, drop_path_rate, layer_scale_init_value)  # GM
        self.features2 = ConvNeXt3D(in_chans, depths, dims, drop_path_rate, layer_scale_init_value)  # WM
        self.features3 = ConvNeXt3D(in_chans, depths, dims, drop_path_rate, layer_scale_init_value)  # CSF
        self.prf_fusion = ProcrustesOrthogonalFusion(dims[-1])
        self.demo_gating = DemographicGating(dims[-1], demo_dim)
        self.ebv_classifier = ResidualMultiPrototypeEBV(
            dims[-1], num_classes=num_classes, num_prototypes_per_class=3, temperature=0.1
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        demo_info: torch.Tensor,
    ):
        f_gm = self.features1(x1)[-1]
        f_wm = self.features2(x2)[-1]
        f_csf = self.features3(x3)[-1]

        fused_feature = self.prf_fusion(f_gm, f_wm, f_csf)
        fused_feature = self.demo_gating(fused_feature, demo_info)

        global_feature = F.adaptive_avg_pool3d(fused_feature, 1).flatten(1)
        logits, probabilities = self.ebv_classifier(global_feature)
        return logits, probabilities, global_feature


def unpack_batch(batch, device: torch.device):
    (images1, images2, images3), labels, demo_info = batch
    images1 = images1.to(device, non_blocking=True)
    images2 = images2.to(device, non_blocking=True)
    images3 = images3.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).long()
    demo_info = demo_info.to(device, non_blocking=True).float()
    return images1, images2, images3, labels, demo_info


def binary_metrics(y_true, y_pred, y_prob=None, pos_label: int = 0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    neg_label = 1 - pos_label

    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, neg_label])
    if cm.shape != (2, 2):
        return {"cm": cm}

    tp, fn, fp, tn = cm.ravel()
    acc = (tp + tn) / max(tp + fn + fp + tn, 1)
    pre = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    spe = tn / max(tn + fp, 1)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    auc_value = np.nan
    if y_prob is not None and len(np.unique(y_true)) == 2:
        y_bin = (y_true == pos_label).astype(int)
        try:
            auc_value = roc_auc_score(y_bin, y_prob)
        except ValueError:
            auc_value = np.nan

    return {
        "cm": cm,
        "acc": acc,
        "precision": pre,
        "recall": rec,
        "specificity": spe,
        "f1": f1,
        "auc": auc_value,
        "mcc": mcc,
    }


def main():
    set_seed(SEED)
    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

    excel_path = find_existing_excel(DEMO_EXCEL_CANDIDATES)
    print(f"Using demographic Excel: {excel_path}")
    demo_df = load_demographic_table(excel_path)
    print(f"Loaded demographic table: n={len(demo_df)}, MDD={(demo_df['label'] == 0).sum()}, HC={(demo_df['label'] == 1).sum()}")

    train_dataset = TripleNpyDemographicDataset(
        root_dir1=TRAIN_ROOTS[0],
        root_dir2=TRAIN_ROOTS[1],
        root_dir3=TRAIN_ROOTS[2],
        demo_df=demo_df,
        demo_stats=None,
    )
    val_dataset = TripleNpyDemographicDataset(
        root_dir1=VAL_ROOTS[0],
        root_dir2=VAL_ROOTS[1],
        root_dir3=VAL_ROOTS[2],
        demo_df=demo_df,
        demo_stats=train_dataset.demo_stats,  # validation uses train statistics; no leakage
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = TripleInputConvNeXt(
        in_chans=1,
        num_classes=NUM_CLASSES,
        depths=(1, 1, 3, 1),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        demo_dim=3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = -1.0
    no_improvement_count = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_targets, train_preds = [], []

        for batch in train_loader:
            images1, images2, images3, labels, demo_info = unpack_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            logits, probabilities, _ = model(images1, images2, images3, demo_info)

            cls_loss = criterion(logits, labels)
            geo_reg_loss = model.ebv_classifier.get_geometry_regularization()
            loss = cls_loss + GEO_REG_WEIGHT * geo_reg_loss

            # No gradient accumulation: update once per mini-batch.
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_targets.extend(labels.detach().cpu().numpy())
            train_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())

        train_loss = train_loss_sum / len(train_dataset)
        train_metrics = binary_metrics(train_targets, train_preds, pos_label=POS_LABEL)

        model.eval()
        val_loss_sum = 0.0
        val_targets, val_preds, val_probs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                images1, images2, images3, labels, demo_info = unpack_batch(batch, device)
                logits, probabilities, _ = model(images1, images2, images3, demo_info)

                cls_loss = criterion(logits, labels)
                geo_reg_loss = model.ebv_classifier.get_geometry_regularization()
                loss = cls_loss + GEO_REG_WEIGHT * geo_reg_loss

                val_loss_sum += loss.item() * labels.size(0)
                val_targets.extend(labels.detach().cpu().numpy())
                val_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
                val_probs.extend(probabilities[:, POS_LABEL].detach().cpu().numpy())

        val_loss = val_loss_sum / len(val_dataset)
        val_metrics = binary_metrics(val_targets, val_preds, val_probs, POS_LABEL)

        print(f"\nEpoch {epoch + 1:03d}/{NUM_EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6g}")
        print(f"Train Loss: {train_loss:.6f}, Train Acc: {100 * train_metrics.get('acc', 0):.2f}%")
        print("Train Confusion Matrix [rows: MDD, HC; cols: MDD, HC]:")
        print(train_metrics["cm"])

        print(
            f"Val Loss: {val_loss:.6f}, Val Acc: {100 * val_metrics.get('acc', 0):.2f}%, "
            f"Precision: {100 * val_metrics.get('precision', 0):.2f}%, "
            f"Recall/SEN: {100 * val_metrics.get('recall', 0):.2f}%, "
            f"Specificity: {100 * val_metrics.get('specificity', 0):.2f}%, "
            f"F1: {100 * val_metrics.get('f1', 0):.2f}%, "
            f"AUC: {100 * val_metrics.get('auc', np.nan):.2f}%, "
            f"MCC: {val_metrics.get('mcc', 0):.4f}"
        )
        print("Validation Confusion Matrix [rows: MDD, HC; cols: MDD, HC]:")
        print(val_metrics["cm"])

        val_acc = val_metrics.get("acc", 0.0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improvement_count = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "pos_label": POS_LABEL,
                    "demo_columns": DEMO_COLUMNS,
                    "demo_stats": train_dataset.demo_stats,
                },
                SAVE_PATH,
            )
            print(f"Saved best model to: {SAVE_PATH}  Best Val Acc: {100 * best_val_acc:.2f}%")
        else:
            no_improvement_count += 1

        scheduler.step()

        if no_improvement_count >= NUM_EPOCHS:
            print(f"Early stopping at epoch {epoch + 1}")
            break


if __name__ == "__main__":
    main()
