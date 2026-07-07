"""Dataset utilities for shallow 2D acoustic inversion.

The dataset returns:
  input:  [C, T, R] filtered-waveform or processed-envelope B-scan channels
  target: [1, H, W] normalized Vp model
"""
from __future__ import annotations

import json
import random
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CHANNEL_ALIASES = {
    "processed_envelope": ("processed_envelope_npy", "seismic_source", "input_npy"),
    "filtered_waveform": ("filtered_waveform_npy",),
    "raw_forward": ("raw_forward_npy",),
}

MARINE_FACIES_CLASS_IDS = tuple(range(27))
DEFAULT_CLASS_IGNORE_INDEX = -1


@dataclass
class SampleRecord:
    name: str
    vp_model: Path
    channels: dict[str, Path]
    forward_window: Path | None = None
    class_label: Path | None = None
    mask: Path | None = None
    meta_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    group_id: str | None = None
    inferred_group: bool = False


def _as_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(value: Any, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path


def _find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_key(value, key)
            if found not in (None, ""):
                return found
    return None


def _infer_scene_id_from_name(name: str) -> str | None:
    match = re.match(r"(.+?)_line_\d+", name)
    if match:
        return match.group(1)
    if name.startswith("scene_"):
        return name
    return None


def _vp_meta_path(vp_path: Path) -> Path:
    if vp_path.name.endswith("_vp.npy"):
        return vp_path.with_name(vp_path.name.replace("_vp.npy", "_meta.json"))
    return vp_path.with_suffix(".json")


def _load_related_metadata(meta: dict[str, Any], meta_path: Path | None, vp_path: Path | None) -> dict[str, Any]:
    base = meta_path.parent if meta_path else Path.cwd()
    forward_meta_path = _resolve_path(meta.get("forward_metadata"), base)
    forward_meta = _read_json(forward_meta_path)
    source_vp_path = _resolve_path(meta.get("vp_source"), base)
    if source_vp_path is None:
        source_vp_path = _resolve_path(_find_key(forward_meta, "source_velocity_file"), base)
    if source_vp_path is None:
        source_vp_path = vp_path
    source_meta = _read_json(_vp_meta_path(source_vp_path)) if source_vp_path is not None else {}
    return {"sample": meta, "forward": forward_meta, "source": source_meta}


def _channel_path_from_metadata(channel: str, meta_bundle: dict[str, Any], base_dir: Path) -> Path | None:
    for section in ("sample", "forward"):
        meta = meta_bundle.get(section, {})
        for key in CHANNEL_ALIASES.get(channel, (channel,)):
            path = _resolve_path(meta.get(key), base_dir)
            if path is not None and path.exists():
                return path
    return None


def _make_record_from_training_meta(meta_path: Path) -> SampleRecord | None:
    meta = _read_json(meta_path)
    prefix = meta.get("prefix") or meta_path.name.replace("_meta.json", "")
    base_dir = meta_path.parent
    vp_path = base_dir / f"{prefix}_vp.npy"
    seis_path = base_dir / f"{prefix}_seis.npy"
    if not vp_path.exists() or not seis_path.exists():
        return None

    meta_bundle = _load_related_metadata(meta, meta_path, vp_path)
    primary_channel = str(meta.get(
        "seismic_stage", "processed_envelope"))
    if primary_channel not in CHANNEL_ALIASES:
        primary_channel = "processed_envelope"
    channels = {primary_channel: seis_path}
    for channel in ("filtered_waveform", "raw_forward"):
        if channel in channels:
            continue
        path = _channel_path_from_metadata(channel, meta_bundle, base_dir)
        if path is not None:
            channels[channel] = path
    return SampleRecord(
        name=prefix,
        vp_model=vp_path,
        channels=channels,
        meta_path=meta_path,
        metadata=meta_bundle,
    )


def _make_record_from_window_meta(meta_path: Path) -> SampleRecord | None:
    meta = _read_json(meta_path)
    schema_version = int(meta.get("training_window_schema_version") or 0)
    if meta.get("complete") is not True or schema_version < 3:
        return None
    base_dir = meta_path.parent
    if meta_path.name == "meta.json" and not (base_dir / "_SUCCESS").is_file():
        return None
    forward_path = _resolve_path(meta.get("forward_data_file"), base_dir)
    vp_path = _resolve_path(meta.get("vp_file"), base_dir)
    class_path = _resolve_path(meta.get("class_label_file"), base_dir)
    mask_path = _resolve_path(meta.get("mask_file"), base_dir)
    required = (forward_path, vp_path, class_path)
    if any(path is None or not path.exists() for path in required):
        return None
    if mask_path is not None and not mask_path.exists():
        mask_path = None
    channel_names = list(meta.get("input_channel_names") or ())
    if not channel_names:
        return None
    return SampleRecord(
        name=str(meta.get("sample_id") or meta_path.name[:-len("_meta.json")]),
        vp_model=vp_path,
        channels={},
        forward_window=forward_path,
        class_label=class_path,
        mask=mask_path,
        meta_path=meta_path,
        metadata={"sample": meta, "forward": {}, "source": {}},
    )


def _make_record_from_forward_meta(meta_path: Path) -> SampleRecord | None:
    meta = _read_json(meta_path)
    base_dir = meta_path.parent
    processed = _resolve_path(meta.get("processed_envelope_npy") or meta.get("input_npy"), base_dir)
    forward_meta = meta.get("forward_metadata", {})
    vp_path = _resolve_path(meta.get("vp_model") or meta.get("vp_source"), base_dir)
    if vp_path is None:
        vp_path = _resolve_path(forward_meta.get("source_velocity_file"), base_dir)
    if processed is None or vp_path is None or not processed.exists() or not vp_path.exists():
        return None

    name = forward_meta.get("sample_identity", meta_path.name.replace("_meta.json", ""))
    if isinstance(name, str) and name.endswith("_vp.npy"):
        name = name.replace("_vp.npy", "")
    meta_bundle = {"sample": meta, "forward": meta, "source": _read_json(_vp_meta_path(vp_path))}
    channels = {"processed_envelope": processed}
    for channel in ("filtered_waveform", "raw_forward"):
        path = _channel_path_from_metadata(channel, meta_bundle, base_dir)
        if path is not None:
            channels[channel] = path
    return SampleRecord(
        name=str(name),
        vp_model=vp_path,
        channels=channels,
        meta_path=meta_path,
        metadata=meta_bundle,
    )


def discover_samples(root: str | Path,
                     formal_only: bool = True) -> list[SampleRecord]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    records: list[SampleRecord] = []
    seen: set[tuple[str, str]] = set()

    metadata_paths = (
        sorted(root.rglob("meta.json"))
        if formal_only else sorted({
            *root.rglob("*_meta.json"),
            *root.rglob("meta.json"),
        }))
    for meta_path in metadata_paths:
        if formal_only:
            record = _make_record_from_window_meta(meta_path)
        else:
            record = (_make_record_from_window_meta(meta_path)
                      or _make_record_from_training_meta(meta_path)
                      or _make_record_from_forward_meta(meta_path))
        if record is None:
            continue
        primary_path = record.forward_window or next(iter(record.channels.values()))
        key = (str(record.vp_model.resolve()), str(primary_path.resolve()))
        if key not in seen:
            records.append(record)
            seen.add(key)

    if not formal_only:
        for vp_path in sorted(root.rglob("*_vp.npy")):
            prefix = vp_path.name.replace("_vp.npy", "")
            seis_path = vp_path.with_name(f"{prefix}_seis.npy")
            if not seis_path.exists():
                continue
            key = (str(vp_path.resolve()), str(seis_path.resolve()))
            if key in seen:
                continue
            meta_path = vp_path.with_name(f"{prefix}_meta.json")
            meta = _read_json(meta_path)
            records.append(
                SampleRecord(
                    name=prefix,
                    vp_model=vp_path,
                    channels={"processed_envelope": seis_path},
                    meta_path=meta_path if meta_path.exists() else None,
                    metadata=_load_related_metadata(meta, meta_path if meta_path.exists() else None, vp_path),
                )
            )
            seen.add(key)

    if not records:
        if formal_only:
            raise FileNotFoundError(
                f"No formal inversion samples found under {root}. Expected "
                "samples/<split>/<sample>/meta.json with _SUCCESS plus "
                "forward.npy/vp.npy/class_label.npy.")
        raise FileNotFoundError(
            f"No inversion samples found under {root}. Expected paired *_seis.npy/*_vp.npy "
            "or forward_processed *_meta.json files."
        )
    return records


def _is_formal_window_record(record: SampleRecord) -> bool:
    meta = record.metadata.get("sample", {})
    return (
        record.forward_window is not None
        and int(meta.get("training_window_schema_version") or 0) >= 3
        and meta.get("output_mode") == "time_input_depth_label_fullwaveform"
    )


def _quality_metric(record: SampleRecord, key: str) -> float | None:
    return _as_optional_float(_find_key(record.metadata, key))


def filter_records_for_quality(
        records: list[SampleRecord],
        data_cfg: dict[str, Any],
        split_name: str | None = None) -> list[SampleRecord]:
    cfg = dict(data_cfg.get("quality_filters") or {})
    if not bool(cfg.get("enabled", False)):
        return records

    formal_only = bool(cfg.get("formal_only", True))
    reject_missing = bool(cfg.get("reject_missing_metrics", False))
    thresholds = (
        ("time_window_padding_fraction", "max",
         _as_optional_float(cfg.get("max_time_window_padding_fraction"))),
        ("ignore_label_fraction", "max",
         _as_optional_float(cfg.get("max_ignore_label_fraction"))),
        ("valid_semantic_label_fraction", "min",
         _as_optional_float(cfg.get("min_valid_semantic_label_fraction"))),
    )
    thresholds = tuple(item for item in thresholds if item[2] is not None)
    if not thresholds:
        return records

    kept: list[SampleRecord] = []
    rejected: list[tuple[str, str]] = []
    for record in records:
        if formal_only and not _is_formal_window_record(record):
            kept.append(record)
            continue
        problems = []
        for key, mode, threshold in thresholds:
            value = _quality_metric(record, key)
            if value is None:
                if reject_missing:
                    problems.append(f"{key}=missing")
                continue
            if mode == "max" and value > threshold:
                problems.append(f"{key}={value:.6g}>{threshold:.6g}")
            elif mode == "min" and value < threshold:
                problems.append(f"{key}={value:.6g}<{threshold:.6g}")
        if problems:
            rejected.append((record.name, ", ".join(problems)))
        else:
            kept.append(record)

    if rejected:
        label = f" for {split_name}" if split_name else ""
        preview = "; ".join(
            f"{name}: {reason}" for name, reason in rejected[:5])
        suffix = "" if len(rejected) <= 5 else f"; ... +{len(rejected) - 5}"
        warnings.warn(
            f"Filtered {len(rejected)}/{len(records)} samples{label} by "
            f"data.quality_filters: {preview}{suffix}",
            RuntimeWarning,
        )
    return kept


def assign_group_ids(records: list[SampleRecord], split_cfg: dict[str, Any]) -> None:
    group_keys = split_cfg.get("group_keys", ["model_id", "scene_id", "seed"])
    infer_from_name = bool(split_cfg.get("infer_scene_id_from_name", True))
    warned_missing_scene = False

    for record in records:
        explicit_scene = _find_key(record.metadata, "model_id") or _find_key(record.metadata, "scene_id")
        if explicit_scene in (None, "") and not warned_missing_scene:
            warnings.warn(
                "Sample metadata has no model_id/scene_id. Falling back to inferred scene keys where possible; "
                "add true 3D model IDs to meta.json before formal experiments to avoid leakage.",
                RuntimeWarning,
            )
            warned_missing_scene = True

        group_value = explicit_scene
        inferred = False
        if group_value in (None, "") and infer_from_name:
            group_value = _infer_scene_id_from_name(record.name)
            inferred = group_value is not None
        if group_value in (None, ""):
            for key in group_keys:
                if key in ("model_id", "scene_id"):
                    continue
                group_value = _find_key(record.metadata, key)
                if group_value not in (None, ""):
                    break
        if group_value in (None, ""):
            if split_cfg.get("allow_sample_level_split_when_group_missing", False):
                group_value = f"sample:{record.name}"
                inferred = True
            else:
                group_value = "unknown_scene"
                inferred = True

        record.group_id = str(group_value)
        record.inferred_group = inferred or explicit_scene in (None, "")


def split_records(records: list[SampleRecord], split_cfg: dict[str, Any]) -> dict[str, list[SampleRecord]]:
    assign_group_ids(records, split_cfg)
    rng = random.Random(int(split_cfg.get("seed", 42)))
    groups: dict[str, list[SampleRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id or "unknown_scene", []).append(record)

    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    n_groups = len(group_ids)
    if n_groups == 1:
        warnings.warn(
            "Only one scene/model group is available. Validation and test splits will be empty to avoid leakage.",
            RuntimeWarning,
        )
        split_group_ids = {"train": group_ids, "val": [], "test": []}
    elif n_groups == 2:
        split_group_ids = {"train": group_ids[:1], "val": group_ids[1:], "test": []}
    else:
        val_ratio = float(split_cfg.get("val_ratio", 0.1))
        test_ratio = float(split_cfg.get("test_ratio", 0.1))
        n_test = max(1, round(n_groups * test_ratio))
        n_val = max(1, round(n_groups * val_ratio))
        if n_val + n_test >= n_groups:
            n_val = 1
            n_test = 1
        n_train = n_groups - n_val - n_test
        split_group_ids = {
            "train": group_ids[:n_train],
            "val": group_ids[n_train : n_train + n_val],
            "test": group_ids[n_train + n_val :],
        }

    return {
        split: [record for group_id in ids for record in groups[group_id]]
        for split, ids in split_group_ids.items()
    }


class Shallow2DInversionDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        input_channels: list[str],
        target_time_samples: int,
        target_traces: int,
        target_depth: int,
        target_width: int,
        vp_min: float,
        vp_max: float,
        input_normalization: str = "per_sample_percentile",
        input_percentile: float = 99.5,
        input_clip: float | None = 6.0,
        clip_normalized_vp: bool = True,
        missing_channel_policy: str = "zeros",
        augment_flip: bool = False,
        class_label_cfg: dict[str, Any] | None = None,
        vp_loss_mask_cfg: dict[str, Any] | None = None,
    ) -> None:
        if not records:
            warnings.warn("Created an empty dataset split.", RuntimeWarning)
        self.records = records
        self.input_channels = input_channels
        self.target_time_samples = int(target_time_samples)
        self.target_traces = int(target_traces)
        self.target_depth = int(target_depth)
        self.target_width = int(target_width)
        self.vp_min = float(vp_min)
        self.vp_max = float(vp_max)
        self.input_normalization = input_normalization
        self.input_percentile = float(input_percentile)
        self.input_clip = input_clip
        self.clip_normalized_vp = bool(clip_normalized_vp)
        self.missing_channel_policy = missing_channel_policy
        self.augment_flip = augment_flip
        self.class_label_cfg = dict(class_label_cfg or {})
        self.vp_loss_mask_cfg = dict(vp_loss_mask_cfg or {})
        self.vp_loss_mask_enabled = bool(
            self.vp_loss_mask_cfg.get("enabled", False))
        self.vp_loss_mask_source = str(
            self.vp_loss_mask_cfg.get("source", "below_seafloor"))
        self.vp_loss_mask_min_valid_fraction = float(
            self.vp_loss_mask_cfg.get("min_valid_fraction", 0.01))
        self.class_ignore_index = int(
            self.class_label_cfg.get(
                "ignore_index", DEFAULT_CLASS_IGNORE_INDEX))
        self.class_unknown_policy = str(
            self.class_label_cfg.get("unknown_policy", "ignore"))
        self.class_remap_enabled = bool(
            self.class_label_cfg.get("remap_to_contiguous", True))
        self.class_raw_ids = self._resolve_class_raw_ids(
            self.class_label_cfg.get("raw_ids", "marine_facies_v1"))
        self.raw_to_train_class = {
            int(raw_id): int(train_id)
            for train_id, raw_id in enumerate(self.class_raw_ids)
        }
        self.train_to_raw_class = {
            int(train_id): int(raw_id)
            for raw_id, train_id in self.raw_to_train_class.items()
        }
        self.num_class_labels = len(self.class_raw_ids)
        if self.vp_max <= self.vp_min:
            raise ValueError("vp_max must be greater than vp_min")
        if self.class_unknown_policy not in ("ignore", "error", "keep_raw"):
            raise ValueError(
                "class_label.unknown_policy must be ignore, error or keep_raw")
        if not self.class_raw_ids:
            raise ValueError("class_label.raw_ids must not be empty")
        if self.vp_loss_mask_source not in (
                "below_seafloor", "class_label_valid",
                "class_label_valid_or_below_seafloor"):
            raise ValueError(
                "vp_loss_mask.source must be below_seafloor, "
                "class_label_valid or class_label_valid_or_below_seafloor")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record.forward_window is not None:
            channel_tensors = self._load_window_channels(record)
        else:
            channel_tensors = []
            for channel in self.input_channels:
                path = record.channels.get(channel)
                if path is None or not path.exists():
                    if channel == "processed_envelope" or self.missing_channel_policy == "error":
                        raise FileNotFoundError(f"Missing channel '{channel}' for sample {record.name}")
                    if self.missing_channel_policy == "skip":
                        continue
                    tensor = torch.zeros(self.target_time_samples, self.target_traces, dtype=torch.float32)
                else:
                    tensor = self._load_input_channel(path, record.metadata)
                channel_tensors.append(tensor)

        if not channel_tensors:
            raise RuntimeError(f"No input channels loaded for sample {record.name}")
        inputs = torch.stack(channel_tensors, dim=0)
        target = self._load_vp(record.vp_model)
        class_label = self._load_class_label(record.class_label)
        masks = self._load_masks(record.mask)
        vp_loss_mask = self._load_vp_loss_mask(record, class_label)

        if self.augment_flip and random.random() < 0.5:
            inputs = torch.flip(inputs, dims=[2])
            target = torch.flip(target, dims=[2])
            if class_label is not None:
                class_label = torch.flip(class_label, dims=[2])
            if masks is not None:
                masks = torch.flip(masks, dims=[2])
            if vp_loss_mask is not None:
                vp_loss_mask = torch.flip(vp_loss_mask, dims=[2])

        sample = {
            "input": inputs,
            "target": target,
            "name": record.name,
            "group_id": record.group_id or "",
            "vp_path": str(record.vp_model),
            "meta_path": str(record.meta_path) if record.meta_path else "",
        }
        if class_label is not None:
            sample["class_label"] = class_label
            sample["class_label_ignore_index"] = self.class_ignore_index
        if masks is not None:
            sample["mask"] = masks
        if vp_loss_mask is not None:
            sample["vp_loss_mask"] = vp_loss_mask
        return sample

    @staticmethod
    def _resolve_class_raw_ids(value: Any) -> list[int]:
        if value in (None, "", "marine_facies_v1", "all"):
            return list(MARINE_FACIES_CLASS_IDS)
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            if not parts:
                return list(MARINE_FACIES_CLASS_IDS)
            return [int(part) for part in parts]
        return [int(item) for item in value]

    def _load_window_channels(self, record: SampleRecord) -> list[torch.Tensor]:
        arr = np.load(record.forward_window, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 3:
            raise ValueError(
                f"Expected [C,T,R] forward window, got {arr.shape}")
        names = list(_find_key(record.metadata, "input_channel_names") or ())
        if len(names) != arr.shape[0]:
            raise ValueError(
                f"Channel metadata does not match {record.forward_window}")
        tensors = []
        for channel in self.input_channels:
            if channel not in names:
                if self.missing_channel_policy == "error":
                    raise FileNotFoundError(
                        f"Missing channel '{channel}' for sample {record.name}")
                if self.missing_channel_policy == "skip":
                    continue
                tensor = torch.zeros(
                    self.target_time_samples, self.target_traces,
                    dtype=torch.float32)
            else:
                channel_array = np.asarray(
                    arr[names.index(channel)], dtype=np.float32)
                channel_array = np.nan_to_num(
                    channel_array, nan=0.0, posinf=0.0, neginf=0.0)
                channel_array = self._normalize_input(channel_array)
                tensor = torch.from_numpy(
                    np.ascontiguousarray(channel_array)).float()
                tensor = _resize_2d(
                    tensor, (self.target_time_samples, self.target_traces))
            tensors.append(tensor)
        return tensors

    def _load_input_channel(self, path: Path, metadata: dict[str, Any]) -> torch.Tensor:
        arr = np.load(path)
        arr = self._ensure_time_trace(arr, metadata)
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = self._normalize_input(arr)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
        return _resize_2d(tensor, (self.target_time_samples, self.target_traces))

    def _load_vp(self, path: Path) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected [1,H,W] or [H,W] Vp, got {arr.shape}")
        arr = np.nan_to_num(arr, nan=self.vp_min, posinf=self.vp_max, neginf=self.vp_min)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
        tensor = _resize_2d(tensor, (self.target_depth, self.target_width))
        tensor = (tensor - self.vp_min) / (self.vp_max - self.vp_min)
        if self.clip_normalized_vp:
            tensor = tensor.clamp(0.0, 1.0)
        return tensor.unsqueeze(0)

    def _load_class_label(self, path: Path | None) -> torch.Tensor | None:
        if path is None:
            return None
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected class label [1,H,W], got {arr.shape}")
        arr = arr.astype(np.int64, copy=False)
        if self.class_remap_enabled:
            arr = self._remap_class_label(arr)
        tensor = torch.from_numpy(np.ascontiguousarray(arr.astype(np.int64)))
        return _resize_nearest_2d(
            tensor, (self.target_depth, self.target_width)).unsqueeze(0)

    def _remap_class_label(self, arr: np.ndarray) -> np.ndarray:
        remapped = np.full(arr.shape, self.class_ignore_index, dtype=np.int64)
        ignore_mask = arr == self.class_ignore_index
        for raw_id, train_id in self.raw_to_train_class.items():
            remapped[arr == raw_id] = train_id
        unknown = ~(ignore_mask | np.isin(arr, self.class_raw_ids))
        if np.any(unknown):
            unknown_ids = sorted(set(arr[unknown].astype(int).tolist()))
            if self.class_unknown_policy == "error":
                raise ValueError(
                    f"Class label contains unknown raw facies ids: {unknown_ids}")
            if self.class_unknown_policy == "keep_raw":
                remapped[unknown] = arr[unknown]
        return remapped

    def _load_masks(self, path: Path | None) -> torch.Tensor | None:
        if path is None:
            return None
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 3:
            raise ValueError(f"Expected mask [M,H,W], got {arr.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if tuple(tensor.shape[-2:]) != (self.target_depth, self.target_width):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.target_depth, self.target_width),
                mode="nearest").squeeze(0)
        return tensor

    def _load_vp_loss_mask(
            self, record: SampleRecord,
            class_label: torch.Tensor | None) -> torch.Tensor | None:
        if not self.vp_loss_mask_enabled:
            return None
        masks = []
        if self.vp_loss_mask_source in (
                "class_label_valid", "class_label_valid_or_below_seafloor"):
            if class_label is not None:
                masks.append((class_label[0] != self.class_ignore_index))
            elif self.vp_loss_mask_source == "class_label_valid":
                return None
        if self.vp_loss_mask_source in (
                "below_seafloor", "class_label_valid_or_below_seafloor"):
            below = self._below_seafloor_mask(record)
            if below is not None:
                masks.append(below)
        if not masks:
            return None
        mask = masks[0]
        for next_mask in masks[1:]:
            if self.vp_loss_mask_source == "class_label_valid_or_below_seafloor":
                mask = mask | next_mask
            else:
                mask = mask & next_mask
        mask = mask.to(torch.float32)
        valid_fraction = float(mask.mean().item()) if mask.numel() else 0.0
        if valid_fraction < self.vp_loss_mask_min_valid_fraction:
            warnings.warn(
                f"Vp loss mask for {record.name} has low valid fraction "
                f"{valid_fraction:.4f}; falling back to unmasked Vp loss.",
                RuntimeWarning,
            )
            return None
        return mask.unsqueeze(0)

    def _below_seafloor_mask(self, record: SampleRecord) -> torch.Tensor | None:
        meta = record.metadata.get("sample", {})
        above_m = _as_optional_float(
            meta.get("label_depth_above_m")
            or meta.get("depth_window_above_seafloor_m"))
        dz_m = _as_optional_float(meta.get("label_dz_m")
                                  or meta.get("depth_window_dz_m"))
        source_h = meta.get("label_H") or meta.get("depth_window_samples")
        try:
            source_h = int(source_h)
        except (TypeError, ValueError):
            source_h = None
        if above_m is None or dz_m is None or source_h is None or source_h <= 0:
            return None
        relative_depth_m = (
            np.arange(source_h, dtype=np.float32) * float(dz_m)
            - float(above_m))
        mask = torch.from_numpy(relative_depth_m >= 0.0).to(torch.float32)
        mask = mask[:, None].expand(source_h, self.target_width)
        return _resize_nearest_2d(
            mask.to(torch.int64),
            (self.target_depth, self.target_width)).to(torch.bool)

    def _normalize_input(self, arr: np.ndarray) -> np.ndarray:
        if self.input_normalization == "none":
            out = arr
        elif self.input_normalization == "per_sample_percentile":
            scale = np.percentile(np.abs(arr), self.input_percentile)
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            out = arr / scale
        elif self.input_normalization == "standard":
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            out = (arr - mean) / std
        else:
            raise ValueError(f"Unsupported input_normalization: {self.input_normalization}")
        if self.input_clip is not None:
            out = np.clip(out, -float(self.input_clip), float(self.input_clip))
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _ensure_time_trace(arr: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D B-scan array, got shape {arr.shape}")

        sample_shape = _find_key(metadata, "seis_time_trace")
        if isinstance(sample_shape, (list, tuple)) and len(sample_shape) == 2:
            shape = tuple(int(v) for v in sample_shape)
            if tuple(arr.shape) == shape:
                return arr
            if tuple(arr.shape) == shape[::-1]:
                return arr.T

        target_nt = _find_key(metadata, "target_nt")
        shot_count = _find_key(metadata, "shot_count")
        if target_nt and shot_count:
            expected_rt = (int(shot_count), int(target_nt))
            expected_tr = (int(target_nt), int(shot_count))
            if tuple(arr.shape) == expected_rt:
                return arr.T
            if tuple(arr.shape) == expected_tr:
                return arr

        return arr.T if arr.shape[0] < arr.shape[1] else arr


def _resize_2d(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    tensor4d = tensor.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor4d, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0)


def _resize_nearest_2d(tensor: torch.Tensor,
                       size: tuple[int, int]) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    resized = F.interpolate(
        tensor.to(torch.float32).unsqueeze(0).unsqueeze(0),
        size=size, mode="nearest")
    return resized.squeeze(0).squeeze(0).to(tensor.dtype)


def make_dataset(records: list[SampleRecord], data_cfg: dict[str, Any], augment_flip: bool = False) -> Shallow2DInversionDataset:
    return Shallow2DInversionDataset(
        records=records,
        input_channels=list(data_cfg.get("input_channels", ["processed_envelope"])),
        target_time_samples=int(data_cfg["target_time_samples"]),
        target_traces=int(data_cfg["target_traces"]),
        target_depth=int(data_cfg["target_depth"]),
        target_width=int(data_cfg.get("target_width", data_cfg["target_traces"])),
        vp_min=float(data_cfg["vp_min"]),
        vp_max=float(data_cfg["vp_max"]),
        input_normalization=data_cfg.get("input_normalization", "per_sample_percentile"),
        input_percentile=float(data_cfg.get("input_percentile", 99.5)),
        input_clip=data_cfg.get("input_clip", 6.0),
        clip_normalized_vp=bool(data_cfg.get("clip_normalized_vp", True)),
        missing_channel_policy=data_cfg.get("missing_channel_policy", "zeros"),
        augment_flip=augment_flip,
        class_label_cfg=data_cfg.get("class_label", {}),
        vp_loss_mask_cfg=data_cfg.get("vp_loss_mask", {}),
    )


def _limit_records_for_split(
    records: list[SampleRecord],
    data_cfg: dict[str, Any],
    split: str,
) -> list[SampleRecord]:
    limits = data_cfg.get("max_samples_per_split")
    if limits in (None, "", False):
        return records
    if isinstance(limits, int):
        limit = limits
    elif isinstance(limits, dict):
        value = limits.get(split)
        if value in (None, "", False):
            return records
        limit = int(value)
    else:
        raise ValueError("data.max_samples_per_split must be an int or split mapping")
    if limit <= 0:
        return []
    return records[:limit]


def build_datasets(config: dict[str, Any]) -> dict[str, Shallow2DInversionDataset]:
    data_cfg = config["data"]
    root = Path(data_cfg["root_dir"])
    formal_only = bool(data_cfg.get("formal_only", True))
    explicit_dirs = {
        "train": data_cfg.get("train_dir"),
        "val": data_cfg.get("val_dir"),
        "test": data_cfg.get("test_dir"),
    }
    formal_split_root = root / "samples"
    if not any(explicit_dirs.values()) and formal_split_root.is_dir():
        explicit_dirs = {
            split: str(formal_split_root / split)
            if (formal_split_root / split).is_dir() else None
            for split in ("train", "val", "test")
        }
    elif (not any(explicit_dirs.values()) and not formal_only
          and all((root / d).is_dir() for d in ("train", "val", "test"))):
        explicit_dirs = {split: str(root / split) for split in ("train", "val", "test")}

    if any(explicit_dirs.values()):
        split_records_map = {}
        for split, split_dir in explicit_dirs.items():
            if split_dir:
                records = discover_samples(split_dir, formal_only=formal_only)
                records = filter_records_for_quality(
                    records, data_cfg, split_name=split)
                assign_group_ids(records, data_cfg.get("split", {}))
                records = _limit_records_for_split(records, data_cfg, split)
            else:
                records = []
            split_records_map[split] = records
    else:
        records = discover_samples(root, formal_only=formal_only)
        records = filter_records_for_quality(records, data_cfg)
        split_records_map = split_records(records, data_cfg.get("split", {}))
        split_records_map = {
            split: _limit_records_for_split(records, data_cfg, split)
            for split, records in split_records_map.items()
        }

    return {
        "train": make_dataset(split_records_map.get("train", []), data_cfg, augment_flip=True),
        "val": make_dataset(split_records_map.get("val", []), data_cfg, augment_flip=False),
        "test": make_dataset(split_records_map.get("test", []), data_cfg, augment_flip=False),
    }


def denormalize_vp(vp_norm: torch.Tensor, vp_min: float, vp_max: float) -> torch.Tensor:
    return vp_norm * (vp_max - vp_min) + vp_min


if __name__ == "__main__":
    import yaml

    cfg_path = Path(__file__).parent / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datasets = build_datasets(cfg)
    for split, dataset in datasets.items():
        print(split, len(dataset), [record.group_id for record in dataset.records])
        if len(dataset):
            sample = dataset[0]
            print(sample["name"], tuple(sample["input"].shape), tuple(sample["target"].shape))
