"""纯 NumPy/CuPy 图像分类与开放集识别核心实现。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
xp = np
DEVICE = "cpu"
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def configure_device(requested: str = "auto") -> str:
    """选择计算后端。GPU 模式只用 CuPy，不依赖深度学习框架。"""
    global xp, DEVICE
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("device 只能是 auto、cpu 或 gpu")
    if requested != "cpu":
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("未检测到可用 CUDA 设备")
            cp.ones(1, dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
            xp, DEVICE = cp, "gpu"
            return DEVICE
        except Exception as exc:
            if requested == "gpu":
                raise RuntimeError("无法启用 CuPy GPU；请检查 NVIDIA 驱动与 cupy-cuda11x 安装") from exc
    xp, DEVICE = np, "cpu"
    return DEVICE


def to_cpu(value):
    return xp.asnumpy(value) if DEVICE == "gpu" else np.asarray(value)


def array_module():
    return xp


def seed_everything(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def image_label(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return stem.rsplit("_", 1)[0]


def directory_label(name: str) -> str:
    prefix, dot, rest = name.partition(".")
    return rest if dot and prefix.isdigit() else name


def canonical_label(name: str) -> str:
    return name.lower()


def display_label(name: str) -> str:
    return name.replace("+", " ").replace("_", " ").title()


def collect_dataset(root: Path) -> tuple[list[Path], list[str], list[Path], list[str]]:
    train_dir, valid_dir = root / "train", root / "valid"
    if not train_dir.is_dir() or not valid_dir.is_dir():
        raise FileNotFoundError(f"未找到 train/valid 目录：{root}")
    train_paths, train_labels = [], []
    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                train_paths.append(path)
                train_labels.append(canonical_label(directory_label(class_dir.name)))
    valid_paths = sorted(p for p in valid_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    valid_labels = [canonical_label(image_label(p)) for p in valid_paths]
    if not train_paths or not valid_paths:
        raise RuntimeError("未读取到训练图片或验证图片")
    return train_paths, train_labels, valid_paths, valid_labels


def stratified_split(
    paths: list[Path], labels: list[str], valid_ratio: float, rng: np.random.Generator
) -> tuple[list[Path], list[str], list[Path], list[str]]:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio 必须在 0 和 1 之间")
    by_label: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label, []).append(i)
    train_idx, valid_idx = [], []
    for indices in by_label.values():
        ordered = rng.permutation(indices)
        n_valid = max(1, round(len(indices) * valid_ratio))
        n_valid = min(n_valid, len(indices) - 1) if len(indices) > 1 else 0
        valid_idx.extend(ordered[:n_valid])
        train_idx.extend(ordered[n_valid:])
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    return (
        [paths[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [paths[i] for i in valid_idx],
        [labels[i] for i in valid_idx],
    )


def _random_resized_crop(image: Image.Image, size: int, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    area = width * height
    for _ in range(10):
        target_area = area * rng.uniform(0.72, 1.0)
        aspect = rng.uniform(0.80, 1.25)
        crop_width = int(round(np.sqrt(target_area * aspect)))
        crop_height = int(round(np.sqrt(target_area / aspect)))
        if crop_width <= width and crop_height <= height:
            left = int(rng.integers(0, width - crop_width + 1))
            top = int(rng.integers(0, height - crop_height + 1))
            return image.crop((left, top, left + crop_width, top + crop_height)).resize(
                (size, size), Image.Resampling.BILINEAR
            )
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize((max(size, round(width * scale)), max(size, round(height * scale))), Image.Resampling.BILINEAR)
    width, height = resized.size
    left, top = (width - size) // 2, (height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def load_image(path: Path, size: int, augment: bool, rng: np.random.Generator) -> np.ndarray:
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image = _random_resized_crop(image, size, rng) if augment else _center_crop(image, size)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise RuntimeError(f"无法读取图像：{path}") from exc
    if augment:
        if rng.random() < 0.5:
            array = array[:, ::-1]
        if rng.random() < 0.15:
            gray = array.mean(axis=2, keepdims=True)
            array = gray * 0.25 + array * 0.75
        contrast = rng.uniform(0.82, 1.18)
        brightness = rng.uniform(0.86, 1.14)
        array = np.clip((array - 0.5) * contrast + 0.5, 0, 1) * brightness
        if rng.random() < 0.2:
            array = np.clip(array + rng.normal(0, 0.012, array.shape), 0, 1)
    array = (array - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(array, dtype=np.float32)


def batches(
    paths: list[Path],
    labels: np.ndarray,
    batch_size: int,
    size: int,
    augment: bool,
    rng: np.random.Generator,
    shuffle: bool = True,
    sample_weights: np.ndarray | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    labels = xp.asarray(labels, dtype=xp.int64)
    if shuffle and sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64)
        weights /= weights.sum()
        order = rng.choice(len(paths), size=len(paths), replace=True, p=weights)
    else:
        order = rng.permutation(len(paths)) if shuffle else np.arange(len(paths))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        x = xp.asarray(np.stack([load_image(paths[i], size, augment, rng) for i in idx]))
        yield x, labels[idx]


class Conv2D:
    def __init__(self, in_channels: int, out_channels: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / (in_channels * 9))
        self.w = xp.asarray(rng.normal(0, scale, (out_channels, in_channels, 3, 3)).astype(np.float32))
        self.b = xp.zeros(out_channels, dtype=xp.float32)
        self.dw, self.db = xp.zeros_like(self.w), xp.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        padded = xp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        self.padded = padded
        b, h, w, c = x.shape
        self.cols = xp.empty((b, h, w, c * 9), dtype=xp.float32)
        for iy in range(3):
            for ix in range(3):
                self.cols[..., (iy * 3 + ix) * c : (iy * 3 + ix + 1) * c] = padded[:, iy : iy + h, ix : ix + w, :]
        kernel = self.w.transpose(0, 2, 3, 1).reshape(self.w.shape[0], -1)
        return (self.cols.reshape(-1, c * 9) @ kernel.T + self.b).reshape(b, h, w, -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        b, h, w, out_c = grad.shape
        c = self.x_shape[-1]
        grad2 = grad.reshape(-1, out_c)
        col2 = self.cols.reshape(-1, c * 9)
        self.dw[...] = (grad2.T @ col2).reshape(out_c, 3, 3, c).transpose(0, 3, 1, 2)
        self.db[...] = grad2.sum(axis=0)
        kernel = self.w.transpose(0, 2, 3, 1).reshape(out_c, -1)
        dcol = (grad2 @ kernel).reshape(b, h, w, 3, 3, c)
        dpadded = xp.zeros_like(self.padded)
        for iy in range(3):
            for ix in range(3):
                dpadded[:, iy : iy + h, ix : ix + w, :] += dcol[:, :, :, iy, ix, :]
        return dpadded[:, 1:-1, 1:-1, :]


class BatchNorm2D:
    def __init__(self, channels: int):
        self.gamma = xp.ones(channels, dtype=xp.float32)
        self.beta = xp.zeros(channels, dtype=xp.float32)
        self.dgamma = xp.zeros_like(self.gamma)
        self.dbeta = xp.zeros_like(self.beta)
        self.running_mean = xp.zeros(channels, dtype=xp.float32)
        self.running_var = xp.ones(channels, dtype=xp.float32)
        self.epsilon = 1e-5
        self.momentum = 0.9

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        if training:
            self.mean = x.mean(axis=(0, 1, 2))
            variance = ((x - self.mean) ** 2).mean(axis=(0, 1, 2))
            self.inv_std = 1.0 / xp.sqrt(variance + self.epsilon)
            self.x_hat = (x - self.mean) * self.inv_std
            self.running_mean[...] = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var[...] = self.momentum * self.running_var + (1 - self.momentum) * variance
            return self.gamma * self.x_hat + self.beta
        x_hat = (x - self.running_mean) / xp.sqrt(self.running_var + self.epsilon)
        return self.gamma * x_hat + self.beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        sample_count = grad.shape[0] * grad.shape[1] * grad.shape[2]
        self.dgamma[...] = (grad * self.x_hat).sum(axis=(0, 1, 2))
        self.dbeta[...] = grad.sum(axis=(0, 1, 2))
        summed = grad.sum(axis=(0, 1, 2))
        projected = (grad * self.x_hat).sum(axis=(0, 1, 2))
        return self.gamma * self.inv_std / sample_count * (sample_count * grad - summed - self.x_hat * projected)


class MaxPool2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        b, h, w, c = x.shape
        h2, w2 = h // 2, w // 2
        self.crop_shape = (h2 * 2, w2 * 2)
        windows = x[:, : h2 * 2, : w2 * 2, :].reshape(b, h2, 2, w2, 2, c).transpose(0, 1, 3, 5, 2, 4)
        flat = windows.reshape(b, h2, w2, c, 4)
        self.argmax = flat.argmax(axis=-1)
        return flat.max(axis=-1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        b, h2, w2, c = grad.shape
        flat = xp.zeros((b, h2, w2, c, 4), dtype=grad.dtype)
        flat[xp.arange(b)[:, None, None, None], xp.arange(h2)[None, :, None, None], xp.arange(w2)[None, None, :, None], xp.arange(c)[None, None, None, :], self.argmax] = grad
        windows = flat.reshape(b, h2, w2, c, 2, 2).transpose(0, 1, 4, 2, 5, 3)
        dx = xp.zeros(self.input_shape, dtype=grad.dtype)
        h, w = self.crop_shape
        dx[:, :h, :w, :] = windows.reshape(b, h, w, c)
        return dx


class NumpyCNN:
    """带批归一化的四层卷积分类网络，特征和分类头均由 NumPy 手写。"""

    def __init__(self, classes: int, rng: np.random.Generator):
        self.rng = rng
        self.dropout_rate = 0.2
        self.conv1a, self.bn1a, self.conv1b, self.bn1b = Conv2D(3, 32, rng), BatchNorm2D(32), Conv2D(32, 32, rng), BatchNorm2D(32)
        self.conv2a, self.bn2a, self.conv2b, self.bn2b = Conv2D(32, 64, rng), BatchNorm2D(64), Conv2D(64, 64, rng), BatchNorm2D(64)
        self.conv3a, self.bn3a, self.conv3b, self.bn3b = Conv2D(64, 128, rng), BatchNorm2D(128), Conv2D(128, 128, rng), BatchNorm2D(128)
        self.conv4, self.bn4 = Conv2D(128, 192, rng), BatchNorm2D(192)
        self.pool1, self.pool2, self.pool3 = MaxPool2D(), MaxPool2D(), MaxPool2D()
        feature_width = 192 * 2
        self.fc1_w = xp.asarray(rng.normal(0, np.sqrt(2 / feature_width), (feature_width, 256)).astype(np.float32))
        self.fc1_b = xp.zeros(256, dtype=xp.float32)
        self.fc2_w = xp.asarray(rng.normal(0, np.sqrt(2 / 256), (256, classes)).astype(np.float32))
        self.fc2_b = xp.zeros(classes, dtype=xp.float32)
        self.dfc1_w, self.dfc1_b = xp.zeros_like(self.fc1_w), xp.zeros_like(self.fc1_b)
        self.dfc2_w, self.dfc2_b = xp.zeros_like(self.fc2_w), xp.zeros_like(self.fc2_b)
        self._layers = [
            self.conv1a, self.bn1a, self.conv1b, self.bn1b,
            self.conv2a, self.bn2a, self.conv2b, self.bn2b,
            self.conv3a, self.bn3a, self.conv3b, self.bn3b,
            self.conv4, self.bn4,
        ]
        self.params = []
        self.grads = []
        for layer in self._layers:
            if isinstance(layer, Conv2D):
                self.params.extend([layer.w, layer.b])
                self.grads.extend([layer.dw, layer.db])
            else:
                self.params.extend([layer.gamma, layer.beta])
                self.grads.extend([layer.dgamma, layer.dbeta])
        self.params.extend([self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b])
        self.grads.extend([self.dfc1_w, self.dfc1_b, self.dfc2_w, self.dfc2_b])

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return xp.maximum(z, 0)

    def forward(self, x: np.ndarray, training: bool = False) -> tuple[np.ndarray, np.ndarray]:
        self.z1a = self.conv1a.forward(x); self.n1a = self.bn1a.forward(self.z1a, training); self.a1a = self._relu(self.n1a)
        self.z1b = self.conv1b.forward(self.a1a); self.n1b = self.bn1b.forward(self.z1b, training); self.a1b = self._relu(self.n1b); self.p1 = self.pool1.forward(self.a1b)
        self.z2a = self.conv2a.forward(self.p1); self.n2a = self.bn2a.forward(self.z2a, training); self.a2a = self._relu(self.n2a)
        self.z2b = self.conv2b.forward(self.a2a); self.n2b = self.bn2b.forward(self.z2b, training); self.a2b = self._relu(self.n2b); self.p2 = self.pool2.forward(self.a2b)
        self.z3a = self.conv3a.forward(self.p2); self.n3a = self.bn3a.forward(self.z3a, training); self.a3a = self._relu(self.n3a)
        self.z3b = self.conv3b.forward(self.a3a); self.n3b = self.bn3b.forward(self.z3b, training); self.a3b = self._relu(self.n3b); self.p3 = self.pool3.forward(self.a3b)
        self.z4 = self.conv4.forward(self.p3); self.n4 = self.bn4.forward(self.z4, training); self.a4 = self._relu(self.n4)
        self.spatial_mean = self.a4.mean(axis=(1, 2))
        self.spatial_max = self.a4.max(axis=(1, 2))
        self.spatial_argmax = self.a4.reshape(self.a4.shape[0], -1, self.a4.shape[-1]).argmax(axis=1)
        self.features = xp.concatenate([self.spatial_mean, self.spatial_max], axis=1)
        self.hidden_pre = self.features @ self.fc1_w + self.fc1_b
        self.hidden = self._relu(self.hidden_pre)
        if training:
            self.dropout_mask = xp.asarray(self.rng.random(self.hidden.shape) >= self.dropout_rate, dtype=xp.float32) / (1 - self.dropout_rate)
            self.hidden_used = self.hidden * self.dropout_mask
        else:
            self.hidden_used = self.hidden
        return self.hidden_used @ self.fc2_w + self.fc2_b, self.features

    def loss_backward(self, x: np.ndarray, y: np.ndarray, label_smoothing: float = 0.05) -> tuple[float, float]:
        logits, _ = self.forward(x, training=True)
        shifted = logits - logits.max(axis=1, keepdims=True)
        prob = xp.exp(shifted); prob /= prob.sum(axis=1, keepdims=True)
        count = len(y)
        target = xp.full_like(prob, label_smoothing / prob.shape[1])
        target[xp.arange(count), y] += 1 - label_smoothing
        loss = -(target * xp.log(prob + 1e-12)).sum(axis=1).mean()
        accuracy = float((prob.argmax(axis=1) == y).mean())
        grad = (prob - target) / count
        self.dfc2_w[...] = self.hidden_used.T @ grad; self.dfc2_b[...] = grad.sum(axis=0)
        dh = grad @ self.fc2_w.T; dh_pre = dh * self.dropout_mask * (self.hidden_pre > 0)
        self.dfc1_w[...] = self.features.T @ dh_pre; self.dfc1_b[...] = dh_pre.sum(axis=0)
        df = dh_pre @ self.fc1_w.T
        channels = self.a4.shape[-1]
        spatial = self.a4.shape[1] * self.a4.shape[2]
        dmean = xp.broadcast_to(df[:, None, None, :channels] / spatial, self.a4.shape).copy()
        dmax = xp.zeros_like(self.a4)
        flat = dmax.reshape(dmax.shape[0], -1, channels)
        flat[xp.arange(count)[:, None], self.spatial_argmax, xp.arange(channels)[None, :]] = df[:, channels:]
        da4 = (dmean + dmax) * (self.n4 > 0)
        dz4 = self.bn4.backward(da4); dp3 = self.conv4.backward(dz4)
        da3b = self.pool3.backward(dp3) * (self.n3b > 0); dz3b = self.bn3b.backward(da3b); da3a = self.conv3b.backward(dz3b)
        da3a = da3a * (self.n3a > 0); dz3a = self.bn3a.backward(da3a); dp2 = self.conv3a.backward(dz3a)
        da2b = self.pool2.backward(dp2) * (self.n2b > 0); dz2b = self.bn2b.backward(da2b); da2a = self.conv2b.backward(dz2b)
        da2a = da2a * (self.n2a > 0); dz2a = self.bn2a.backward(da2a); dp1 = self.conv2a.backward(dz2a)
        da1b = self.pool1.backward(dp1) * (self.n1b > 0); dz1b = self.bn1b.backward(da1b); da1a = self.conv1b.backward(dz1b)
        da1a = da1a * (self.n1a > 0); dz1a = self.bn1a.backward(da1a); self.conv1a.backward(dz1a)
        return float(loss), accuracy

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {f"p{i}": p for i, p in enumerate(self.params)}
        for i, layer in enumerate(self._layers):
            if isinstance(layer, BatchNorm2D):
                state[f"bn_mean_{i}"] = layer.running_mean
                state[f"bn_var_{i}"] = layer.running_var
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for i, param in enumerate(self.params):
            param[...] = state[f"p{i}"]
        for i, layer in enumerate(self._layers):
            if isinstance(layer, BatchNorm2D):
                mean_key, var_key = f"bn_mean_{i}", f"bn_var_{i}"
                if mean_key in state:
                    layer.running_mean[...] = state[mean_key]
                    layer.running_var[...] = state[var_key]


class Adam:
    def __init__(self, model: NumpyCNN, lr: float = 1e-3, weight_decay: float = 1e-4, clip_norm: float = 5.0):
        self.lr, self.t, self.weight_decay, self.clip_norm = lr, 0, weight_decay, clip_norm
        self.m = [xp.zeros_like(p) for p in model.params]
        self.v = [xp.zeros_like(p) for p in model.params]

    def step(self, model: NumpyCNN, lr: float | None = None) -> None:
        self.t += 1
        current_lr = self.lr if lr is None else lr
        grad_norm = xp.sqrt(sum((g * g).sum() for g in model.grads))
        scale = min(1.0, self.clip_norm / (float(grad_norm) + 1e-8))
        for p, g, m, v in zip(model.params, model.grads, self.m, self.v):
            g = g * scale
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g * g
            update = (m / (1 - 0.9 ** self.t)) / (xp.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)
            if p.ndim > 1:
                update = update + self.weight_decay * p
            p[:] -= current_lr * update


@dataclass
class OpenSetConfig:
    confidence_threshold: float
    distance_threshold: np.ndarray
    prototypes: np.ndarray
    margin_threshold: float = 0.0


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    p = xp.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def predict_batches(model: NumpyCNN, paths: list[Path], size: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = seed_everything(0)
    all_p, all_f = [], []
    dummy = np.zeros(len(paths), dtype=np.int64)
    for x, _ in batches(paths, dummy, batch_size, size, False, rng, shuffle=False):
        logits, features = model.forward(x, training=False)
        all_p.append(probabilities(logits)); all_f.append(features)
    return xp.concatenate(all_p), xp.concatenate(all_f)


def _unit_features(features: np.ndarray) -> np.ndarray:
    return features / (xp.linalg.norm(features, axis=1, keepdims=True) + 1e-8)


def calibrate_open_set(
    prob: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    classes: int,
    prototype_features: np.ndarray | None = None,
    prototype_labels: np.ndarray | None = None,
) -> OpenSetConfig:
    labels = xp.asarray(labels, dtype=xp.int64)
    prototype_features = features if prototype_features is None else prototype_features
    prototype_labels = labels if prototype_labels is None else xp.asarray(prototype_labels, dtype=xp.int64)
    normalized = _unit_features(prototype_features)
    global_prototype = normalized.mean(axis=0)
    prototypes = xp.stack([
        normalized[prototype_labels == c].mean(axis=0) if bool((prototype_labels == c).any()) else global_prototype
        for c in range(classes)
    ])
    prototypes = _unit_features(prototypes)
    eval_features = _unit_features(features)
    pred = prob.argmax(axis=1)
    confidence = prob.max(axis=1)
    order = xp.sort(prob, axis=1)
    margin = order[:, -1] - order[:, -2]
    distance = 1.0 - (eval_features * prototypes[pred]).sum(axis=1)
    confidence_threshold = float(xp.quantile(confidence, 0.03))
    global_distance = float(xp.quantile(distance, 0.97))
    class_thresholds = []
    for c in range(classes):
        values = distance[pred == c]
        class_thresholds.append(float(xp.quantile(values, 0.97)) if len(values) >= 3 else global_distance)
    margin_threshold = max(0.0, float(xp.quantile(margin, 0.03)))
    return OpenSetConfig(confidence_threshold, xp.asarray(class_thresholds, dtype=xp.float32), prototypes, margin_threshold)


def open_set_predict(prob: np.ndarray, features: np.ndarray, config: OpenSetConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = prob.argmax(axis=1)
    confidence = prob.max(axis=1)
    order = xp.sort(prob, axis=1)
    margin = order[:, -1] - order[:, -2]
    normalized = _unit_features(features)
    distance = 1.0 - (normalized * config.prototypes[labels]).sum(axis=1)
    thresholds = xp.asarray(config.distance_threshold)
    if thresholds.ndim == 0:
        thresholds = xp.full((config.prototypes.shape[0],), thresholds)
    known = (confidence >= config.confidence_threshold) & (distance <= thresholds[labels]) & (margin >= config.margin_threshold)
    return labels, known, distance


def save_model(path: Path, model: NumpyCNN, classes: list[str], config: OpenSetConfig, image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {key: to_cpu(value) for key, value in model.state_dict().items()}
    np.savez_compressed(
        path,
        **cpu_state,
        prototypes=to_cpu(config.prototypes),
        confidence_threshold=config.confidence_threshold,
        distance_threshold=to_cpu(config.distance_threshold),
        margin_threshold=config.margin_threshold,
    )
    path.with_suffix(".json").write_text(json.dumps({"classes": classes, "image_size": image_size}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(path: Path) -> tuple[NumpyCNN, list[str], OpenSetConfig, int]:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    state = np.load(path)
    model = NumpyCNN(len(meta["classes"]), seed_everything(0))
    model.load_state_dict(state)
    margin = float(state["margin_threshold"]) if "margin_threshold" in state else 0.0
    config = OpenSetConfig(
        float(state["confidence_threshold"]),
        xp.asarray(state["distance_threshold"]),
        xp.asarray(state["prototypes"]),
        margin,
    )
    return model, meta["classes"], config, int(meta["image_size"])


def draw_report(history: dict[str, list[float]], labels: np.ndarray, pred: np.ndarray, class_names: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["loss"]) + 1)
    ax[0].plot(epochs, history["loss"], label="train loss")
    ax[0].plot(epochs, history["val_loss"], label="valid loss")
    ax[0].legend(); ax[0].set_xlabel("epoch")
    ax[1].plot(epochs, history["acc"], label="train accuracy")
    ax[1].plot(epochs, history["val_acc"], label="valid accuracy")
    ax[1].legend(); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(output / "training_curves.png", dpi=160); plt.close(fig)
    labels, pred = to_cpu(labels), to_cpu(pred)
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    np.add.at(cm, (labels, pred), 1)
    fig, ax = plt.subplots(figsize=(max(7, len(class_names) * 0.35), max(6, len(class_names) * 0.32)))
    ax.imshow(cm, cmap="Blues"); ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=180); plt.close(fig)
