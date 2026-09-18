"""纯 NumPy 的图像分类与开放集识别核心实现。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
xp = np
DEVICE = "cpu"


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
    """返回当前计算后端；供训练脚本在 GPU 模式下使用。"""
    return xp


def seed_everything(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def image_label(path: Path) -> str:
    """从扁平验证集文件名恢复类别名，兼容 CUB 鸟类和动物数据集。"""
    stem = path.stem
    # CUB: Acadian_Flycatcher_0006_795595 -> Acadian_Flycatcher
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    # 动物数据：giant+panda_10001 -> giant+panda
    return stem.rsplit("_", 1)[0]


def directory_label(name: str) -> str:
    """CUB 训练目录形如 001.Black_footed_Albatross，验证文件没有数字前缀。"""
    prefix, dot, rest = name.partition(".")
    return rest if dot and prefix.isdigit() else name


def canonical_label(name: str) -> str:
    """忽略原始数据中仅大小写不同的同一类别命名。"""
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


def stratified_split(paths: list[Path], labels: list[str], valid_ratio: float, rng: np.random.Generator) -> tuple[list[Path], list[str], list[Path], list[str]]:
    """每个类别均同时出现在训练和验证中，保证所有给定类别都会训练。"""
    by_label: dict[str, list[int]] = {}
    for i, label in enumerate(labels): by_label.setdefault(label, []).append(i)
    train_idx, valid_idx = [], []
    for indices in by_label.values():
        ordered = rng.permutation(indices); n_valid = max(1, round(len(indices) * valid_ratio))
        n_valid = min(n_valid, len(indices) - 1) if len(indices) > 1 else 0
        valid_idx.extend(ordered[:n_valid]); train_idx.extend(ordered[n_valid:])
    return ([paths[i] for i in train_idx], [labels[i] for i in train_idx],
            [paths[i] for i in valid_idx], [labels[i] for i in valid_idx])


def load_image(path: Path, size: int, augment: bool, rng: np.random.Generator) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise RuntimeError(f"无法读取图像：{path}") from exc
    if augment:
        if rng.random() < 0.5:
            array = array[:, ::-1]
        # 亮度、对比度和轻微噪声用于提升对拍摄环境变化的鲁棒性。
        array = np.clip(array * rng.uniform(0.78, 1.22) + rng.uniform(-0.08, 0.08), 0, 1)
        if rng.random() < 0.25:
            array = np.clip(array + rng.normal(0, 0.015, array.shape), 0, 1)
    return (array - 0.5) / 0.5


def batches(paths: list[Path], labels: np.ndarray, batch_size: int, size: int, augment: bool,
            rng: np.random.Generator, shuffle: bool = True, sample_weights: np.ndarray | None = None) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    labels = xp.asarray(labels, dtype=xp.int64)
    if shuffle and sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64); weights /= weights.sum()
        order = rng.choice(len(paths), size=len(paths), replace=True, p=weights)
    else:
        order = rng.permutation(len(paths)) if shuffle else np.arange(len(paths))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        x = xp.asarray(np.stack([load_image(paths[i], size, augment, rng) for i in idx]))
        yield x, labels[idx]


class Conv2D:
    def __init__(self, in_channels: int, out_channels: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / (in_channels * 9))
        self.w = xp.asarray(rng.normal(0, scale, (out_channels, in_channels, 3, 3)).astype(np.float32))
        self.b = xp.zeros(out_channels, dtype=xp.float32)
        self.dw, self.db = xp.zeros_like(self.w), xp.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # SAME 3x3 convolution, stride 2. Layout: NHWC.
        self.x_shape = x.shape
        padded = xp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        self.padded = padded
        b, h, w, c = x.shape
        oh, ow = (h + 1) // 2, (w + 1) // 2
        self.cols = xp.empty((b, oh, ow, c * 9), dtype=xp.float32)
        for iy in range(3):
            for ix in range(3):
                self.cols[..., (iy * 3 + ix) * c:(iy * 3 + ix + 1) * c] = padded[:, iy:iy + oh * 2:2, ix:ix + ow * 2:2, :]
        kernel = self.w.transpose(0, 2, 3, 1).reshape(self.w.shape[0], -1)
        return (self.cols.reshape(-1, c * 9) @ kernel.T + self.b).reshape(b, oh, ow, -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        b, oh, ow, out_c = grad.shape
        c = self.x_shape[-1]
        grad2 = grad.reshape(-1, out_c)
        col2 = self.cols.reshape(-1, c * 9)
        self.dw[...] = (grad2.T @ col2).reshape(out_c, 3, 3, c).transpose(0, 3, 1, 2)
        self.db[...] = grad2.sum(axis=0)
        kernel = self.w.transpose(0, 2, 3, 1).reshape(out_c, -1)
        dcol = (grad2 @ kernel).reshape(b, oh, ow, 3, 3, c)
        dpadded = xp.zeros_like(self.padded)
        for iy in range(3):
            for ix in range(3):
                dpadded[:, iy:iy + oh * 2:2, ix:ix + ow * 2:2, :] += dcol[:, :, :, iy, ix, :]
        return dpadded[:, 1:-1, 1:-1, :]


class NumpyCNN:
    """三层卷积 + 隐藏全连接层的轻量网络，含完整反向传播。"""
    def __init__(self, classes: int, rng: np.random.Generator):
        self.conv1, self.conv2, self.conv3 = Conv2D(3, 24, rng), Conv2D(24, 48, rng), Conv2D(48, 64, rng)
        self.fc1_w = xp.asarray(rng.normal(0, np.sqrt(2 / 64), (64, 128)).astype(np.float32)); self.fc1_b = xp.zeros(128, dtype=xp.float32)
        self.fc2_w = xp.asarray(rng.normal(0, np.sqrt(2 / 128), (128, classes)).astype(np.float32)); self.fc2_b = xp.zeros(classes, dtype=xp.float32)
        self.dfc1_w, self.dfc1_b = xp.zeros_like(self.fc1_w), xp.zeros_like(self.fc1_b)
        self.dfc2_w, self.dfc2_b = xp.zeros_like(self.fc2_w), xp.zeros_like(self.fc2_b)
        self.params = [self.conv1.w, self.conv1.b, self.conv2.w, self.conv2.b, self.conv3.w, self.conv3.b, self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b]
        self.grads = [self.conv1.dw, self.conv1.db, self.conv2.dw, self.conv2.db, self.conv3.dw, self.conv3.db, self.dfc1_w, self.dfc1_b, self.dfc2_w, self.dfc2_b]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.z1 = self.conv1.forward(x); self.a1 = xp.maximum(self.z1, 0)
        self.z2 = self.conv2.forward(self.a1); self.a2 = xp.maximum(self.z2, 0)
        self.z3 = self.conv3.forward(self.a2); self.a3 = xp.maximum(self.z3, 0)
        self.features = self.a3.mean(axis=(1, 2))
        self.hidden_pre = self.features @ self.fc1_w + self.fc1_b; self.hidden = xp.maximum(self.hidden_pre, 0)
        return self.hidden @ self.fc2_w + self.fc2_b, self.features

    def loss_backward(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        logits, _ = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        prob = xp.exp(shifted); prob /= prob.sum(axis=1, keepdims=True)
        loss = -xp.log(prob[xp.arange(len(y)), y] + 1e-12).mean()
        accuracy = float((prob.argmax(axis=1) == y).mean())
        grad = prob.copy(); grad[xp.arange(len(y)), y] -= 1; grad /= len(y)
        self.dfc2_w[...] = self.hidden.T @ grad; self.dfc2_b[...] = grad.sum(axis=0)
        dh = grad @ self.fc2_w.T; dh_pre = dh * (self.hidden_pre > 0)
        self.dfc1_w[...] = self.features.T @ dh_pre; self.dfc1_b[...] = dh_pre.sum(axis=0)
        df = dh_pre @ self.fc1_w.T
        da3 = xp.broadcast_to(df[:, None, None, :] / (self.a3.shape[1] * self.a3.shape[2]), self.a3.shape).copy()
        da2 = self.conv3.backward(da3 * (self.z3 > 0))
        da1 = self.conv2.backward(da2 * (self.z2 > 0))
        self.conv1.backward(da1 * (self.z1 > 0))
        return float(loss), accuracy

    def state_dict(self) -> dict[str, np.ndarray]:
        return {f"p{i}": p for i, p in enumerate(self.params)}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for i, param in enumerate(self.params): param[...] = state[f"p{i}"]


class Adam:
    def __init__(self, model: NumpyCNN, lr: float = 1e-3):
        self.lr, self.t = lr, 0
        self.m = [xp.zeros_like(p) for p in model.params]; self.v = [xp.zeros_like(p) for p in model.params]

    def step(self, model: NumpyCNN) -> None:
        self.t += 1
        for p, g, m, v in zip(model.params, model.grads, self.m, self.v):
            m[:] = .9 * m + .1 * g; v[:] = .999 * v + .001 * g * g
            p[:] -= self.lr * (m / (1 - .9 ** self.t)) / (xp.sqrt(v / (1 - .999 ** self.t)) + 1e-8)


@dataclass
class OpenSetConfig:
    confidence_threshold: float
    distance_threshold: float
    prototypes: np.ndarray


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    p = xp.exp(logits); return p / p.sum(axis=1, keepdims=True)


def predict_batches(model: NumpyCNN, paths: list[Path], size: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = seed_everything(0); all_p, all_f = [], []
    dummy = np.zeros(len(paths), dtype=np.int64)
    for x, _ in batches(paths, dummy, batch_size, size, False, rng, shuffle=False):
        logits, features = model.forward(x); all_p.append(probabilities(logits)); all_f.append(features)
    return xp.concatenate(all_p), xp.concatenate(all_f)


def calibrate_open_set(prob: np.ndarray, features: np.ndarray, labels: np.ndarray, classes: int) -> OpenSetConfig:
    labels = xp.asarray(labels)
    prototypes = xp.stack([features[labels == c].mean(axis=0) for c in range(classes)])
    pred = prob.argmax(axis=1); confidence = prob.max(axis=1)
    distance = xp.linalg.norm(features - prototypes[pred], axis=1)
    # 仅用已知验证集标定，使至少 98% 的已知验证样本不被误拒绝。
    return OpenSetConfig(float(xp.quantile(confidence, .02)), float(xp.quantile(distance, .98)), prototypes)


def open_set_predict(prob: np.ndarray, features: np.ndarray, config: OpenSetConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = prob.argmax(axis=1); confidence = prob.max(axis=1)
    distance = xp.linalg.norm(features - config.prototypes[labels], axis=1)
    known = (confidence >= config.confidence_threshold) & (distance <= config.distance_threshold)
    return labels, known, distance


def save_model(path: Path, model: NumpyCNN, classes: list[str], config: OpenSetConfig, image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {key: to_cpu(value) for key, value in model.state_dict().items()}
    np.savez_compressed(path, **cpu_state, prototypes=to_cpu(config.prototypes),
                        confidence_threshold=config.confidence_threshold, distance_threshold=config.distance_threshold)
    path.with_suffix(".json").write_text(json.dumps({"classes": classes, "image_size": image_size}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(path: Path) -> tuple[NumpyCNN, list[str], OpenSetConfig, int]:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8")); state = np.load(path)
    model = NumpyCNN(len(meta["classes"]), seed_everything(0)); model.load_state_dict(state)
    config = OpenSetConfig(float(state["confidence_threshold"]), float(state["distance_threshold"]), xp.asarray(state["prototypes"]))
    return model, meta["classes"], config, int(meta["image_size"])


def draw_report(history: dict[str, list[float]], labels: np.ndarray, pred: np.ndarray, class_names: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); epochs = range(1, len(history["loss"]) + 1)
    ax[0].plot(epochs, history["loss"], label="train loss"); ax[0].plot(epochs, history["val_loss"], label="valid loss"); ax[0].legend(); ax[0].set_xlabel("epoch")
    ax[1].plot(epochs, history["acc"], label="train accuracy"); ax[1].plot(epochs, history["val_acc"], label="valid accuracy"); ax[1].legend(); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(output / "training_curves.png", dpi=160); plt.close(fig)
    labels, pred = to_cpu(labels), to_cpu(pred)
    cm = np.zeros((len(class_names), len(class_names)), dtype=int); np.add.at(cm, (labels, pred), 1)
    fig, ax = plt.subplots(figsize=(max(7, len(class_names) * .35), max(6, len(class_names) * .32)))
    ax.imshow(cm, cmap="Blues"); ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6); ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=180); plt.close(fig)
