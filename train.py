from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openset import (
    Adam,
    NumpyCNN,
    array_module,
    batches,
    calibrate_open_set,
    collect_dataset,
    configure_device,
    draw_report,
    predict_batches,
    probabilities,
    save_model,
    seed_everything,
    stratified_split,
    to_cpu,
)


def evaluate(model, paths, labels, size, batch_size):
    rng = seed_everything(2026)
    backend = array_module()
    losses, predictions = [], []
    for x, y in batches(paths, labels, batch_size, size, False, rng, shuffle=False):
        logits, _ = model.forward(x, training=False)
        prob = probabilities(logits)
        losses.append(float(-backend.log(prob[backend.arange(len(y)), y] + 1e-12).mean()))
        predictions.append(to_cpu(prob.argmax(1)))
    predictions = np.concatenate(predictions)
    labels = np.asarray(labels)
    accuracy = float((predictions == labels).mean())
    per_class = []
    for class_id in np.unique(labels):
        mask = labels == class_id
        per_class.append(float((predictions[mask] == labels[mask]).mean()))
    return float(np.mean(losses)), accuracy, float(np.mean(per_class))


def scheduled_lr(base_lr: float, epoch: int, epochs: int) -> float:
    warmup = max(1, min(5, epochs // 10))
    if epoch <= warmup:
        return base_lr * epoch / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="NumPy open-set recognition training")
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    print("device:", configure_device(args.device))
    rng = seed_everything(args.seed)
    given_train_p, given_train_l, given_valid_p, given_valid_l = collect_dataset(Path(args.data_root) / args.dataset)
    all_paths, all_labels = given_train_p + given_valid_p, given_train_l + given_valid_l
    train_p, train_l, valid_p, valid_l = stratified_split(all_paths, all_labels, args.val_ratio, rng)
    classes = sorted(set(all_labels))
    index = {name: i for i, name in enumerate(classes)}
    y_train = np.asarray([index[x] for x in train_l], dtype=np.int64)
    y_valid = np.asarray([index[x] for x in valid_l], dtype=np.int64)
    print(f"{args.dataset}: {len(all_paths)} images, {len(classes)} classes; train {len(train_p)}, valid {len(valid_p)}")

    counts = np.bincount(y_train, minlength=len(classes))
    sample_weights = 1.0 / counts[y_train]
    model = NumpyCNN(len(classes), rng)
    optimizer = Adam(model, args.lr, weight_decay=1e-4)
    best_acc, best_balanced, best_state = -1.0, -1.0, None
    stale_epochs = 0
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "val_balanced_acc": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        current_lr = scheduled_lr(args.lr, epoch, args.epochs)
        losses, hits, count = [], [], 0
        for x, y in batches(train_p, y_train, args.batch_size, args.image_size, True, rng, sample_weights=sample_weights):
            loss, accuracy = model.loss_backward(x, y)
            optimizer.step(model, current_lr)
            losses.append(loss)
            hits.append(accuracy * len(y))
            count += len(y)
        valid_loss, valid_acc, balanced_acc = evaluate(model, valid_p, y_valid, args.image_size, args.batch_size)
        history["loss"].append(float(np.mean(losses)))
        history["acc"].append(float(np.sum(hits) / count))
        history["val_loss"].append(valid_loss)
        history["val_acc"].append(valid_acc)
        history["val_balanced_acc"].append(balanced_acc)
        history["lr"].append(current_lr)
        print(f"epoch {epoch:03d}/{args.epochs}: loss={history['loss'][-1]:.4f}, train={history['acc'][-1]:.3f}, valid={valid_acc:.3f}, balanced={balanced_acc:.3f}, lr={current_lr:.2e}")
        if balanced_acc > best_balanced or (balanced_acc == best_balanced and valid_acc > best_acc):
            best_acc, best_balanced = valid_acc, balanced_acc
            best_state = {key: value.copy() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stop after {args.patience} stale epochs")
                break

    if best_state is None:
        raise RuntimeError("training did not produce a model")
    model.load_state_dict(best_state)
    train_prob, train_feat = predict_batches(model, train_p, args.image_size, args.batch_size)
    valid_prob, valid_feat = predict_batches(model, valid_p, args.image_size, args.batch_size)
    config = calibrate_open_set(
        valid_prob,
        valid_feat,
        y_valid,
        len(classes),
        prototype_features=train_feat,
        prototype_labels=y_train,
    )
    pred = valid_prob.argmax(1)
    output = Path("outputs") / args.dataset
    save_model(output / "best_model.npz", model, classes, config, args.image_size)
    draw_report(history, y_valid, pred, classes, output)
    report = {
        "dataset": args.dataset,
        "classes": len(classes),
        "train_images": len(train_p),
        "valid_images": len(valid_p),
        "best_valid_accuracy": best_acc,
        "best_balanced_accuracy": best_balanced,
        "confidence_threshold": config.confidence_threshold,
        "distance_thresholds": to_cpu(config.distance_threshold).tolist(),
        "margin_threshold": config.margin_threshold,
        "epochs_ran": len(history["loss"]),
    }
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done:", output)


if __name__ == "__main__":
    main()
