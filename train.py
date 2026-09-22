from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from openset import (Adam, NumpyCNN, batches, calibrate_open_set, collect_dataset, draw_report,
                     predict_batches, probabilities, save_model, seed_everything, stratified_split)
from openset import array_module, configure_device


def evaluate(model, paths, labels, size, batch_size):
    rng = seed_everything(2026); losses=[]; correct=[]
    backend = array_module()
    for x, y in batches(paths, labels, batch_size, size, False, rng, shuffle=False):
        logits, _ = model.forward(x); p = probabilities(logits)
        losses.append(float(-backend.log(p[backend.arange(len(y)), y] + 1e-12).mean())); correct.append(int((p.argmax(1) == y).sum()))
    return float(np.mean(losses)), float(np.sum(correct) / len(labels))


def main():
    parser = argparse.ArgumentParser(description="纯 NumPy 开集识别训练")
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    parser.add_argument("--data-root", default="data"); parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=24); parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=5e-4); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=.1, help="从全部样本中按类别留作验证的比例")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="auto 会优先使用 CuPy GPU")
    args = parser.parse_args(); print("计算设备：", configure_device(args.device)); rng = seed_everything(args.seed)
    given_train_p, given_train_l, given_valid_p, given_valid_l = collect_dataset(Path(args.data_root) / args.dataset)
    # 题目数据的两个文件夹均为可用标注数据；合并后再按类别划分，确保全部类别参与训练。
    all_paths, all_labels = given_train_p + given_valid_p, given_train_l + given_valid_l
    train_p, train_l, valid_p, valid_l = stratified_split(all_paths, all_labels, args.val_ratio, rng)
    classes = sorted(set(all_labels)); index = {name:i for i,name in enumerate(classes)}
    y_train=np.array([index[x] for x in train_l]); y_valid=np.array([index[x] for x in valid_l])
    print(f"{args.dataset}: 全部 {len(all_paths)} 张、{len(classes)} 类；训练 {len(train_p)} 张，验证 {len(valid_p)} 张")
    counts=np.bincount(y_train, minlength=len(classes)); sample_weights=1.0 / counts[y_train]
    model=NumpyCNN(len(classes), rng); optimizer=Adam(model, args.lr); best_acc=-1.; best_state=None
    history={"loss":[], "acc":[], "val_loss":[], "val_acc":[]}
    for epoch in range(1, args.epochs + 1):
        losses=[]; hits=[]; count=0
        for x,y in batches(train_p, y_train, args.batch_size, args.image_size, True, rng, sample_weights=sample_weights):
            loss,acc=model.loss_backward(x,y); optimizer.step(model); losses.append(loss); hits.append(acc*len(y)); count += len(y)
        vl,va=evaluate(model, valid_p,y_valid,args.image_size,args.batch_size)
        history["loss"].append(float(np.mean(losses))); history["acc"].append(float(np.sum(hits)/count)); history["val_loss"].append(vl); history["val_acc"].append(va)
        print(f"Epoch {epoch:03d}/{args.epochs}: loss={history['loss'][-1]:.4f}, train_acc={history['acc'][-1]:.3f}, val_acc={va:.3f}")
        if va > best_acc: best_acc=va; best_state={k:v.copy() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_prob, train_feat=predict_batches(model,train_p,args.image_size,args.batch_size)
    config=calibrate_open_set(train_prob,train_feat,y_train,len(classes))
    valid_prob,valid_feat=predict_batches(model,valid_p,args.image_size,args.batch_size)
    pred=valid_prob.argmax(1); output=Path("outputs")/args.dataset
    save_model(output/"best_model.npz",model,classes,config,args.image_size); draw_report(history,y_valid,pred,classes,output)
    report={"dataset":args.dataset,"classes":len(classes),"train_images":len(train_p),"valid_images":len(valid_p),"best_valid_accuracy":best_acc,"confidence_threshold":config.confidence_threshold,"distance_threshold":config.distance_threshold}
    (output/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("完成。模型和可视化结果保存在", output)

if __name__ == "__main__": main()
