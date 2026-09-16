from __future__ import annotations
import argparse
from pathlib import Path
from openset import display_label, load_image, load_model, open_set_predict, probabilities, seed_everything

def main():
    p=argparse.ArgumentParser(description="单张开放集识别"); p.add_argument("--dataset",choices=["dataset1","dataset2"],required=True); p.add_argument("--image",required=True); args=p.parse_args()
    model,classes,config,size=load_model(Path("outputs")/args.dataset/"best_model.npz")
    x=load_image(Path(args.image),size,False,seed_everything(0))[None]; logits,features=model.forward(x); prob=probabilities(logits); labels,known,distance=open_set_predict(prob,features,config)
    result=display_label(classes[labels[0]]) if known[0] else "未知类别"
    print(f"结果：{result}\n置信度：{prob.max():.2%}\n原型距离：{distance[0]:.4f}\n已知判断：{'是' if known[0] else '否'}")
if __name__ == "__main__": main()
