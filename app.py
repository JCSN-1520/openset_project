from __future__ import annotations
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from openset import configure_device, display_label, load_image, load_model, open_set_predict, probabilities, seed_everything

class Application:
    def __init__(self, dataset: str):
        self.model,self.classes,self.config,self.size=load_model(Path("outputs")/dataset/"best_model.npz")
        self.root=tk.Tk(); self.root.title("开放集图像识别演示"); self.root.geometry("620x570")
        tk.Button(self.root,text="选择图片",font=("Microsoft YaHei",14),command=self.choose).pack(pady=16)
        self.image_label=tk.Label(self.root); self.image_label.pack(); self.result=tk.Label(self.root,text="请选择一张图片",font=("Microsoft YaHei",14),justify="left"); self.result.pack(pady=16)
    def choose(self):
        name=filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if not name:return
        image=Image.open(name).convert("RGB"); image.thumbnail((380,330)); self.photo=ImageTk.PhotoImage(image); self.image_label.configure(image=self.photo)
        x=load_image(Path(name),self.size,False,seed_everything(0))[None]; logits,features=self.model.forward(x); prob=probabilities(logits); labels,known,distance=open_set_predict(prob,features,self.config)
        label=display_label(self.classes[labels[0]]) if known[0] else "未知类别"
        self.result.configure(text=f"识别结果：{label}\n置信度：{prob.max():.2%}\n类别原型距离：{distance[0]:.4f}\n开放集判定：{'已知类别' if known[0] else '未知类别'}")
    def run(self): self.root.mainloop()
def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["dataset1","dataset2"],required=True);p.add_argument("--device",choices=["auto","cpu","gpu"],default="auto");args=p.parse_args();configure_device(args.device);Application(args.dataset).run()
if __name__=="__main__":main()
