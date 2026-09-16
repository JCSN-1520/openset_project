# 开集识别项目

本项目使用 **NumPy 手写卷积神经网络** 完成两套图像数据的分类和开放集识别；验收代码不依赖 PyTorch、TensorFlow 或其他深度学习框架。

## 环境

在你的 Conda 环境中安装：

```bash
pip install -r requirements.txt
```

## 训练

在 PyCharm 中将工作目录设为项目根目录，运行：

```bash
python train.py --dataset dataset1 --epochs 35
python train.py --dataset dataset2 --epochs 35
```

训练结果会写入 `outputs/dataset1` 或 `outputs/dataset2`：最佳模型、类别表、开放集阈值、训练曲线、混淆矩阵和验证指标都保存在对应目录。

已安装 `cupy-cuda11x` 时会自动使用 NVIDIA GPU；需要强制指定时，可加 `--device gpu`。无法使用 GPU 时会自动回退至 CPU，例如 `python train.py --dataset dataset1 --device cpu`。

## 单张预测与演示

```bash
python predict.py --dataset dataset1 --image path/to/image.jpg
python app.py --dataset dataset1
```

图形界面中选择任意图片即可显示预测类别、置信度、与类别原型的距离，以及“已知/未知”结论。

## 开放集机制

训练时使用提供数据中的**全部类别**，并将两个原始文件夹的图像合并后按类别随机划分训练/验证样本；不会把任何现有类别当作未知类。模型把最后一层卷积特征的每类均值保存为类别原型；预测时同时考察 Softmax 最大置信度和至预测类别原型的距离。任一指标越过由验证集标定的阈值，就返回“未知类别”。因此，验收时加入的全新动物/鸟类不必重训即可被拒识。

## 代码结构

- `openset.py`：数据读取、NumPy 卷积网络、训练、指标、开放集判定和可视化
- `train.py`：训练入口
- `predict.py`：单张图片预测入口
- `app.py`：Tkinter 演示界面
