# 运行说明

`README.md` 保留作者的科普文章原文。本页说明当前程序的行为：CNN 对整张图片输出“月饼/非月饼”。蛋糕、曲奇、蛋挞等负样本只统一标成“非月饼”，不会输出这些具体类别。

随包模型是从随机权重训练的 Residual CNN，没有加载外部预训练模型。分数用于阈值分类，未校准成概率。

## 下载原始 AI 月饼图片

下载 GitHub Release `v1.1.0` 里的两份**未压缩 TAR**：

- [mooncake-ai-originals-01.tar](https://github.com/AIXF666/cnn-mooncake/releases/download/v1.1.0/mooncake-ai-originals-01.tar)
- [mooncake-ai-originals-02.tar](https://github.com/AIXF666/cnn-mooncake/releases/download/v1.1.0/mooncake-ai-originals-02.tar)

把它们放到项目根目录解包。TAR 不会重编码或压缩 PNG；解包后文件位于 `data/mooncake/ai_original/`。

macOS：

~~~sh
tar -xf mooncake-ai-originals-01.tar -C .
tar -xf mooncake-ai-originals-02.tar -C .
~~~

Windows PowerShell：

~~~powershell
tar -xf .\mooncake-ai-originals-01.tar -C .
tar -xf .\mooncake-ai-originals-02.tar -C .
~~~

每个 TAR 不到 2 GiB，图片文件内容与生成源 PNG 逐字节一致。

## 安装并训练

macOS（需 Homebrew）：

~~~sh
brew install python@3.12
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
~~~

Windows 命令提示符：

~~~bat
winget install -e --id Python.Python.3.12
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train.py
~~~

训练图片包括 998 张原始 AI 月饼图、207 张真实月饼图，以及 200 张训练用和 200 张验证用的非月饼图。训练从随机初始化开始，默认最多 40 轮；验证 AUC 停滞后提前停止。模型按验证集来源平衡 AUC 选轮次，阈值用验证集正负样本共同确定。训练日志和每轮权重保存在 `checkpoints/from-scratch/`。

## 预测

默认使用随包提供的 `checkpoints/best.pt`：

~~~sh
.venv/bin/python predict.py "你的图片.jpg"
~~~

Windows：

~~~bat
.venv\Scripts\python.exe predict.py "你的图片.jpg"
~~~

改用刚训练出的模型：

~~~sh
.venv/bin/python predict.py "你的图片.jpg" --checkpoint checkpoints/from-scratch/best.pt
~~~

真实照片效果受拍摄角度、月饼外形和背景影响；发布说明中的评估数字来自开发数据，不代表任意照片上的准确率。请用未参与训练和调参的新照片复核。

如需按显卡安装 CUDA 版 PyTorch，请参照 <https://pytorch.org/get-started/locally/>。
