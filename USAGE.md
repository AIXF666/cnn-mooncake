# 技术运行说明

README.md 保留作者的文章原文。本页记录代码的实际行为：train.py 把月饼图片按 8:2 分为训练集和验证集；文章最后下载的新图片是手动测试，没有预先划分的测试集。

这个安装包只包含月饼训练图片和一个月饼类别。train.py 从零训练单类 CNN；predict.py 根据模型的重建误差与月饼特征距离，输出“像月饼”或“不像月饼”。模型不会输出蛋糕、汽车等其他类别。

data/mooncake 中有 1205 张图片：998 张 AI 月饼合成背景变体和 207 张来自公开数据集的真实月饼图片。没有用户拍摄的照片，也没有非月饼训练图片。图片来源见 DATASET_SOURCES.md 与 data/source_manifest.csv。

## macOS

安装 Homebrew 后，在解压的项目目录运行：

~~~sh
brew install python@3.12
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/python predict.py "你的图片.jpg"
~~~

## Windows 命令提示符

~~~bat
winget install -e --id Python.Python.3.12
~~~

关闭并重新打开命令提示符，回到项目目录：

~~~bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe predict.py "你的图片.jpg"
~~~

默认训练 35 轮，80% 月饼图用于训练，20% 月饼图用于验证；每轮模型保存在 checkpoints/from-scratch/，其中 best.pt 是验证误差最低的一轮。若想启用早停，可在训练命令后加 --patience 10。重新训练后，可用以下命令测试新模型：

~~~sh
python predict.py "你的图片.jpg" --checkpoint checkpoints/from-scratch/best.pt
~~~

上面这条简写命令要求先激活虚拟环境；未激活时请改用对应系统的虚拟环境 Python 路径。

checkpoints/oneclass/best.pt 是此前选定的示例模型，只用于让读者解压后直接体验；predict.py 默认始终读取它，不会自动切换到新训练的模型。train.py 始终从零训练，也不会加载它。只有显式传入 --checkpoint checkpoints/from-scratch/best.pt，predict.py 才读取读者新训练的模型。

示例模型使用与本包同一批月饼图片训练，但单类判断仍可能把蛋糕等糕点误认为月饼。异常分数不是概率，也不能保证任意照片的判断正确。

如果要按自己的显卡安装 CUDA 版 PyTorch，请参照官方安装页：https://pytorch.org/get-started/locally/
