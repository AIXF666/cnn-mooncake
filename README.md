# 中秋节，手把手训练一个识别月饼的CNN！🥮

我之前写过《图片识别模型到底是怎么训练出来的？》。今天就是中秋节了，所以我写了一个识别月饼的模型手把手，训练出自己的模型。这篇不需要你提前懂CNN，跟着准备图片、运行代码，最后拿一张新图片考考模型就可以了！

> 有提前训练好的模型best.pt，可以开箱即用

> 运行内存：运行内存至少8G
> 储存空间：至少剩余5G

## 第一步：安装Python和PyTorch

Python和PyTorch是训练模型必装的东西，所以我们要先安装。Python是运行代码需要的语言，PyTorch则帮我们计算CNN、Loss和反向传播。

如果你是Mac，并且安装了Homebrew（安装链接：[Homebrew](https://brew.sh/)）在项目文件夹打开终端运行：
```
brew install python@3.12 && python3.12 -m venv .venv && .venv/bin/python -m pip install torch torchvision pillow
```

如果你是Windows，那么可以运行：
```
winget install -e --id Python.Python.3.12
```

安装后关掉命令提示符，重新打开，在项目文件夹里输入：
```
py -3.12 -m venv .venv && .venv\Scripts\python.exe -m pip install torch torchvision pillow
```

Windows的安装方法可以参考[微软说明](https://learn.microsoft.com/en-us/windows/dev-environment/python)

下载需要时间，现在去吃一块月饼吧！

## 第二步：准备数据集

恭喜你太棒了，你已经完成了第一步！现在我们要完成第二步，就是“写”教程，也就是准备数据集。我们要准备一种图片：就是月饼图片。我准备了1000+张图片，1000+张都是是月饼图片这样子你就不用费尽心思找图片并且标注图片啦。图片的目录是：data/mooncake/

我分成了训练集、验证集和测试集三个。现在来复习一下吧：训练集是平时做题，验证集是模拟考试，测试集留到最后当期末考试。

## 第三步：让CNN开始训练

现在训练的准备工作都完成了，我们就开始训练吧！

我们在《图片识别模型到底是怎么训练出来的？》讲过一个CNN模型首先得先用RGB“变”成数字，也就是说我们要用RGB先处理图片，然后丢给隐藏层进行处理：从图片中提取特征，生成特征图。然后利用反向传播修改模型参数。输入图片 → 做出预测 → 计算Loss → 反向传播 → 调整参数 → 再看下一批图片

如果你是Mac的话运行：
```
.venv/bin/python train.py
```

如果你是Windows运行：
```
.venv\Scripts\python.exe train.py
```

然后终端就会出现各种训练数据，比如Loss等等。

## 训练完成了，现在该验证一下啦

首先，你要先下载一张在训练数据里面没有出现的图片，然后运行：

Mac：
```bash
.venv/bin/python predict.py "你的图片.jpg" --checkpoint checkpoints/from-scratch/best.pt
```

Windows：
```bash
.venv\Scripts\python.exe predict.py "你的图片.jpg" --checkpoint checkpoints/from-scratch/best.pt
```

假如你认为这样子太麻烦了也可以把PTH文件发给Codex，让他帮你写一个客户端。

假如模型识别错误了，你也不用着急，你可以问一问聪明的Codex，或者在评论区留言问一下我，我会乐意为你帮助的。

崩溃了！！！为什么蛋糕曲奇总是识别成月饼，不想要改了，就当作一个可以识别蛋糕曲奇和月饼的模型吧

喵，这次我们真的从图片开始，训练出了一个识别月饼的小CNN。背后除了卷积、Loss和反向传播，当然还有工程师的头发。中秋快乐，快拿月饼考考你的模型吧！🥮🐱
