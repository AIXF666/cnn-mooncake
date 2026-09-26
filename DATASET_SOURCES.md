# 月饼训练图片来源

data/mooncake 中共有 1205 张月饼图片：

- ai_0001.jpg 至 ai_0998.jpg：原始月饼图片由 GPT Image 2 生成，后来抠出月饼并合成随机噪声、彩色色块或真实背景。这些图片只表示 AI 合成图像的分布。
- real_0001.jpg 至 real_0207.jpg：公开来源的真实月饼图片，其中 184 张来自 DimSum50 数据集的 Mooncake 类，23 张来自 Wikimedia Commons 的月饼图片。它们经过缩小或 JPG 重新编码。

data/source_manifest.csv 把打包文件名对应回原始文件名和来源页面。真实图片的许可及署名要求应以对应来源页面为准。DimSum50 来源：https://www.kaggle.com/datasets/rock3yu/dimsum50-0-1

本包没有包含用户通过对话上传的照片。示例 best.pt 使用同一批月饼图片训练；训练和验证只包含月饼，因此验证误差不能表示它对非月饼的误报率。
