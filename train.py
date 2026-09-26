"""从月饼图片出发训练单类 CNN，不需要非月饼训练图片。"""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from oneclass_model import OneClassCNN, anomaly_score


ROOT = Path(__file__).resolve().parent


class MooncakeImages(Dataset):
    def __init__(self, paths, training=False):
        self.paths = paths
        steps = [transforms.Resize((64, 64))]
        if training:
            steps += [
                transforms.RandomRotation(12),
                transforms.ColorJitter(.12, .12, .12, .04),
            ]
        steps += [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        ]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def validation_scores(model, loader, center, device):
    model.eval()
    values = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            values.extend(anomaly_score(model, images, center).cpu().tolist())
    return values


def main():
    parser = argparse.ArgumentParser(description="只用月饼图片从零训练单类 CNN。")
    parser.add_argument("--data", type=Path, default=ROOT / "data/mooncake")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/from-scratch")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=0, help="0 表示训练满指定轮数；正数表示验证误差连续未改善时早停")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 0 or args.batch_size < 1:
        parser.error("训练轮数和批量大小必须大于 0，耐心轮数不能小于 0。")

    paths = sorted(args.data.glob("*.jpg"))
    if len(paths) < 20:
        parser.error(f"至少需要 20 张月饼图片，当前找到 {len(paths)} 张。")
    random.Random(20260925).shuffle(paths)
    split = round(len(paths) * .8)
    train_paths, val_paths = paths[:split], paths[split:]
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device_name)
    train_loader = DataLoader(MooncakeImages(train_paths, True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MooncakeImages(val_paths), batch_size=args.batch_size)
    torch.manual_seed(20260925)
    model = OneClassCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    mse = nn.MSELoss()
    # 固定一个月饼特征中心，只使用训练集月饼计算。
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images in train_loader:
            _, features = model(images.to(device))
            embeddings.append(features)
    center = torch.cat(embeddings).mean(0).detach()

    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for images in train_loader:
            images = images.to(device)
            reconstruction, features = model(images)
            loss = mse(reconstruction, images) + .02 * ((features - center) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
            count += len(images)

        scores = validation_scores(model, val_loader, center, device)
        val_score = sum(scores) / len(scores)
        checkpoint = {
            "state_dict": model.state_dict(),
            "center": center.cpu(),
            "epoch": epoch,
            "val_score": val_score,
            "image_size": 64,
        }
        torch.save(checkpoint, args.output / f"epoch_{epoch:02d}.pt")
        print(
            f"第 {epoch:02d} 轮 | 训练 Loss {total_loss/count:.4f} | "
            f"月饼验证误差 {val_score:.4f}",
            flush=True,
        )
        if val_score < best - 1e-5:
            best = val_score
            stale = 0
            torch.save(checkpoint, args.output / "best.pt")
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"验证误差连续 {args.patience} 轮未改善，提前停止。")
                break
    print(f"最佳模型保存在：{args.output / 'best.pt'}")
    print("请用 predict.py --checkpoint 指向刚训练好的模型。")


if __name__ == "__main__":
    main()
