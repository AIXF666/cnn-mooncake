"""Run the included experimental one-class checkpoint on a single image."""

import argparse
from pathlib import Path

import torch

from oneclass_model import OneClassCNN, anomaly_score, load_image


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="使用附带的单类月饼实验模型。")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/oneclass/best.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"图片不存在：{args.image}")
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_file():
        parser.error(f"模型文件不存在：{checkpoint_path}")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device_name)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = OneClassCNN().to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    image = load_image(args.image).unsqueeze(0).to(device)
    center = saved["center"].to(device)
    threshold = float(saved["val_score"]) * 1.5
    with torch.no_grad():
        score = float(anomaly_score(model, image, center).item())
    print(f"图片：{args.image}")
    print(f"模型判断：{'像月饼' if score <= threshold else '不像月饼'}")
    print(f"异常分数：{score:.4f}；阈值：{threshold:.4f}")
    print("这是实验性相似度判断；分数不是概率，蛋糕等糕点可能误判。")


if __name__ == "__main__":
    main()
