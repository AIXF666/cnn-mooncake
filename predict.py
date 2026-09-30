"""Predict whether an image looks like a mooncake."""

import argparse
from pathlib import Path
import torch

from mooncake_cnn import ResidualMooncakeCNN, preprocess_image
from oneclass_model import OneClassCNN, anomaly_score, load_image

ROOT = Path(__file__).resolve().parent

def choose_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    return torch.device('cuda' if torch.cuda.is_available() else
                        ('mps' if torch.backends.mps.is_available() else 'cpu'))

def main():
    parser = argparse.ArgumentParser(description='整张图片分类：月饼 / 非月饼。')
    parser.add_argument('images', type=Path, nargs='+')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/best.pt')
    parser.add_argument('--threshold', type=float, default=None,
                        help='覆盖模型内置阈值；须在0到1之间。')
    parser.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='auto')
    args = parser.parse_args()
    for path in args.images:
        if not path.is_file():
            parser.error(f'图片不存在：{path}')
    if not args.checkpoint.is_file():
        parser.error(f'模型文件不存在：{args.checkpoint}')
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error('阈值必须在0到1之间。')

    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('architecture') == 'ResidualMooncakeCNN':
        model = ResidualMooncakeCNN().to(device)
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval()
        threshold = float(saved['threshold']) if args.threshold is None else args.threshold
        for path in args.images:
            image = (preprocess_image(path, int(saved.get('input_size', 192))) * 2 - 1).unsqueeze(0).to(device)
            with torch.inference_mode():
                score = float(torch.sigmoid(model(image)).item())
            label = '月饼' if score >= threshold else '非月饼'
            print(f'{path.name}：{label}；分数={score:.6f}；阈值={threshold:.6f}')
        print('分数用于分类，未校准为概率。')
        return

    # Keep compatibility with the earlier one-class example checkpoint.
    if 'center' in saved and 'val_score' in saved:
        model = OneClassCNN().to(device)
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval()
        threshold = float(saved['val_score']) * 1.5
        for path in args.images:
            image = load_image(path).unsqueeze(0).to(device)
            with torch.inference_mode():
                score = float(anomaly_score(model, image, saved['center'].to(device)).item())
            print(f'{path.name}：{"像月饼" if score <= threshold else "不像月饼"}；异常分数={score:.6f}；阈值={threshold:.6f}')
        print('旧版异常分数也不是概率。')
        return
    parser.error('无法识别此模型文件格式。')

if __name__ == '__main__':
    main()
