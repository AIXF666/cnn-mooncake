"""Train the one-target mooncake/not-mooncake CNN from random initialization."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mooncake_cnn import ResidualMooncakeCNN, preprocess_image

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 192
NEGATIVE_WEIGHT = .5


def split_positive_sources(ai_paths, real_paths):
    ai_paths, real_paths = list(ai_paths), list(real_paths)
    random.Random(7).shuffle(ai_paths)
    random.Random(11).shuffle(real_paths)
    ai_cut = int(.8 * len(ai_paths))
    real_cut = int(.8 * len(real_paths))
    ai_train, ai_holdout = ai_paths[:ai_cut], ai_paths[ai_cut:]
    real_train, real_holdout = real_paths[:real_cut], real_paths[real_cut:]
    random.Random(23).shuffle(ai_holdout)
    random.Random(19).shuffle(real_holdout)
    ai_mid, real_mid = len(ai_holdout) // 2, len(real_holdout) // 2
    return ({'ai': ai_train, 'real': real_train},
            {'ai': ai_holdout[:ai_mid], 'real': real_holdout[:real_mid]},
            {'ai': ai_holdout[ai_mid:], 'real': real_holdout[real_mid:]})


def auc_score(positive, negative):
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0).mean() + .5 * (delta == 0).mean()))


def select_threshold(pos_scores, neg_scores):
    candidates = np.unique(np.concatenate([np.array([0.]), *pos_scores.values(), *neg_scores.values(), np.array([1.])]))
    best_threshold, best_score = .5, -1.
    for threshold in candidates:
        recalls = [(pos_scores[src] >= threshold).mean() for src in ('ai', 'real')]
        specificities = [(neg_scores[src] < threshold).mean() for src in ('ai', 'commons')]
        balanced = .5 * (sum(recalls) / len(recalls) + sum(specificities) / len(specificities))
        if balanced > best_score:
            best_threshold, best_score = float(threshold), float(balanced)
    return best_threshold, best_score


def balanced_aug(x):
    x = x.clone()
    flip = torch.rand(len(x), device=x.device) < .5
    x[flip] = x[flip].flip(-1)
    contrast = .8 + .4 * torch.rand((len(x), 1, 1, 1), device=x.device)
    brightness = -.10 + .20 * torch.rand((len(x), 1, 1, 1), device=x.device)
    x = ((x - .5) * contrast + .5 + brightness).clamp(0, 1) * 2 - 1
    angle = (torch.rand(len(x), device=x.device) - .5) * .40
    c, s = angle.cos(), angle.sin()
    scale = (c.abs() + s.abs()) * (1.04 + .06 * torch.rand(len(x), device=x.device))
    theta = torch.zeros((len(x), 2, 3), device=device)
    theta[:, 0, 0], theta[:, 0, 1] = scale * c, -scale * s
    theta[:, 1, 0], theta[:, 1, 1] = scale * s, scale * c
    theta[:, :, 2] = (torch.rand((len(x), 2), device=device) - .5) * .03
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')


def stack_images(paths):
    return torch.stack([preprocess_image(path, IMAGE_SIZE) for path in paths])


def predict(model, tensor):
    model.eval()
    scores = []
    with torch.inference_mode():
        for batch in tensor.split(32):
            scores.extend(torch.sigmoid(model((batch.to(device) * 2 - 1)).flatten()).cpu().tolist())
    return np.asarray(scores)


def main():
    global device
    parser = argparse.ArgumentParser(description='从零训练整图 CNN，输出“月饼/非月饼”。')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints/from-scratch')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--negative-weight', type=float, default=NEGATIVE_WEIGHT)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    args = parser.parse_args()
    mooncake = args.data / 'mooncake'
    ai_paths = sorted((mooncake / 'ai_original').glob('*.png'))
    real_paths = sorted(mooncake.glob('real_*.jpg'))
    neg_train_dir = args.data / 'not_mooncake' / 'train'
    neg_val_dir = args.data / 'not_mooncake' / 'validation'
    neg_train = sorted(neg_train_dir.glob('*'))
    neg_val = sorted(neg_val_dir.glob('*'))
    if len(ai_paths) < 100 or len(real_paths) < 20:
        parser.error('原始 AI 月饼图或真实月饼图未找到；先解压 v1.1.0 数据附件到 data/mooncake/ai_original。')
    if not neg_train or not neg_val:
        parser.error('未找到负样本；先从 data/not_mooncake/source_manifest.csv 对应的发布包准备训练和验证图片。')
    if any(path.suffix.lower() not in ('.jpg', '.jpeg', '.png') for path in neg_train + neg_val):
        parser.error('负样本目录里有非图像文件。')
    pos_train, pos_val, pos_test = split_positive_sources(ai_paths, real_paths)
    neg_train_gen = [p for p in neg_train if p.name.startswith('gen_')]
    neg_train_commons = [p for p in neg_train if p.name.startswith('commons_')]
    neg_val_gen = [p for p in neg_val if p.name.startswith('gen_')]
    neg_val_commons = [p for p in neg_val if p.name.startswith('commons_')]
    if min(map(len, (neg_train_gen, neg_train_commons, neg_val_gen, neg_val_commons))) == 0:
        parser.error('训练和验证负样本都须同时包括 AI 图与 Commons 图。')

    torch.manual_seed(20260930)
    random.seed(20260930)
    torch.set_num_threads(4)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device)
    tensors = {f'pos_train_{k}': stack_images(v) for k, v in pos_train.items()}
    tensors.update({f'pos_val_{k}': stack_images(v) for k, v in pos_val.items()})
    tensors.update({f'pos_test_{k}': stack_images(v) for k, v in pos_test.items()})
    tensors['neg_train_gen'] = stack_images(neg_train_gen)
    tensors['neg_train_commons'] = stack_images(neg_train_commons)
    tensors['neg_val_gen'] = stack_images(neg_val_gen)
    tensors['neg_val_commons'] = stack_images(neg_val_commons)
    model = ResidualMooncakeCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=3e-5)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    steps = (len(pos_train['ai']) + 15) // 16
    rng = random.Random(20260930)
    best_auc, best_epoch, stale = -1., 0, 0
    args.output.mkdir(parents=True, exist_ok=True)
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(steps):
            ai_i = rng.choices(range(len(tensors['pos_train_ai'])), k=16)
            real_i = rng.choices(range(len(tensors['pos_train_real'])), k=16)
            gen_i = rng.choices(range(len(tensors['neg_train_gen'])), k=8)
            com_i = rng.choices(range(len(tensors['neg_train_commons'])), k=8)
            pos = torch.cat((tensors['pos_train_ai'][ai_i], tensors['pos_train_real'][real_i]))
            neg = torch.cat((tensors['neg_train_gen'][gen_i], tensors['neg_train_commons'][com_i]))
            x = balanced_aug(torch.cat((pos, neg)).to(device))
            y = torch.cat((torch.ones(len(pos), device=device), torch.zeros(len(neg), device=device)))
            values = criterion(model(x).flatten(), y)
            loss = values[:len(pos)].mean() + args.negative_weight * values[len(pos):].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        pv = {k: predict(model, tensors[f'pos_val_{k}']) for k in ('ai', 'real')}
        nv = {'ai': predict(model, tensors['neg_val_gen']),
              'commons': predict(model, tensors['neg_val_commons'])}
        aucs = {'ai': auc_score(pv['ai'], nv['ai']), 'real': auc_score(pv['real'], nv['commons'])}
        macro_auc = sum(aucs.values()) / 2
        threshold, balanced = select_threshold(pv, nv)
        row = {'epoch': epoch, 'train_loss': sum(losses)/len(losses),
               'macro_validation_auc': macro_auc, 'validation_auc_by_source': aucs,
               'validation_threshold': threshold, 'validation_balanced_accuracy': balanced}
        log.append(row)
        if macro_auc > best_auc:
            best_auc, best_epoch, stale = macro_auc, epoch, 0
            torch.save({'state_dict': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'architecture': 'ResidualMooncakeCNN', 'pretrained': False,
                        'input_size': IMAGE_SIZE, 'threshold': threshold,
                        'negative_weight': args.negative_weight, **row}, args.output/'best.pt')
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        torch.save(model.state_dict(), args.output/f'epoch_{epoch:02d}.pt')
        (args.output/'history.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
        if epoch >= 20 and args.patience > 0 and stale >= args.patience:
            print(f'验证 AUC 连续 {args.patience} 轮未改善，最佳第 {best_epoch} 轮。', flush=True)
            break

    saved = torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'])
    pt = {k: predict(model, tensors[f'pos_test_{k}']) for k in ('ai', 'real')}
    report = {'epoch': saved['epoch'], 'threshold': saved['threshold'],
              'positive_holdout': {k:{'accepted':int((v>=saved['threshold']).sum()),'total':len(v)} for k,v in pt.items()},
              'negative_test':'No separate negative test set is bundled; verify using new images.'}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('最佳模型：', args.output/'best.pt', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
