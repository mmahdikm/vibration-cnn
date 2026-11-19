"""Train the original CNN using synthetic signals or an explicit NPZ dataset."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import CNN1D

def synthetic_data(count, seed):
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 3
    time = np.linspace(0, 1, 1300, dtype=np.float32)
    signals = np.stack([
        np.stack([np.sin(2 * np.pi * (3 + 4 * label) * time + channel * .15)
                  for channel in range(6)]) for label in labels
    ]).astype(np.float32)
    signals += rng.normal(0, .05, signals.shape).astype(np.float32)
    return signals, labels.astype(np.int64)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='NPZ with x: (N,6,1300), y: (N,) in 0..2')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--size', choices=['small', 'large'], default='small')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 2 or args.batch_size < 2:
        parser.error('epochs >= 1, samples >= 2, and batch-size >= 2 are required')
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.data:
        with np.load(args.data, allow_pickle=False) as data:
            x, y = data['x'].astype(np.float32), data['y'].astype(np.int64)
    else:
        x, y = synthetic_data(args.samples, args.seed)
        print('Synthetic signals: this run checks the pipeline, not real-world accuracy.')
    if x.ndim != 3 or x.shape[1:] != (6, 1300) or y.shape != (len(x),):
        parser.error('expected x=(N,6,1300) and y=(N,)')
    if len(x) < 2 or not np.isfinite(x).all() or not np.isin(y, [0, 1, 2]).all():
        parser.error('data must have >=2 finite signals and labels in 0..2')
    dataset = TensorDataset(torch.from_numpy(x).unsqueeze(2), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = CNN1D(size=args.size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total = correct = seen = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        print(f'epoch={epoch + 1} loss={total / seen:.4f} training_accuracy={correct / seen:.3f}')
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)

if __name__ == '__main__':
    main()
