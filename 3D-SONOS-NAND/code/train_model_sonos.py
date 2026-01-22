import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from physicsnemo.models.fno import FNO
from sonos_dataset import SonosDataset

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset
    # Resolution (64, 128) -> (R, X)
    full_ds = SonosDataset(data_dir=args.data_dir, resolution=(64, 128))
    if len(full_ds) == 0: return

    # Save stats
    torch.save(full_ds.get_stats(), os.path.join(args.out_dir, "stats.pt"))

    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Model
    # in_channels=6 (Spacing, TL, PGM, Time, X_grid, R_grid)
    model = FNO(
        in_channels=6, 
        out_channels=1, 
        decoder_layers=1, 
        decoder_layer_size=32, 
        dimension=2,
        latent_channels=32,
        num_fno_layers=4,
        num_fno_modes=12,
        padding=8
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader))
    criterion = nn.MSELoss()

    print("Starting training...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))

    print("Training Complete.")

if __name__ == "__main__":
    train()