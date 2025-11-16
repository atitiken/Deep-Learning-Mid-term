

import os
import gc
import copy
import json
import hashlib
import random
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

STUDENT_ID = "2702220422"
STUDENT_NAME = "Aeriel Nathen"

LATENT_DIM = 100
IMG_CHANNELS = 3
IMG_SIZE = 32
GEN_FEATURES = 64
DISC_FEATURES = 64
G_LR = 0.0002
D_LR = 0.0002
BATCH_SIZE = 32  # Smaller batch for more stable GAN training and frequent updates
REAL_LABEL = 1.0
FAKE_LABEL = 0.0
BETA1 = 0.5
BETA2 = 0.999
GAN_EPOCHS = 3000
SAVE_INTERVAL = 50  # Save less frequently for 3000 epoch training

CLS_EPOCHS = 25
CLS_LR = 2e-4
CLS_WEIGHT_DECAY = 1e-2  # Higher weight decay for AdamW
NUM_GENERATED = 500
VAL_SPLIT = 0.1
COMPARISON_COUNT = 10

GAN_MEAN = (0.5, 0.5, 0.5)
GAN_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR100_CLASSES = {
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
    5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
    10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
    15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
    20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
    25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
    30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
    35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
    40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
    45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
    50: "mouse", 51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid",
    55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree",
    60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
    65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
    70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
    75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
    80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table",
    85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
    90: "train", 91: "trout", 92: "tulip", 93: "turtle", 94: "wardrobe",
    95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm",
}


class TrainingLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.log_path, 'w') as f:
            f.write(f"Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message: str, also_print: bool = True):
        with open(self.log_path, 'a') as f:
            f.write(message + "\n")
        if also_print:
            print(message)
    
    def log_section(self, title: str):
        separator = "="*60
        self.log(f"\n{separator}")
        self.log(title)
        self.log(f"{separator}\n")


def setup_reproducibility(student_id: str):
    hash_value = int(hashlib.md5(student_id.encode("utf-8")).hexdigest(), 16)
    class_id = hash_value % 100
    seed_value = hash_value % (2**31)
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return class_id, seed_value


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(GAN_MEAN).view(-1, 1, 1)
    std = torch.tensor(GAN_STD).view(-1, 1, 1)
    return x * std + mean


def prepare_data(class_id: int, data_root: Path, seed: int, val_split: float = VAL_SPLIT, logger=None):
    log_fn = logger.log if logger else print
    
    log_fn(f"\n{'='*60}")
    log_fn("DATA PREPARATION")
    log_fn(f"{'='*60}")

    aug_transforms = [transforms.RandomHorizontalFlip()]
    if IMG_SIZE == 32:
        aug_transforms.append(transforms.RandomCrop(IMG_SIZE, padding=4, padding_mode='reflect'))
    else:
        aug_transforms.append(transforms.Resize(IMG_SIZE))
    aug_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(GAN_MEAN, GAN_STD),
    ])

    train_transform = transforms.Compose(aug_transforms)
    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(GAN_MEAN, GAN_STD),
    ])

    data_root.mkdir(exist_ok=True)
    train_full = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
    train_eval = datasets.CIFAR100(root=data_root, train=True, download=False, transform=eval_transform)
    test_full = datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_transform)

    log_fn(f"Filtering class: {CIFAR100_CLASSES[class_id]} (ID: {class_id})")

    train_indices = [idx for idx, label in enumerate(train_full.targets) if label == class_id]
    test_indices = [idx for idx, label in enumerate(test_full.targets) if label == class_id]

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_indices)
    val_count = max(1, int(len(shuffled) * val_split))
    val_indices = shuffled[:val_count].tolist()
    train_split_indices = shuffled[val_count:].tolist()

    turt_train = Subset(train_full, train_split_indices)
    turt_val = Subset(train_eval, val_indices)
    turt_eval = Subset(train_eval, train_indices)
    turt_test = Subset(test_full, test_indices)

    log_fn(f"train samples: {len(train_split_indices)}")
    log_fn(f"val samples: {len(val_indices)}")
    log_fn(f"test samples: {len(test_indices)}")

    clear_memory()

    return turt_train, turt_val, turt_test, turt_eval


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1 -> GEN_FEATURES*8 x 4 x 4
            nn.ConvTranspose2d(LATENT_DIM, GEN_FEATURES * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FEATURES * 8),
            nn.ReLU(True),
            # GEN_FEATURES*8 x 4 x 4 -> GEN_FEATURES*4 x 8 x 8
            nn.ConvTranspose2d(GEN_FEATURES * 8, GEN_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURES * 4),
            nn.ReLU(True),
            # GEN_FEATURES*4 x 8 x 8 -> GEN_FEATURES*2 x 16 x 16
            nn.ConvTranspose2d(GEN_FEATURES * 4, GEN_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURES * 2),
            nn.ReLU(True),
            # GEN_FEATURES*2 x 16 x 16 -> 3 x 32 x 32
            nn.ConvTranspose2d(GEN_FEATURES * 2, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Input: 3 x 32 x 32 -> DISC_FEATURES x 16 x 16
            nn.Conv2d(IMG_CHANNELS, DISC_FEATURES, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # DISC_FEATURES x 16 x 16 -> DISC_FEATURES*2 x 8 x 8
            nn.Conv2d(DISC_FEATURES, DISC_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURES * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # DISC_FEATURES*2 x 8 x 8 -> DISC_FEATURES*4 x 4 x 4
            nn.Conv2d(DISC_FEATURES * 2, DISC_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURES * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # DISC_FEATURES*4 x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(DISC_FEATURES * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img):
        return self.main(img).view(-1)


def evaluate_gan_losses(generator, discriminator, loader, device, criterion):
    generator.eval()
    discriminator.eval()
    g_losses, d_losses = [], []
    d_x_scores, d_g_z_scores = [], []
    
    with torch.no_grad():
        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            label_real = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), FAKE_LABEL, dtype=torch.float, device=device)
            
            real_output = discriminator(real_imgs).view(-1)
            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            fake = generator(noise)
            fake_output = discriminator(fake).view(-1)
            
            d_loss_real = criterion(real_output, label_real)
            d_loss_fake = criterion(fake_output, label_fake)
            d_loss = d_loss_real + d_loss_fake
            g_loss = criterion(fake_output, label_real)
            
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            d_x_scores.append(torch.sigmoid(real_output).mean().item())
            d_g_z_scores.append(torch.sigmoid(fake_output).mean().item())
    
    generator.train()
    discriminator.train()
    
    return {
        'g_loss': float(np.mean(g_losses)) if g_losses else 0.0,
        'd_loss': float(np.mean(d_losses)) if d_losses else 0.0,
        'd_x': float(np.mean(d_x_scores)) if d_x_scores else 0.0,
        'd_g_z': float(np.mean(d_g_z_scores)) if d_g_z_scores else 0.0,
    }


def summarize_trend(values):
    if len(values) < 2:
        return "insufficient data"
    delta = values[-1] - values[0]
    direction = "decreased" if delta < 0 else "increased"
    magnitude = abs(delta)
    return f"{direction} by {magnitude:.3f} from {values[0]:.3f} to {values[-1]:.3f}"


def save_gan_analysis(train_losses, val_losses, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    analysis = []
    if train_losses:
        analysis.append(f"Generator train loss {summarize_trend(train_losses)}")
    if val_losses:
        analysis.append(f"Generator val loss {summarize_trend(val_losses)}")
    path.write_text("\n".join(analysis))


def save_comparison_grid(real_dataset, fake_images, outputs_dir: Path, count: int = COMPARISON_COUNT):
    figure_path = outputs_dir / "figures" / "real_vs_fake.png"
    (outputs_dir / "figures").mkdir(exist_ok=True, parents=True)
    
    # Collect real images
    real_tensors = []
    for i in range(min(count, len(real_dataset))):
        tensor = real_dataset[i][0]
        real_tensors.append(denormalize(tensor).clamp(0, 1))
    
    # Denormalize fake images
    fake_tensors = [denormalize(img).clamp(0, 1) for img in fake_images[:count]]
    
    cols = min(count, len(real_tensors), len(fake_tensors))
    if cols == 0:
        return figure_path
    
    # Create comparison grid
    plt.figure(figsize=(2 * cols, 4))
    for idx in range(cols):
        # Real images in top row
        plt.subplot(2, cols, idx + 1)
        plt.imshow(real_tensors[idx].permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Real {idx+1}")
        # Fake images in bottom row
        plt.subplot(2, cols, cols + idx + 1)
        plt.imshow(fake_tensors[idx].permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Fake {idx+1}")
    
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return figure_path


def save_classifier_analysis(train_losses, val_losses, outputs_dir: Path):
    """Save classifier training analysis"""
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    analysis_path = figures_dir / "classifier_training_analysis.txt"
    text = [
        f"Classifier train loss {summarize_trend(train_losses)}" if train_losses else "",
        f"Classifier val loss {summarize_trend(val_losses)}" if val_losses else "",
    ]
    analysis_path.write_text("\n".join([line for line in text if line]))
    return analysis_path


def train_gan(turt_train, turt_val, device, outputs_dir, resume_from=None, epochs: int = GAN_EPOCHS,
              g_lr: float = G_LR, d_lr: float = D_LR, beta1: float = BETA1, beta2: float = BETA2,
              log_prefix: str = "", logger=None, use_amp: bool = False):
    """Train DCGAN with AdamW optimizer and optional mixed precision
    
    Args:
        use_amp: Set to False to use full FP32 precision for better GAN stability
    """
    log_fn = logger.log if logger else print
    
    log_fn(f"\n{'='*60}")
    log_fn(f"{log_prefix}DCGAN TRAINING WITH ADAMW")
    log_fn(f"{'='*60}")
    log_fn(f"Device: {device}")
    log_fn(f"Architecture: Deep Convolutional GAN")
    log_fn(f"Loss: Binary Cross-Entropy with Logits")
    log_fn(f"Optimizer: AdamW")
    log_fn(f"Precision: {'Mixed (FP16/FP32)' if use_amp else 'Full (FP32)'}")
    log_fn(f"Epochs: {epochs}")
    log_fn(f"Batch size: {BATCH_SIZE}")
    log_fn(f"G_LR: {g_lr}, D_LR: {d_lr}")
    log_fn(f"Beta1: {beta1}, Beta2: {beta2}")
    log_fn(f"{'='*60}\n")
    
    figures_dir = outputs_dir / "figures"
    checkpoints_dir = outputs_dir / "checkpoints"
    figures_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    criterion = nn.BCEWithLogitsLoss()
    
    opt_G = optim.AdamW(generator.parameters(), lr=g_lr, betas=(beta1, beta2), weight_decay=1e-4)
    opt_D = optim.AdamW(discriminator.parameters(), lr=d_lr, betas=(beta1, beta2), weight_decay=1e-4)
    

    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and use_amp) else None
    if scaler:
        log_fn("Using automatic mixed precision (AMP)")
    else:
        log_fn("Using full FP32 precision for maximum stability")
    
    turt_loader = DataLoader(
        turt_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )
    turt_val_loader = DataLoader(
        turt_val, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    G_losses, D_losses, D_x_scores, D_G_z_scores = [], [], [], []
    G_val_losses, D_val_losses = [], []
    start_epoch = 1
    
    if resume_from and resume_from.exists():
        log_fn(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        G_losses = checkpoint['g_losses']
        D_losses = checkpoint['d_losses']
        D_x_scores = checkpoint.get('d_x_scores', [])
        D_G_z_scores = checkpoint.get('d_g_z_scores', [])
        G_val_losses = checkpoint.get('g_val_losses', [])
        D_val_losses = checkpoint.get('d_val_losses', [])
        start_epoch = checkpoint['epoch'] + 1
        log_fn(f"Resumed from epoch {checkpoint['epoch']}")
    
    log_fn("Starting Training Loop...")
    
    LOG_INTERVAL = 100  
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_g_losses, epoch_d_losses = [], []
        epoch_d_x, epoch_d_g_z = [], []
        
        pbar = tqdm(turt_loader, desc=f"Epoch {epoch}/{epochs}", ncols=120)
        
        for i, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(device, non_blocking=True)  
            b_size = real_imgs.size(0)
            label_real = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), FAKE_LABEL, dtype=torch.float, device=device)
            

            opt_D.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    real_output = discriminator(real_imgs).view(-1)
                    errD_real = criterion(real_output, label_real)
                    noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
                    fake = generator(noise)
                    fake_output = discriminator(fake.detach()).view(-1)
                    errD_fake = criterion(fake_output, label_fake)
                    errD = errD_real + errD_fake
                scaler.scale(errD).backward()
                scaler.step(opt_D)
                scaler.update()
            else:
                real_output = discriminator(real_imgs).view(-1)
                errD_real = criterion(real_output, label_real)
                noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
                fake = generator(noise)
                fake_output = discriminator(fake.detach()).view(-1)
                errD_fake = criterion(fake_output, label_fake)
                errD = errD_real + errD_fake
                errD.backward()
                opt_D.step()
            
            D_x = torch.sigmoid(real_output).mean().item()
            D_G_z1 = torch.sigmoid(fake_output).mean().item()
            
            # Train Generator with AMP
            opt_G.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
                    fake = generator(noise)
                    fake_output = discriminator(fake).view(-1)
                    errG = criterion(fake_output, label_real)
                scaler.scale(errG).backward()
                scaler.step(opt_G)
                scaler.update()
            else:
                noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
                fake = generator(noise)
                fake_output = discriminator(fake).view(-1)
                errG = criterion(fake_output, label_real)
                errG.backward()
                opt_G.step()
            
            D_G_z2 = torch.sigmoid(fake_output).mean().item()
            
            epoch_g_losses.append(errG.item())
            epoch_d_losses.append(errD.item())
            epoch_d_x.append(D_x)
            epoch_d_g_z.append(D_G_z2)
            
            # Log more frequently with smaller batches to monitor training closely
            if i % LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'D_loss': f'{errD.item():.3f}',
                    'G_loss': f'{errG.item():.3f}',
                    'D(x)': f'{D_x:.3f}',
                    'D(G(z))': f'{D_G_z2:.3f}'
                })
        
        # Calculate epoch statistics
        avg_g_loss = np.mean(epoch_g_losses)
        avg_d_loss = np.mean(epoch_d_losses)
        avg_d_x = np.mean(epoch_d_x)
        avg_d_g_z = np.mean(epoch_d_g_z)
        
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)
        D_x_scores.append(avg_d_x)
        D_G_z_scores.append(avg_d_g_z)
        
        # Evaluate on validation set (less frequently to save time)
        # Only validate every 10 epochs to reduce overhead
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            val_metrics = evaluate_gan_losses(generator, discriminator, turt_val_loader, device, criterion)
            G_val_losses.append(val_metrics['g_loss'])
            D_val_losses.append(val_metrics['d_loss'])
        else:
            # Append last validation loss to maintain list length
            if G_val_losses:
                G_val_losses.append(G_val_losses[-1])
                D_val_losses.append(D_val_losses[-1])
        
        # Save checkpoints and visualizations periodically
        if epoch % SAVE_INTERVAL == 0 or epoch == 1 or epoch == epochs:
            log_msg = (f"\n{log_prefix}Epoch [{epoch}/{epochs}] "
                      f"D_loss: {avg_d_loss:.4f} (val {D_val_losses[-1]:.4f}) "
                      f"G_loss: {avg_g_loss:.4f} (val {G_val_losses[-1]:.4f}) "
                      f"D(x): {avg_d_x:.3f} D(G(z)): {avg_d_g_z:.3f}")
            log_fn(log_msg)
            
            # Generate sample images
            with torch.no_grad():
                generator.eval()
                fake_samples = generator(fixed_noise).cpu()
                generator.train()
            
            fake_samples = (fake_samples + 1) / 2
            grid = vutils.make_grid(fake_samples, nrow=8, normalize=True)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
            plt.axis("off")
            plt.title(f"Epoch {epoch}")
            plt.savefig(figures_dir / f"samples_epoch_{epoch:04d}.png", dpi=150, bbox_inches="tight")
            plt.close()
            
            # Save checkpoint with optimized I/O
            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            checkpoint_dict = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(),
                'opt_D_state_dict': opt_D.state_dict(),
                'g_losses': G_losses,
                'd_losses': D_losses,
                'd_x_scores': D_x_scores,
                'd_g_z_scores': D_G_z_scores,
                'g_val_losses': G_val_losses,
                'd_val_losses': D_val_losses,
            }
            if scaler:
                checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
            
            # Use torch.save with optimal settings for NVMe
            torch.save(checkpoint_dict, checkpoint_path, _use_new_zipfile_serialization=True)
            
            # More aggressive checkpoint cleanup - keep only last 3 and every 500th
            if epoch > SAVE_INTERVAL and epoch % 500 != 0:
                # Delete checkpoints older than 3 save intervals
                for old_epoch in range(epoch - SAVE_INTERVAL * 3, epoch - SAVE_INTERVAL, SAVE_INTERVAL):
                    if old_epoch > 0 and old_epoch % 500 != 0:
                        old_checkpoint = checkpoints_dir / f"checkpoint_epoch_{old_epoch:04d}.pt"
                        if old_checkpoint.exists():
                            old_checkpoint.unlink()
            
            # Clear cache less frequently to reduce overhead
            if epoch % 100 == 0:
                clear_memory()
    
    log_fn("\nGenerating training plots...")
    
    # Create comprehensive training visualization
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(G_losses, label="G Train", alpha=0.8, linewidth=1.5)
    plt.plot(D_losses, label="D Train", alpha=0.8, linewidth=1.5)
    if G_val_losses:
        plt.plot(G_val_losses, label="G Val", linestyle='--')
    if D_val_losses:
        plt.plot(D_val_losses, label="D Val", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Discriminator scores
    plt.subplot(2, 2, 2)
    plt.plot(D_x_scores, label="D(x) - Real", alpha=0.8, linewidth=1.5)
    plt.plot(D_G_z_scores, label="D(G(z)) - Fake", alpha=0.8, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Discriminator Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Moving average
    plt.subplot(2, 2, 3)
    window = 50
    if len(G_losses) >= window:
        g_ma = np.convolve(G_losses, np.ones(window)/window, mode='valid')
        d_ma = np.convolve(D_losses, np.ones(window)/window, mode='valid')
        plt.plot(g_ma, label="G Loss (MA)", alpha=0.8)
        plt.plot(d_ma, label="D Loss (MA)", alpha=0.8)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Moving Average (window={window})")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Score gap
    plt.subplot(2, 2, 4)
    diff_scores = [abs(x - y) for x, y in zip(D_x_scores, D_G_z_scores)]
    plt.plot(diff_scores, label="D(x) - D(G(z))", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Difference")
    plt.title("Score Gap")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "training_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save final model
    final_dict = {
        'epoch': epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_losses': G_losses,
        'd_losses': D_losses,
        'd_x_scores': D_x_scores,
        'd_g_z_scores': D_G_z_scores,
        'g_val_losses': G_val_losses,
        'd_val_losses': D_val_losses,
    }
    torch.save(final_dict, checkpoints_dir / "final_model.pt")
    save_gan_analysis(G_losses, G_val_losses, outputs_dir / "figures" / "gan_training_analysis.txt")
    
    log_fn(f"\nTraining complete. Models saved to {checkpoints_dir}")
    return generator, discriminator


def generate_fake_images(generator, device, outputs_dir, count=512, logger=None):
    """Generate synthetic images using trained generator"""
    log_fn = logger.log if logger else print
    
    log_fn(f"\n{'='*60}")
    log_fn(f"GENERATING {count} IMAGES")
    log_fn(f"{'='*60}")
    
    generated_dir = outputs_dir / "generated"
    generated_dir.mkdir(exist_ok=True, parents=True)
    
    generator.eval()
    all_samples = []
    
    with torch.no_grad():
        for i in tqdm(range(0, count, BATCH_SIZE), desc="Generating"):
            batch_count = min(BATCH_SIZE, count - i)
            z = torch.randn(batch_count, LATENT_DIM, 1, 1, device=device)
            imgs = generator(z).cpu()
            all_samples.append(imgs)
    
    synth_images = torch.cat(all_samples, dim=0)[:count]
    
    # Save generated images as numpy arrays
    for idx, tensor_img in enumerate(synth_images):
        np.save(generated_dir / f"fake_{idx:04d}.npy", tensor_img.numpy())
    
    log_fn(f"Saved {len(synth_images)} images to {generated_dir}")
    clear_memory()
    
    return synth_images


class FakeOriginalClassifier(nn.Module):
    """EfficientNet-B0 based binary classifier for real vs fake images"""
    def __init__(self, pretrained=True):
        super().__init__()
        # Use EfficientNet-B0 instead of ResNet-18
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        
        self.backbone = models.efficientnet_b0(weights=weights)
        # Replace classifier head for binary classification
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.backbone(x)


def prepare_classification_data(turt_dataset_eval, synth_images, logger=None):
    """Prepare real and fake images for binary classification"""
    log_fn = logger.log if logger else print
    
    log_fn(f"\n{'='*60}")
    log_fn("PREPARING CLASSIFICATION DATASET")
    log_fn(f"{'='*60}")
    
    # Denormalize images for proper processing
    real_imgs = torch.stack([denormalize(turt_dataset_eval[i][0]) for i in range(len(turt_dataset_eval))])
    fake_imgs = torch.stack([denormalize(img) for img in synth_images])
    
    # Convert to numpy arrays
    real_np = real_imgs.permute(0, 2, 3, 1).numpy()
    fake_np = fake_imgs.permute(0, 2, 3, 1).numpy()
    
    # Combine and create labels (0=real, 1=fake)
    X = np.concatenate([real_np, fake_np], axis=0)
    y = np.concatenate([np.zeros(len(real_np)), np.ones(len(fake_np))], axis=0).astype(np.int64)
    
    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Normalize with ImageNet statistics for EfficientNet
    imagenet_norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def to_dataset(images, labels):
        tensors = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        for i in range(len(tensors)):
            tensors[i] = imagenet_norm(tensors[i])
        return TensorDataset(tensors, torch.from_numpy(labels))
    
    train_ds = to_dataset(X_train, y_train)
    val_ds = to_dataset(X_val, y_val)
    test_ds = to_dataset(X_test, y_test)
    
    log_fn(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    # Clean up memory
    del real_imgs, fake_imgs, real_np, fake_np, X, y, X_train, X_temp, X_val, X_test
    clear_memory()
    
    return train_ds, val_ds, test_ds


def train_classifier(train_ds, val_ds, device, outputs_dir, epochs: int = CLS_EPOCHS,
                     lr: float = CLS_LR, weight_decay: float = CLS_WEIGHT_DECAY,
                     log_prefix: str = "", save_artifacts: bool = True, logger=None, use_amp: bool = False):
    """Train EfficientNet-B0 classifier with AdamW and optional mixed precision
    
    Args:
        use_amp: Set to False to use full FP32 precision
    """
    log_fn = logger.log if logger else print
    
    log_fn(f"\n{'='*60}")
    log_fn("TRAINING EFFICIENTNET-B0 CLASSIFIER WITH ADAMW")
    log_fn(f"{'='*60}")
    log_fn(f"Optimizer: AdamW")
    log_fn(f"Precision: {'Mixed (FP16/FP32)' if use_amp else 'Full (FP32)'}")
    log_fn(f"Learning rate: {lr}")
    log_fn(f"Weight decay: {weight_decay}")
    log_fn(f"Epochs: {epochs}")
    log_fn(f"{'='*60}\n")
    
    # Adjusted batch size for classifier with FP32
    train_loader = DataLoader(
        train_ds, 
        batch_size=64,  # Moderate batch size works well for classifier
        shuffle=True, 
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Initialize EfficientNet-B0 classifier
    model = FakeOriginalClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
    # Setup automatic mixed precision only if requested
    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and use_amp) else None
    
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"{log_prefix}Epoch {epoch}/{epochs}", ncols=100)
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Training with AMP
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)
                else:
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * len(X_batch)
        
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        log_msg = f"{log_prefix}Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}"
        log_fn(log_msg)
        clear_memory()
    
    if save_artifacts:
        figures_dir = outputs_dir / "figures"
        figures_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot training curves
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("EfficientNet-B0 Classifier Training")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(figures_dir / "classifier_losses.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Save model
        torch.save(model.state_dict(), outputs_dir / "classifier_model.pt")
        log_fn(f"Classifier saved to {outputs_dir / 'classifier_model.pt'}")
    
    return model, train_losses, val_losses


def evaluate_classifier(model, test_ds, device, outputs_dir, logger=None, use_amp: bool = False):
    """Evaluate classifier on test set and generate metrics
    
    Args:
        use_amp: Set to False to use full FP32 precision
    """
    log_fn = logger.log if logger else print
    
    log_fn(f"\n{'='*60}")
    log_fn("EVALUATING CLASSIFIER")
    log_fn(f"{'='*60}")
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    
    model.eval()
    all_preds, all_labels = [], []
    
    # Setup AMP for evaluation only if requested
    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and use_amp) else None
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Testing", ncols=100):
            X_batch = X_batch.to(device)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(X_batch)
            else:
                logits = model(X_batch)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Log results
    log_fn(f"\nTest Metrics:")
    log_fn(f"Accuracy:  {acc:.4f}")
    log_fn(f"Precision: {precision:.4f}")
    log_fn(f"Recall:    {recall:.4f}")
    log_fn(f"F1 Score:  {f1:.4f}")
    log_fn(f"\nConfusion Matrix:")
    log_fn(str(cm))
    
    # Plot confusion matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - EfficientNet-B0")
    plt.tight_layout()
    plt.savefig(outputs_dir / "figures" / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save metrics to file
    with open(outputs_dir / "test_metrics.txt", "w") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"{cm}\n")
    
    log_fn(f"\nMetrics saved to {outputs_dir / 'test_metrics.txt'}")


def main():
    parser = argparse.ArgumentParser(description="DCGAN Training with AdamW, EfficientNet-B0")
    parser.add_argument("--skip-gan", action="store_true", help="Skip GAN training")
    parser.add_argument("--skip-classifier", action="store_true", help="Skip classifier training")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=GAN_EPOCHS, help="Number of epochs")
    parser.add_argument("--class-id", type=int, help="Override CIFAR-100 class ID")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT, help="Validation split fraction")
    parser.add_argument("--comparison-count", type=int, default=COMPARISON_COUNT, help="Images per row in comparison")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision (FP16). Default is FP32 for GAN stability")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("COMP6826001 - PROBLEM 2: DCGAN (ENHANCED)")
    print(f"{'='*60}")
    print(f"Student: {STUDENT_NAME}")
    print(f"ID: {STUDENT_ID}")
    print(f"Enhancements: AdamW + EfficientNet-B0 + {'AMP' if args.use_amp else 'FP32'} + Logging")
    print(f"Batch Size: {BATCH_SIZE} (optimized for GAN stability)")
    print(f"{'='*60}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup reproducibility
    class_id, seed = setup_reproducibility(STUDENT_ID)
    if args.class_id is not None:
        class_id = args.class_id % 100
        print(f"Overriding class ID via CLI: {class_id}")
    print(f"Class: {CIFAR100_CLASSES[class_id]} (ID: {class_id})")
    print(f"Seed: {seed}\n")
    
    # Setup output directories
    outputs_dir = Path("problem2_outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Initialize logger
    log_path = outputs_dir / "training_log.txt"
    logger = TrainingLogger(log_path)
    logger.log(f"Student: {STUDENT_NAME} ({STUDENT_ID})")
    logger.log(f"Class: {CIFAR100_CLASSES[class_id]} (ID: {class_id})")
    logger.log(f"Seed: {seed}")
    logger.log(f"Device: {device}")
    logger.log(f"Batch Size: {BATCH_SIZE}")
    logger.log(f"Precision: {'Mixed (FP16/FP32)' if args.use_amp else 'Full (FP32) for maximum stability'}")
    logger.log(f"Enhancements: AdamW optimizer, EfficientNet-B0")
    
    # Prepare data
    data_root = Path("cifar100-cache")
    turt_train, turt_val, turt_test, turt_eval = prepare_data(class_id, data_root, seed, 
                                                                val_split=args.val_split, logger=logger)
    
    # Train GAN
    if not args.skip_gan:
        resume_path = Path(args.resume) if args.resume else None
        generator, discriminator = train_gan(
            turt_train, turt_val, device, outputs_dir,
            resume_from=resume_path, epochs=args.epochs, logger=logger, use_amp=args.use_amp)
    else:
        logger.log("Loading GAN from checkpoint...")
        generator = Generator().to(device)
        checkpoint = torch.load(outputs_dir / "checkpoints" / "final_model.pt", 
                              map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator = None
    
    # Generate fake images
    synth_images = generate_fake_images(generator, device, outputs_dir, 
                                        count=NUM_GENERATED, logger=logger)
    comparison_path = save_comparison_grid(turt_eval, synth_images, outputs_dir, 
                                          count=args.comparison_count)
    logger.log(f"Saved real vs fake comparison grid to {comparison_path}")
    
    # Train classifier
    if not args.skip_classifier:
        train_ds, val_ds, test_ds = prepare_classification_data(turt_eval, synth_images, logger=logger)
        
        logger.log(f"\nClassifier hyperparameters -> lr: {CLS_LR:.2e}, weight_decay: {CLS_WEIGHT_DECAY:.2e}")
        classifier, cls_train_losses, cls_val_losses = train_classifier(
            train_ds, val_ds, device, outputs_dir,
            epochs=CLS_EPOCHS, lr=CLS_LR, weight_decay=CLS_WEIGHT_DECAY, logger=logger, use_amp=args.use_amp)
        
        cls_analysis = save_classifier_analysis(cls_train_losses, cls_val_losses, outputs_dir)
        logger.log(f"Saved classifier training analysis to {cls_analysis}")
        
        evaluate_classifier(classifier, test_ds, device, outputs_dir, logger=logger, use_amp=args.use_amp)
    
    # Final summary
    logger.log_section("COMPLETED")
    logger.log(f"All outputs saved to: {outputs_dir}")
    logger.log(f"Training log: {log_path}")
    logger.log("="*60)
    
    print(f"\n{'='*60}")
    print("COMPLETED")
    print(f"{'='*60}")
    print(f"Outputs: {outputs_dir}")
    print(f"Training log: {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()