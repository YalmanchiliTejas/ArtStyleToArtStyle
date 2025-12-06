import os
import glob
import argparse

import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F

from baseline_model import BaselineCycleGan
from modified_model import ImprovedCycleGan
import torch.nn as nn

from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, root, transform = None):


        super().__init__()

        self.dir_x = os.path.join(root, "trainA")
        self.dir_y = os.path.join(root, "trainB")

        self.sorted_paths_x = sorted(glob.glob(os.path.join(self.dir_x, "*")))
        self.sorted_paths_y = sorted(glob.glob(os.path.join(self.dir_y, "*")))


        self.transform = transform

        if len(self.sorted_paths_x) == 0 or len(self.sorted_paths_y) == 0:
            print(f"This is an issue here it's empty")
            return
        random.seed(42)
        random.shuffle(self.sorted_paths_x)
        random.shuffle(self.sorted_paths_y)

        max_per_domain = 500   # 500 A + 500 B = 1000 total
        self.sorted_paths_x = self.sorted_paths_x[:max_per_domain]
        self.sorted_paths_y = self.sorted_paths_y[:max_per_domain]
    
    def __len__(self):
        return max(len(self.sorted_paths_y), len(self.sorted_paths_x))
    

    def __getitem__(self, index):

        path_x = self.sorted_paths_x[index % len(self.sorted_paths_x)]
        path_y = self.sorted_paths_y[index % len(self.sorted_paths_y)]


        x_image = Image.open(path_x).convert("RGB")
        y_image = Image.open(path_y).convert("RGB")

        if self.transform:
            x_image = self.transform(x_image)
            y_image = self.transform(y_image)
        
        return {"X": x_image, "Y": y_image}



def save_checkpoint(model, G, D, epoch , out_dir):

    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    core = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "epoch":epoch,
        "G_xy": core.G_xy.state_dict(),
        "F_yx": core.F_yx.state_dict(),
        "D_x": core.D_x.state_dict(),
        "D_y": core.D_y.state_dict(),
        "G": G.state_dict(),
        "D": D.state_dict(),
    },
    path,
    )
    print(f"The checkpoint is saved to this path{path}", flush=True)
    return


def load_checkpoint(model, G, D, path, device="cpu"):

    if not os.path.isfile(path):
        print(f"this is the wrong checkpoint path{path}")
        return
    
    checkpoint = torch.load(path, map_location=device)
    core = model.module if isinstance(model, nn.DataParallel) else model
    core.G_xy.load_state_dict(checkpoint["G_xy"])
    core.F_yx.load_state_dict(checkpoint["F_yx"])
    core.D_x.load_state_dict(checkpoint["D_x"])
    core.D_y.load_state_dict(checkpoint["D_y"])

    if G is not None and "G" in checkpoint:
        G.load_state_dict(checkpoint["G"])
    if D is not None and "D" in checkpoint:
        D.load_state_dict(checkpoint["D"])

    epoch = checkpoint.get("epoch", 0)
    print(f"[Checkpoint] Loaded epoch {epoch}")
    return epoch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def save_samples(model, device, sample_batch, output, epoch, max_num):

    os.makedirs(output, exist_ok=True)

    model.eval()

    real_x = sample_batch["X"].to(device)
    real_y = sample_batch["Y"].to(device)

    real_x, real_y, fake_x , fake_y, rec_x , rec_y = model(real_x, real_y)
    visuals = torch.cat(
        [real_x, fake_y, rec_x, real_y, fake_x, rec_y],
        dim=0,
    )

    visuals = (visuals + 1.0) / 2.0
    visuals = F.interpolate(visuals, size=(256, 256), mode="bilinear", align_corners=False)
    save_path = os.path.join(output, f"epoch_{epoch:04d}.png")
  
    save_image(visuals, save_path, nrow=real_x.size(0))
    print(f"[Samples] Saved to {save_path}")
    return


def train(args):

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}")


    normalize = transforms.Compose(
        [
            transforms.Resize(args.load_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )
    dataset_root = os.path.join(args.root, args.dataset_name)
    training_dataset = ImageDataset(dataset_root, transform=normalize)

    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, pin_memory=True, drop_last=True)

    sample_batch = next(iter(train_loader))

    if args.model_type == "baseline":
        model = BaselineCycleGan(in_channels=3, out_channels=3, lambda_cycle=args.lambda_cycle, lambda_identity = args.lambda_identity)
    
    if args.model_type == "improved":
        model = ImprovedCycleGan( in_channels=3,out_channels=3,lambda_cycle=args.lambda_cycle,lambda_identity=args.lambda_identity,lambda_content=args.lambda_content,lambda_style=args.lambda_style,lambda_fm=args.lambda_fm,gan_mode=args.gan_mode)
    
    if torch.cuda.device_count() >= 2:
        print("Using", torch.cuda.device_count(), "GPUs with DataParallel")
        model = nn.DataParallel(model) 

    model.to(device)

    core = model.module if isinstance(model, nn.DataParallel) else model
    generator_params = list(core.G_xy.parameters()) + list(core.F_yx.parameters())
    discriminator_params = list(core.D_x.parameters()) + list(core.D_y.parameters())

    G = torch.optim.Adam(generator_params, lr=args.lr, betas=(args.beta1, args.beta2))
    D = torch.optim.Adam(discriminator_params, lr=args.lr , betas = (args.beta1, args.beta2))

    start_epoch = 1
    if args.resume_path is not None:
        last_epoch = load_checkpoint(core, G, D, args.resume_path, device=device)
        start_epoch = last_epoch + 1
    
        print(f"Training is resuming from this epoch{start_epoch}\n", flush=True)
    
    global_step = 0
    max_iter = 500
    for epoch in range(start_epoch, args.num_epochs + 1):

        model.train()
        for i , batch in enumerate(train_loader):

            if i >= max_iter:
                break
            real_x = batch["X"].to(device)
            real_y = batch["Y"].to(device)

            for p in core.D_x.parameters():
                p.requires_grad = False
            for p in core.D_y.parameters():
                p.requires_grad = False

            G.zero_grad()
            g_losses = core.compute_generator_loss(real_x, real_y)
            g_total = g_losses["loss_g_total"]
            g_total.backward()
            G.step()


            for p in core.D_x.parameters():
                p.requires_grad = True
            for p in core.D_y.parameters():
                p.requires_grad = True
            
            D.zero_grad()
            d_losses = core.compute_discriminator_loss(real_x, real_y)
            d_total = d_losses["loss_d_total"]
            d_total.backward()
            D.step()

            global_step += 1

            if ( i + 1) % 50 == 0:
                log_str = (
                    f"[Epoch {epoch}/{args.num_epochs}] "
                    f"[Iter {i+1}/{len(train_loader)}] "
                    f"D: {d_total.item():.4f}, G: {g_total.item():.4f}"
                )

                # For improved model, also log extra terms if they exist
                for key in [
                    "loss_g_adversial",
                    "loss_cycle",
                    "loss_identity",
                    "loss_content",
                    "loss_style",
                    "loss_fm",
                ]:
                    if key in g_losses:
                        log_str += f", {key}: {g_losses[key].item():.4f}"
                for key in ["loss_d_x", "loss_d_y"]:
                    if key in d_losses:
                        log_str += f", {key}: {d_losses[key].item():.4f}"

                print(log_str)
        
        if epoch % 100 == 0:
            save_checkpoint(model, G, D, epoch=epoch, out_dir = args.checkpoint_dir)
            save_samples(model, device, sample_batch, output = args.sample_dir, epoch=epoch,max_num=3)

            print(f"Saved the checkpoint at:{args.checkpoint_dir}\nSaved the samples at:{args.sample_dir}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train (Baseline / Improved) CycleGAN on an unpaired image dataset"
    )

    # Data
    parser.add_argument(
        "--root",
        type=str,
        default="datasets",
        help="Root folder containing the dataset subfolders",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset folder inside root (containing A/ and B/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--load_size",
        type=int,
        default=286,
        help="Resize shorter side to this before cropping",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="Central crop size",
    )

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="improved",
        choices=["baseline", "improved"],
        help="Which CycleGAN variant to use",
    )
    parser.add_argument(
        "--gan_mode",
        type=str,
        default="lsgan",
        choices=["lsgan", "vanilla", "hinge"],
        help="GAN loss type (for ImprovedCycleGan)",
    )

    # Loss weights
    parser.add_argument(
        "--lambda_cycle",
        type=float,
        default=10.0,
        help="Weight for cycle consistency loss",
    )
    parser.add_argument(
        "--lambda_identity",
        type=float,
        default=0.5,
        help="Weight for identity loss",
    )
    parser.add_argument(
        "--lambda_content",
        type=float,
        default=1.0,
        help="Weight for perceptual content loss (improved model)",
    )
    parser.add_argument(
        "--lambda_style",
        type=float,
        default=1.0,
        help="Weight for style loss (improved model)",
    )
    parser.add_argument(
        "--lambda_fm",
        type=float,
        default=1.0,
        help="Weight for feature-matching loss (improved model)",
    )

    # Optimizer
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for both G and D",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Adam beta1",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam beta2",
    )

    # Training
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )

    # Checkpoints & samples
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Folder to store model checkpoints",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="samples",
        help="Folder to store sample images",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from (optional)",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()






