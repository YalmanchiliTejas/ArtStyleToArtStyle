import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# Import your models
from baseline_model import BaselineCycleGan
from modified_model import ImprovedCycleGan

# --- 1. Dataset --------------------------------------------------------------
class SingleDomainDataset(Dataset):
    def __init__(self, root, folder_name, transform=None):
        super().__init__()
        self.dir = os.path.join(root, folder_name)
        
        # Fallback logic
        if not os.path.exists(self.dir):
            if "test" in folder_name:
                alt = folder_name.replace("test", "train")
                self.dir = os.path.join(root, alt)
        
        self.paths = sorted(glob.glob(os.path.join(self.dir, "*")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"img": image, "path": path}

# --- 2. Utils ----------------------------------------------------------------
def load_weights(model, path, device):
    print(f"Loading weights: {os.path.basename(path)}")
    checkpoint = torch.load(path, map_location=device)
    core = model.module if isinstance(model, nn.DataParallel) else model
    core.G_xy.load_state_dict(checkpoint["G_xy"])
    core.F_yx.load_state_dict(checkpoint["F_yx"])

def denorm_and_resize(tensor):
    """Normalizes (-1,1) -> (0,1) and resizes to 256x256."""
    tensor = (tensor + 1.0) / 2.0
    tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    return tensor.cpu()

# --- 3. Inference Logic ------------------------------------------------------
def run_inference(model, root, dataset_name, device, output_dir, max_images):
    # Transforms matching training (286 -> 256)
    transform = transforms.Compose([
        transforms.Resize(286),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    os.makedirs(output_dir, exist_ok=True)

    # --- PART A: Test A (Real A -> Fake B -> Rec A) ---
    dataset_A = SingleDomainDataset(os.path.join(root, dataset_name), "testA", transform)
    loader_A = DataLoader(dataset_A, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"Processing {len(dataset_A)} images from Domain A...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader_A):
            if i >= max_images: break
            
            if i % 10 != 0:
                continue
            real_A = batch["img"].to(device)
            
            # Forward Cycle
            fake_B = model.G_xy(real_A)
            rec_A = model.F_yx(fake_B)
            
            # Stack: [Real A, Fake B, Rec A]
            stack = torch.cat([
                denorm_and_resize(real_A), 
                denorm_and_resize(fake_B), 
                denorm_and_resize(rec_A)
            ], dim=0)

            # Save: testA_1.png
            save_path = os.path.join(output_dir, f"testA_{i+1}.png")
            save_image(stack, save_path, nrow=1) # nrow=1 for vertical column

    # --- PART B: Test B (Real B -> Fake A -> Rec B) ---
    dataset_B = SingleDomainDataset(os.path.join(root, dataset_name), "testB", transform)
    loader_B = DataLoader(dataset_B, batch_size=1, shuffle=False, num_workers=4)

    print(f"Processing {len(dataset_B)} images from Domain B...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader_B):
            if i >= max_images: break
            if i % 20 != 0:
                continue
            real_B = batch["img"].to(device)
            
            # Reverse Cycle
            fake_A = model.F_yx(real_B)
            rec_B = model.G_xy(fake_A)

            # Stack: [Real B, Fake A, Rec B]
            stack = torch.cat([
                denorm_and_resize(real_B), 
                denorm_and_resize(fake_A), 
                denorm_and_resize(rec_B)
            ], dim=0)

            # Save: testB_1.png
            save_path = os.path.join(output_dir, f"testB_{i+1}.png")
            save_image(stack, save_path, nrow=1)

# --- 4. Main -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="baseline")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--root", type=str, default="datasets")
    
    # Default is infinity (process everything)
    parser.add_argument("--max_images", type=float, default=float('inf'))

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    if args.model_type == "baseline":
        model = BaselineCycleGan(3, 3, lambda_cycle=10, lambda_identity=0.5)
    else:
        model = ImprovedCycleGan(3, 3, lambda_cycle=10, lambda_identity=0.5, 
                                 lambda_content=0, lambda_style=0, lambda_fm=0, gan_mode="lsgan")
    
    model.to(device)
    model.eval()

    if os.path.isdir(args.checkpoint_dir):
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    else:
        checkpoints = [args.checkpoint_dir]

    if not checkpoints:
        print("No checkpoints found.")
        return

    for cp_path in checkpoints:
        cp_name = os.path.splitext(os.path.basename(cp_path))[0]
        output_dir = os.path.join("test_results", f"{args.dataset_name}_{args.model_type}", cp_name)
        
        load_weights(model, cp_path, device)
        run_inference(model, args.root, args.dataset_name, device, output_dir, args.max_images)
        print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    main()