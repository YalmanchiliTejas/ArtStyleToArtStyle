import os
import glob
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# Import your models
from baseline_model import BaselineCycleGan
from modified_model import ImprovedCycleGan

# --- 1. Reused Data Loading Logic (from train_baseline.py) ---
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='test'):
        super().__init__()
        # Adjusting folder names based on standard CycleGAN structure (testA/testB)
        # If your data is just A/B, change "testA" to "A"
        self.dir_x = os.path.join(root, "testA")
        self.dir_y = os.path.join(root, "testB")

        # Fallback to A/B if testA/testB don't exist
        if not os.path.exists(self.dir_x):
            self.dir_x = os.path.join(root, "A")
        if not os.path.exists(self.dir_y):
            self.dir_y = os.path.join(root, "B")

        self.sorted_paths_x = sorted(glob.glob(os.path.join(self.dir_x, "*")))
        self.sorted_paths_y = sorted(glob.glob(os.path.join(self.dir_y, "*")))

        self.transform = transform

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

        # Return paths as well for HTML generation
        return {"X": x_image, "Y": y_image, "path_X": path_x, "path_Y": path_y}

# --- 2. Utils for HTML and Checkpoints ---

def load_checkpoint(model, path, device="cpu"):
    if not os.path.isfile(path):
        print(f"Error: Checkpoint file not found at {path}")
        return
    
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device)

    model.G_xy.load_state_dict(checkpoint["G_xy"])
    model.F_yx.load_state_dict(checkpoint["F_yx"])
    # We don't strictly need Discriminators for inference, but loading them ensures model integrity
    if "D_x" in checkpoint: model.D_x.load_state_dict(checkpoint["D_x"])
    if "D_y" in checkpoint: model.D_y.load_state_dict(checkpoint["D_y"])

    epoch = checkpoint.get("epoch", 0)
    print(f"Loaded model trained for {epoch} epochs.")

def create_html(web_dir, img_dict_list):
    """
    Replicates the Lua HTML generation.
    img_dict_list: list of dictionaries containing relative image paths for a row.
    """
    html_path = os.path.join(web_dir, 'index.html')
    with open(html_path, 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<body>\n')
        f.write('<h3>CycleGAN Test Results</h3>\n')
        f.write('<table style="text-align:center;">\n')
        
        # Header
        if len(img_dict_list) > 0:
            f.write('<tr>\n')
            f.write('<td> <b>Id</b> </td>\n')
            for key in img_dict_list[0].keys():
                f.write(f'<td> <b>{key}</b> </td>\n')
            f.write('</tr>\n')

        # Rows
        for i, img_dict in enumerate(img_dict_list):
            f.write('<tr>\n')
            f.write(f'<td> {i+1} </td>\n')
            for key, path in img_dict.items():
                f.write(f'<td><img src="{path}" style="width:200px"></td>\n')
            f.write('</tr>\n')
        
        f.write('</table>\n</body>\n</html>')
    print(f"Webpage saved to: {html_path}")

def denorm(tensor):
    """Reverses the Normalization (mean=0.5, std=0.5) -> [0, 1]"""
    return (tensor + 1) / 2.0

# --- 3. Main Testing Logic ---

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Create directories
    web_dir = os.path.join(args.results_dir, args.name)
    img_dir = os.path.join(web_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # Transform (Note: No RandomCrop or Flip for testing, just Resize and Normalize)
    transform = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size)), # Usually square for CycleGAN
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset
    dataset_root = os.path.join(args.root, args.dataset_name)
    test_dataset = ImageDataset(dataset_root, transform=transform, mode='test')
    
    # DataLoader (Shuffle=False is important for testing to keep order)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    print(f"Dataset loaded: {len(test_dataset)} images.")

    # Model Initialization
    if args.model_type == "baseline":
        model = BaselineCycleGan(in_channels=3, out_channels=3) # Lambdas don't matter for inference
    elif args.model_type == "improved":
        # Params required for init, though unused in inference
        model = ImprovedCycleGan(in_channels=3, out_channels=3)
    else:
        raise ValueError("Model type must be 'baseline' or 'improved'")

    model.to(device)
    model.eval()

    # Load Weights
    load_checkpoint(model, args.checkpoint_path, device=device)

    # Helper for saving results for HTML
    webpage_data = []

    print(f"Starting inference on {min(args.how_many, len(test_dataset))} images...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.how_many:
                break
            
            real_X = batch["X"].to(device)
            real_Y = batch["Y"].to(device)
            path_X = batch["path_X"][0]
            path_Y = batch["path_Y"][0]
            
            name_X = os.path.splitext(os.path.basename(path_X))[0]
            name_Y = os.path.splitext(os.path.basename(path_Y))[0]

            # Forward Pass
            # Generate Fake Y from Real X
            fake_Y = model.G_xy(real_X)
            # Reconstruct X from Fake Y
            rec_X = model.F_yx(fake_Y)
            
            # Generate Fake X from Real Y
            fake_X = model.F_yx(real_Y)
            # Reconstruct Y from Fake X
            rec_Y = model.G_xy(fake_X)

            # Process images for saving
            visuals = {
                "real_A": real_X, 
                "fake_B": fake_Y, 
                "rec_A": rec_X,
                "real_B": real_Y, 
                "fake_A": fake_X, 
                "rec_B": rec_Y
            }

            row_dict = {}

            for label, tensor in visuals.items():
                img_tensor = denorm(tensor)
                # Use name_X for A-domain stuff, name_Y for B-domain stuff to keep filenames unique
                base_name = name_X if "A" in label or "fake_B" in label else name_Y
                save_filename = f"{base_name}_{label}.png"
                save_path = os.path.join(img_dir, save_filename)
                
                save_image(img_tensor, save_path)
                
                # Store relative path for HTML
                row_dict[label] = os.path.join('images', save_filename)

            webpage_data.append(row_dict)
            print(f"Processed batch {i+1}: {name_X} / {name_Y}")

    # Generate HTML
    create_html(web_dir, webpage_data)

def parse_args():
    parser = argparse.ArgumentParser(description="Test CycleGAN")

    parser.add_argument("--root", type=str, default="datasets", help="Root containing dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset folder name")
    parser.add_argument("--name", type=str, default="experiment_name", help="Name of the experiment (for results folder)")
    parser.add_argument("--results_dir", type=str, default="results", help="Where to save results")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .pt file to load")
    
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "improved"])
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--load_size", type=int, default=256, help="Scale images to this size")
    
    parser.add_argument("--how_many", type=int, default=50, help="Number of test images to run")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test(args)