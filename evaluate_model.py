import os
import torch
import glob
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

# ================= CONFIGURATION =================
OUTPUT_FILE = "eval_results.txt"

# List of style names to process
STYLES = ['cezanne', 'monet', 'ukiyoe', 'vangogh']

# List of model versions
VERSIONS = ['baseline', 'improved']

# Image indices in the stack
IDX_REAL_ART = 0
IDX_FAKE_PHOTO = 1
IDX_REAL_PHOTO = 3
IDX_FAKE_ART = 4

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= UTILS =================

def load_and_slice_image(img_path):
    """
    Loads a stacked image and slices it into 6 tensors.
    """
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None

    w, h = img.size
    
    # Stack
    num_patches = 6
    patch_h = h // num_patches
    patches = [img.crop((0, i * patch_h, w, (i + 1) * patch_h)) for i in range(num_patches)]

    # Transform to tensor [0, 1]
    to_tensor = transforms.ToTensor() 
    patch_tensors = [to_tensor(p).unsqueeze(0).to(device) for p in patches]

    return patch_tensors

# Metrics

def get_fid_metric():
    return FrechetInceptionDistance(feature=64, normalize=True).to(device)

def get_lpips_metric():
    return LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

# Main

def main():
    print(f"Current Working Directory: {os.getcwd()}") # <--- CHECK THIS OUTPUT
    
    # Open file for writing
    with open(OUTPUT_FILE, "w") as f:
        
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            f.flush()

        header = f"{'Style':<10} | {'Ver':<10} | {'FID (A2P)':<10} | {'LPIPS (A2P)':<12} | {'LPIPS (P2A)':<12}"
        log(header)
        log("-" * 70)

        found_any = False

        for style in STYLES:
            for version in VERSIONS:
                dir_name = f"samples/{style}2photo_{version}"
                
                # Get absolute path to be sure
                full_path = os.path.abspath(dir_name)
                
                if not os.path.exists(full_path):
                    print(f"--> SKIPPING: Could not find '{dir_name}'")
                    print(f"    (Looked at: {full_path})") 
                    continue

                # Check for images (case insensitive)
                extensions = ["*.jpg", "*.JPG", "*.png", "*.PNG"]
                img_files = []
                for ext in extensions:
                    img_files.extend(glob.glob(os.path.join(full_path, ext)))
                
                img_files = sorted(list(set(img_files))) # Remove duplicates

                if len(img_files) == 0:
                    log(f"{style:<10} | {version:<10} | {'EMPTY DIR':<10} | {'-':<12} | {'-':<12}")
                    continue

                found_any = True
                
                # Initialize metrics
                fid = get_fid_metric()
                lpips = get_lpips_metric()
                
                lpips_a2p_vals = []
                lpips_p2a_vals = []

                # Process images
                for img_file in tqdm(img_files, desc=f"{style}-{version}", leave=False):
                    patches = load_and_slice_image(img_file)
                    if patches is None: continue
                    
                    real_art = patches[IDX_REAL_ART]
                    fake_photo = patches[IDX_FAKE_PHOTO]
                    real_photo = patches[IDX_REAL_PHOTO]
                    fake_art = patches[IDX_FAKE_ART]

                    fid.update(real_photo, real=True)
                    fid.update(fake_photo, real=False)

                    val_a2p = lpips(real_art, fake_photo)
                    lpips_a2p_vals.append(val_a2p.item())

                    val_p2a = lpips(real_photo, fake_art)
                    lpips_p2a_vals.append(val_p2a.item())

                # Compute
                try:
                    final_fid = fid.compute().item()
                except:
                    final_fid = float('nan')
                    
                final_lpips_a2p = sum(lpips_a2p_vals) / len(lpips_a2p_vals) if lpips_a2p_vals else 0
                final_lpips_p2a = sum(lpips_p2a_vals) / len(lpips_p2a_vals) if lpips_p2a_vals else 0

                row = f"{style:<10} | {version:<10} | {final_fid:<10.2f} | {final_lpips_a2p:<12.4f} | {final_lpips_p2a:<12.4f}"
                log(row)
                
                fid.reset()
                lpips.reset()

        if not found_any:
            print("\nCRITICAL ERROR: No directories were processed.")
            print("Please move this script to the folder containing 'cezanne2photo_baseline', etc.")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()