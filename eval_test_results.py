import os
import torch
import glob
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# --- CONFIGURATION ---
INPUT_ROOT_DIR = './test_results'
OUTPUT_FILE = 'evaluation_summary_best.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- TRANSFORMS ---
lpips_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def to_uint8_tensor(pil_img):
    t = transforms.ToTensor()(pil_img)
    return (t * 255).to(dtype=torch.uint8)

def load_and_split(image_path):
    """
    Splits the CycleGAN output into (Real, Fake, Rec).
    Automatically detects Vertical (nrow=1) vs Horizontal stacking.
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    # Check Aspect Ratio to determine split direction
    if h > w: 
        # VERTICAL STACK (Your current script output)
        # [Real]
        # [Fake]
        # [Rec]
        single_h = h // 3
        part1 = img.crop((0, 0, w, single_h))
        part2 = img.crop((0, single_h, w, single_h * 2))
        part3 = img.crop((0, single_h * 2, w, h))
        
    else:
        # HORIZONTAL STACK (Standard CycleGAN)
        # [Real | Fake | Rec]
        single_w = w // 3
        part1 = img.crop((0, 0, single_w, h))
        part2 = img.crop((single_w, 0, single_w * 2, h))
        part3 = img.crop((single_w * 2, 0, w, h))
    
    # Resize parts to match exactly (prevents 1px rounding errors)
    target_size = part1.size
    if part2.size != target_size: part2 = part2.resize(target_size, Image.BICUBIC)
    if part3.size != target_size: part3 = part3.resize(target_size, Image.BICUBIC)
    
    return part1, part2, part3

def parse_folder_info(folder_path):
    """Extracts style and version from folder path strings."""
    parts = folder_path.split(os.sep)
    experiment_name = ""
    
    for p in parts:
        if '2photo_' in p:
            experiment_name = p
            break
            
    if not experiment_name:
        return None, None

    try:
        style_part, version = experiment_name.split('2photo_')
        style = style_part
    except ValueError:
        style = experiment_name
        version = "unknown"
        
    return style, version

def calculate_metrics_for_folder(folder_path):
    """Runs metrics on a single folder and returns scores."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))])
    
    # Filter for valid test files
    testA_files = [f for f in files if 'testA' in f]
    testB_files = [f for f in files if 'testB' in f]

    if not testA_files or not testB_files:
        return None

    # Initialize Metrics
    # Resetting metrics per folder is safer to avoid state accumulation issues
    fid_metric = FrechetInceptionDistance(feature=64).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(DEVICE)
    
    scores = {'lpips_a': [], 'lpips_b': []}

    # --- PROCESS TEST B (Real Photos) ---
    # Stack: Real Photo -> Fake Art -> Rec Photo
    for f in testB_files:
        p1_real, p2_fake, p3_rec = load_and_split(os.path.join(folder_path, f))
        
        # FID: Add Real Photos to Reference Dist (real=True)
        fid_metric.update(to_uint8_tensor(p1_real).unsqueeze(0).to(DEVICE), real=True)
        
        # LPIPS: Real Photo vs Rec Photo
        real_t = lpips_transform(p1_real).unsqueeze(0).to(DEVICE)
        rec_t = lpips_transform(p3_rec).unsqueeze(0).to(DEVICE)
        scores['lpips_b'].append(lpips_metric(real_t, rec_t).item())

    # --- PROCESS TEST A (Fake Photos) ---
    # Stack: Real Art -> Fake Photo -> Rec Art
    for f in testA_files:
        p1_real, p2_fake, p3_rec = load_and_split(os.path.join(folder_path, f))
        
        # FID: Add Fake Photos to Generated Dist (real=False)
        fid_metric.update(to_uint8_tensor(p2_fake).unsqueeze(0).to(DEVICE), real=False)
        
        # LPIPS: Real Art vs Rec Art
        real_t = lpips_transform(p1_real).unsqueeze(0).to(DEVICE)
        rec_t = lpips_transform(p3_rec).unsqueeze(0).to(DEVICE)
        scores['lpips_a'].append(lpips_metric(real_t, rec_t).item())

    try:
        # Compute final scores
        fid = fid_metric.compute().item()
        
        # Safety check for empty lists
        avg_lpips_a = sum(scores['lpips_a']) / len(scores['lpips_a']) if scores['lpips_a'] else 999.0
        avg_lpips_b = sum(scores['lpips_b']) / len(scores['lpips_b']) if scores['lpips_b'] else 999.0
        
        return fid, avg_lpips_a, avg_lpips_b
    except Exception as e:
        print(f"Error computing metrics for {folder_path}: {e}")
        return None

# --- MAIN ---
print(f"Scanning {INPUT_ROOT_DIR}...")
results_db = defaultdict(list)

# 1. Walk and Process
for root, dirs, files in os.walk(INPUT_ROOT_DIR):
    if any(f.endswith('.png') for f in files):
        style, version = parse_folder_info(root)
        if style:
            print(f"  -> Processing {style} | {version} ...")
            metrics = calculate_metrics_for_folder(root)
            if metrics:
                fid, lp_a, lp_b = metrics
                results_db[(style, version)].append({
                    'fid': fid,
                    'lpips_a': lp_a,
                    'lpips_b': lp_b
                })

# 2. Format Table (Independent Bests)
header = f"{'Style':<10} | {'Ver':<10} | {'Best FID (A2P)':<15} | {'Best LPIPS (A2P)':<18} | {'Best LPIPS (P2A)':<18}"
sep = "-" * len(header)
table_lines = [header, sep]

print("\nGenerating Independent Best Metrics Report...")

sorted_keys = sorted(results_db.keys())

for (style, version) in sorted_keys:
    runs = results_db[(style, version)]
    
    if not runs:
        continue

    # Calculate independent bests (Minimums)
    best_fid = min(r['fid'] for r in runs)
    best_lpips_a = min(r['lpips_a'] for r in runs)
    best_lpips_b = min(r['lpips_b'] for r in runs)
    
    line = (
        f"{style:<10} | "
        f"{version:<10} | "
        f"{best_fid:<15.2f} | "
        f"{best_lpips_a:<18.4f} | "
        f"{best_lpips_b:<18.4f}"
    )
    table_lines.append(line)

full_output = "\n".join(table_lines)
print("\n" + full_output)

with open(OUTPUT_FILE, 'w') as f:
    f.write(full_output)

print(f"\nSaved to {OUTPUT_FILE}")