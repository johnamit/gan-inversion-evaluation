import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim
from facenet_pytorch import InceptionResnetV1
import pandas as pd
import os
import argparse
import sys


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help="Path to the REAL target image")
    parser.add_argument('--native', type=str, required=True, help="Path to the NATIVE reconstructed image")
    parser.add_argument('--e4e', type=str, required=True, help="Path to the E4E reconstructed image")
    parser.add_argument('--img2style', type=str, required=True, help="Path to the Img2Style reconstructed image")

    return parser.parse_args()


# helper functions
def load_and_process_image(path, device, size=(256, 256)):
    """Loads an image and returns it in multiple formats required by different metrics."""
    # check if file exists
    if not os.path.exists(path):
        print(f"ERROR: Image not found at {path}")
        return None, None, None
    
    # handle grayscale images
    img = Image.open(path).convert('RGB')
    img = img.resize(size)

    # Transform for tensors [-1, 1] - LPIPS, FaceNet
    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform_tensor(img).unsqueeze(0).to(device)

    # Transform for numpy arrays [0, 1] - SSIM
    img_np = np.array(img).astype(np.float32) / 255.0

    return img, img_tensor, img_np


def calculate_id_similarity(img1_tensor, img2_tensor, facenet_model):
    """Calculates identity similarity using FaceNet embeddings."""
    with torch.no_grad():
        emb1 = facenet_model(img1_tensor)
        emb2 = facenet_model(img2_tensor)
    cos_sim = nn.functional.cosine_similarity(emb1, emb2).item()
    
    return cos_sim


# main function
def main():
    # parse arguments
    args = parse_args()

    target_path = args.target
    generated_paths = {
        'NATIVE': args.native,
        'E4E': args.e4e,
        'Img2Style': args.img2style
    }

    # device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load metrics models
    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    print("Loading FaceNet model...")
    facenet_model = InceptionResnetV1(pretrained='vggface2').to(device).eval()

    # process target image
    print(f"Processing target image: {target_path}")
    target_img, target_tensor, target_np = load_and_process_image(target_path, device)
    
    if target_tensor is None:
        sys.exit(1)

    results = []

    # evaluation loop
    for method_name, gen_path in generated_paths.items():
        print(f"Evaluating {method_name} - image: {gen_path}")
        
        gen_img, gen_tensor, gen_np = load_and_process_image(gen_path, device)
        
        if gen_tensor is None:
            print(f"Skipping {method_name}, no image found.")
            continue

        # MSE (L2 Distance) - Lower is better
        mse_score = nn.functional.mse_loss(gen_tensor, target_tensor).item()

        # SSIM - Higher is better
        try:
            # channel_axis=2 is for [H, W, C] arrays (Newer Scikit-Image)
            ssim_score = ssim(target_np, gen_np, channel_axis=2, data_range=1.0)
        except TypeError:
            # Fallback for older versions
            ssim_score = ssim(target_np, gen_np, multichannel=True, data_range=1.0)

        # LPIPS - Lower is better
        with torch.no_grad():
            lpips_score = lpips_model(target_tensor, gen_tensor).item()

        # ID Similarity - Higher is better
        id_score = calculate_id_similarity(target_tensor, gen_tensor, facenet_model)

        results.append({
            'Method': method_name,
            "MSE (Lower is better)": mse_score,
            "SSIM (Higher is better)": ssim_score,
            "LPIPS (Lower is better)": lpips_score,
            "ID Similarity (Higher is better)": id_score
        })

    # output results
    print("\n" + "-"*50 + "\nEvaluation Results:\n" + "-"*50)
    df = pd.DataFrame(results)
    
    # Pandas formatting options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    print(df.to_string(index=False))
    print("\n" + "-"*50)

if __name__ == "__main__":
    main()


