import sys, os
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../repos/stylegan2-ada-pytorch'))
sys.path.append(repo_path)

import copy
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
import dnnlib
import legacy
from loss import ProjectorLoss

def project(
    G, # generator
    target: torch.Tensor, # target image
    num_steps=1000, # number of optimization steps
    w_avg_samples=10000, # number of samples to calculate w_avg
    initial_learning_rate=0.1, # initial learning rate
    initial_noise_factor=0.05, # initial noise factor
    # ramp means the fraction of the total steps
    lr_rampdown_length=0.25, # length of lr rampdown
    lr_rampup_length=0.05, # length of lr rampup
    noise_ramp_length=0.75, # length of noise ramp
    regularize_noise_weight=1e5, # weight of noise regularization
    device: torch.device = torch.device('cuda'), # run on device CUDA
    verbose=True # print progress
):
    """
    Project a target image into the latent space of a given Generator.
    """

    # 1. Initialise the latent vector w.
    # compute the average latent vector w_avg to use as a starting point for the optimization. Starting at the mean ensures convergence to a realistic result
    # sample the z vectors and convert to torch tensor
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim) 
    z_samples = torch.from_numpy(z_samples).to(device)
    # map to w space using the mapping network
    w_samples = G.mapping(z_samples, None)  # [N, L, 512]
    w_avg = torch.mean(w_samples, dim=0, keepdim=True)  # [1, L, 512]

    # clone w_avg to create the optimisable w_opt
    # We want to optimize W+, so we ensure it has shape [1, num_ws, 512].
    # .detach() creates a leaf node in the graph. .requires_grad_(True) enables Adam to update it.
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True

    
    # 2. Set up the noise buffer
    # We locate the 'noise_const' buffers in the generator synthesis blocks.
    # These are normally fixed Gaussian noise. We want to optimize them slightly to capture high-frequency texture details (hair, pores) that W can't capture.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # We initialize the optimizer with both the Latent Code and the Noise Buffers.
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate) # betas are hyperparameters for Adam. 0.9 and 0.999 are common default values.

    # 3. Preparation for optimization loop
    # initialize the VGE + MSE loss function
    loss_fn = ProjectorLoss(device=device)

    # Normalize target image to [-1, 1] to match StyleGAN generator output
    # Input target is [0, 255].
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 127.5 - 1.0


    # 4. Optimization loop
    for step in range(num_steps):
        # learning rate schedule
        t = step / num_steps
        w_noise_scale = w_opt.std() * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)  # cosine rampdown
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)  # rampup
        lr = initial_learning_rate * lr_ramp

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # stochastic search
        # We add random noise to w_opt before the forward pass. This helps the optimizer escape out of local minima in the complex W space.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise

        # forward pass
        # noise_mode='const' forces the G to use our `noise_bufs` (which we are optimizing) rather than generating fresh random noise every step.
        synth_images = G.synthesis(ws, noise_mode='const')

        # handle greyscale images by duplicating channels
        if synth_images.shape[1] == 1 and target_images.shape[1] == 3: # if G is greyscale but target is RGB
             synth_images = synth_images.repeat(1, 3, 1, 1) # repeat channels

        # compute loss
        loss_mse, loss_percept = loss_fn(target_images, synth_images)
        
        # Combine losses
        # The Image2StyleGAN paper treats them equally (1.0 weights), but in practice, VGG loss magnitude is much higher, dominating the total loss.
        loss = loss_mse + loss_percept

        # noise regularization
        # This prevents "Signal Sneaking" (the artifacts described in StyleGAN2 paper).
        # We penalize the noise maps if pixels are correlated with their neighbors.
        reg_loss = 0.0
        for val in noise_bufs.values():
            if val.ndim == 2: # single channel noise
                noise = val
            elif val.ndim == 3: # multi-channel noise
                noise = val[0]
            else: # multi-batch multi-channel
                noise = val[0, 0]

            while True:
                # calculate autocorrelation in both dimensions
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=0)).mean() ** 2 # shift by 1, then compute mean squared correlation
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=1)).mean() ** 2 # shift by 1, then compute mean squared correlation

                # downsample noise and repeat (multi-scale regularization)
                if noise.shape[0] <= 8: # 8 x 8 is the minimum size
                    break
                noise = F.avg_pool2d(noise.unsqueeze(0).unsqueeze(0), kernel_size=2).squeeze()
        
        loss += reg_loss * regularize_noise_weight

        # backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Post-Step normalisation
        # Noise maps in StyleGAN are expected to be unit variance Gaussian.
        # Optimization might blow up their values, so we force them back to N(0, 1).
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        # log the progress
        if verbose and step % 100 == 0: # print every 100 steps
            print(f'step {step + 1:>4d}/{num_steps}: '
                  f'mse {loss_mse:<5.2f} '
                  f'percept {loss_percept:<5.2f} '
                  f'noise_reg {reg_loss * regularize_noise_weight:<5.2f}')
        
    # return the optimised w vector and final image
    return w_opt.detach(), synth_images.detach()
    

def run_projection(
    network_pkl: str, # path to the StyleGAN2 network pickle
    target_fname: str, # path to the target image
    outdir: str, # output directory
    num_steps: int, # number of optimization steps
    seed: int # random seed
):
    # initialise seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda')

    # load pre-trained networks
    print(f"Loading networks from {network_pkl}")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # load target image
    print(f'Loading target image from {target_fname}')
    target_pil = Image.open(target_fname).convert('RGB')

    # handle greyscale models
    if G.img_channels == 1:
        target_pil = target_pil.convert('L') # convert to greyscale

    # Center crop and resize
    w, h = target_pil.size # get height and width 
    s = min(w, h) # size of the shortest side
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)) # center crop
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), Image.Resampling.LANCZOS) # resize to model resolution

    # Convert to Tensor [C, H, W]
    target_tensor = torch.tensor(np.array(target_pil)).to(device)
    
    if target_tensor.ndim == 2: 
        target_tensor = target_tensor.unsqueeze(0)
    elif target_tensor.ndim == 3: 
        target_tensor = target_tensor.permute(2, 0, 1)

    # project the image to latent space
    print("Starting projection ...")
    projected_w, projected_img_tensor = project(G, target_tensor, num_steps=num_steps, device=device)

    # save results:
    os.makedirs(outdir, exist_ok=True) # make output directory
    
    print(f'Saving results to "{outdir}"')
    # save projected w as .npy
    projected_w_np = projected_w.cpu().numpy()
    np.savez(f'{outdir}/projected_w.npz', w=projected_w_np[0,0], ws=projected_w_np[0]) # save both W and W+

    # save projected image as .png
    projected_img = (projected_img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_np = projected_img[0].cpu().numpy()

    mode = 'RGB'
    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)
        mode = 'L'

    Image.fromarray(img_np, mode=mode).save(f'{outdir}/projected_img.png')
    target_pil.save(f'{outdir}/target_img.png')

    print(f"Projection complete. Results saved to {outdir}")

if __name__ == "__main__":
    # Argparse Setup
    parser = argparse.ArgumentParser(description="Project an image into the latent space of a StyleGAN2-ADA model.")
    parser.add_argument('--network', required=True, help='Path to the network pickle filename')
    parser.add_argument('--target', required=True, help='Path to the target image file to project')
    parser.add_argument('--outdir', required=True, help='Directory to save the output images and npz')
    parser.add_argument('--num-steps', type=int, default=1000, help='Number of optimization steps (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--save_steps', type=int, default=1000, help='Interval of steps to save intermediate results (default: 1000)')

    args = parser.parse_args()

    # run the projection
    run_projection(
        network_pkl=args.network,
        target_fname=args.target,
        outdir=args.outdir,
        num_steps=args.num_steps,
        seed=args.seed
    )

# Note: To run this script with the default parameters, use the command below:

# python projector.py \
#     --network checkpoints/your_model.pkl \
#     --target targets/face.png \
#     --outdir image2stylegan_results/test_1 \

# optional arguments --num_steps and --seed can also be specified.