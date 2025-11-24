import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Load VGG16 Weights
        # We use VGG16 because its architecture (simple stacks of convs) retains 
        # spatial information better than ResNets (which downsample aggressively).
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features.eval() # Freeze BatchNorm layers

        # 2. Register Normalization Constants
        # VGG was trained on ImageNet, so it expects inputs normalized to these specific stats.
        # We use register_buffer so these tensors are saved with the model state but not treated as trainable parameters.
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 3. Freeze Parameters
        # We are not training VGG; we are only using it to measure distance. 
        # Setting requires_grad=False saves significant memory (no gradients stored for VGG weights).
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 4. Grayscale Handling
        # VGG's first conv layer filters have shape [64, 3, 3, 3]. It physically cannot accept 1-channel input.
        # If we project a grayscale image, we must broadcast it to 3 channels (RGB) so the math works.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 5. Resolution Matching (The Receptive Field Issue)
        # StyleGAN2 generates 1024x1024 images. VGG was trained on 224x224.
        # If we feed 1024px images, the features (eyes, nose) are too large for VGG's kernels to recognize 
        # as "eyes" or "noses". The features would look like low-frequency gradients to VGG.
        # Downsampling to 256px realigns the frequency domain of the image with VGG's learned filters.
        if x.shape[-1] > 256:
            x = F.interpolate(x, size=(256, 256), mode='area')

        # 6. Domain Adaptation (StyleGAN space -> ImageNet space)
        # StyleGAN generator output is tanh-like: [-1, 1]
        # VGG input requirement: Normalized with ImageNet mean/std (approx range [-2, 2])
        x = (x + 1) * 0.5             # Un-normalize: [-1, 1] -> [0, 1]
        x = (x - self.mean) / self.std # Re-normalize: [0, 1] -> ImageNet Standardized

        # 7. Feature Extraction
        # We extract activations from specific layers to capture different levels of abstraction.
        # indices correspond to: conv1_1, conv1_2, conv3_2, conv4_2
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            # These specific indices [1, 3, 13, 20] correspond to the Rectified Linear Units (ReLUs) 
            # after the convolutions specified in the Image2StyleGAN paper.
            if i in [1, 3, 13, 20]: 
                features.append(x)
            if i == 20: # Optimization: Stop forward pass early to save compute
                break
        return features

class ProjectorLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.vgg = VGG16FeatureExtractor().to(device)
        self.mse = nn.MSELoss()

    def forward(self, target_img, generated_img):
        """
        Calculates the gradients for W latent optimization.
        """
        
        # 8. Pixel-wise Loss (High Frequency / Color)
        # MSE pushes the pixel values to be identical. This handles color balance and overall tone 
        # but results in blurry images if used alone (regression to the mean).
        loss_mse = self.mse(generated_img, target_img)

        # 9. Perceptual Loss (Structural / Semantic)
        # We project both images into VGG feature space.
        target_features = self.vgg(target_img)
        gen_features = self.vgg(generated_img)

        loss_perceptual = 0
        # We sum the L2 distances between features at 4 different scales.
        # Early layers (conv1_1) ensure texture/edge alignment.
        # Deeper layers (conv4_2) ensure semantic alignment (e.g., "is there an eye here?").
        for targ_feat, gen_feat in zip(target_features, gen_features):
            loss_perceptual += F.mse_loss(gen_feat, targ_feat)

        return loss_mse, loss_perceptual