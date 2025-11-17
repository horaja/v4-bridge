import sys
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import numpy as np

sys.path.append('./models/v4_digital_twin')
from DTwin_infer import TorchModel

def compute_rsa(act1, act2):
    act1_flat = act1.reshape(act1.shape[0], -1).numpy()
    act2_flat = act2.reshape(act2.shape[0], -1).numpy()

    rdm1 = pdist(act1_flat, metric='correlation')
    rdm2 = pdist(act2_flat, metric='correlation')

    rho, _ = spearmanr(rdm1, rdm2)
    return rho

def compute_cka(act1, act2):
    act1_flat = act1.reshape(act1.shape[0], -1).numpy()
    act2_flat = act2.reshape(act2.shape[0], -1).numpy()

    K = act1_flat @ act1_flat.T
    L = act2_flat @ act2_flat.T

    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    K_c = H @ K @ H
    L_c = H @ L @ H

    hsic = np.sum(K_c * L_c)
    normalization = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))

    return hsic / normalization

def main():
    transform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = STL10(root='./data', split='train', download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=50, shuffle=False)

    resnet = models.resnet50(pretrained=True)
    resnet.eval()

    activations = {}
    def hook_fn(name):
        def fn(module, input, output):
            activations[name] = output.detach().cpu()
        return fn

    for name, module in resnet.named_children():
        module.register_forward_hook(hook_fn(name))

    images, _ = next(iter(loader))
    with torch.no_grad():
        resnet(images)

    v4_model = TorchModel()
    v4_model.load_state_dict(torch.load('models/v4_digital_twin/pytorch_model_weights.pth'))
    v4_model.eval()

    with torch.no_grad():
        v4_output = v4_model(images).cpu()

    print("STL-10 batch shape:", images.shape)
    print(f"V4 output: {v4_output.shape}")

    print("\nSimilarity scores (ResNet layers vs V4):")
    rsa_scores = {}
    cka_scores = {}
    for name, act in activations.items():
        rsa_scores[name] = compute_rsa(act, v4_output)
        cka_scores[name] = compute_cka(act, v4_output)
        print(f"  {name}: RSA={rsa_scores[name]:.4f}, CKA={cka_scores[name]:.4f}")

    best_rsa = max(rsa_scores, key=rsa_scores.get)
    best_cka = max(cka_scores, key=cka_scores.get)
    print(f"\nBest RSA: {best_rsa} ({rsa_scores[best_rsa]:.4f})")
    print(f"Best CKA: {best_cka} ({cka_scores[best_cka]:.4f})")

if __name__ == "__main__":
    main()
