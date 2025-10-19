import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 5
batch_size = 128
T = 300  # number of diffusion steps
lr = 1e-4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data = datasets.MNIST(root='./data', download=True, transform=transform)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

betas = linear_beta_schedule(T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Simple UNet Denoiser
class SimpleUNet(nn.Module):
    def __init__(self, c_in=1, c_out=1):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c_in, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.mid = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, c_out, 3, padding=1)
        )

    def forward(self, x, t):
        # (t) not used here — simple baseline, add embeddings later
        x = self.down(x)
        x = self.mid(x)
        x = self.up(x)
        return x

model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for x, _ in loop:
        x = x.to(device)
        t = torch.randint(0, T, (x.size(0),), device=device)
        noise = torch.randn_like(x)

        sqrt_alpha_cum = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        x_noisy = sqrt_alpha_cum * x + sqrt_one_minus * noise

        pred_noise = model(x_noisy, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

print("✅ Training complete!")


# Sampling Function
@torch.no_grad()
def sample(model, n=8):
    model.eval()
    x = torch.randn((n, 1, 28, 28), device=device)
    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_cum_t = alphas_cumprod[t]

        pred_noise = model(x, torch.tensor([t] * n, device=device))
        x = (1 / alpha_t.sqrt()) * (x - ((1 - alpha_t) / ((1 - alpha_cum_t).sqrt())) * pred_noise) + beta_t.sqrt() * z
    return x


# Generate Samples
samples = sample(model, n=8).cpu()

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    axes[i].imshow(samples[i].squeeze(), cmap="gray")
    axes[i].axis("off")
plt.show()