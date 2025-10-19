# Noise-to-Image Diffusion Generator

A PyTorch implementation of a diffusion model for generating MNIST digits from pure noise. This project demonstrates the core concepts of denoising diffusion probabilistic models (DDPM) with a simple UNet architecture.

## 🎯 Project Overview

This implementation generates realistic MNIST handwritten digits by learning to reverse a noise process. Starting from random noise, the model learns to gradually denoise and create recognizable digit images through a series of diffusion steps.

## ✨ Features

- **Simple UNet Architecture**: Lightweight denoising network with encoder-decoder structure
- **Linear Beta Scheduling**: Configurable noise scheduling for the diffusion process
- **MNIST Dataset**: Trains on the classic handwritten digits dataset
- **Real-time Training Progress**: Progress bars and loss tracking during training
- **Sample Generation**: Generate new digits from pure noise
- **Visualization**: Matplotlib integration for displaying generated samples

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib tqdm
```

### Running the Model

```bash
python basic_diffusion_mnist.py
```

## 📊 Model Architecture

### UNet Denoiser
- **Input**: Noisy images + timestep information
- **Encoder**: 2D convolutions with stride for downsampling
- **Bottleneck**: Middle processing layers
- **Decoder**: Transpose convolutions for upsampling
- **Output**: Predicted noise to be removed

### Diffusion Process
- **Forward Process**: Gradually adds noise to clean images
- **Reverse Process**: Model learns to remove noise step by step
- **Timesteps**: 300 diffusion steps (configurable)
- **Noise Schedule**: Linear beta scheduling from 1e-4 to 0.02

## ⚙️ Configuration

```python
epochs = 5          # Training epochs
batch_size = 128    # Batch size for training
T = 300            # Number of diffusion timesteps
lr = 1e-4          # Learning rate
```

## 📈 Training Process

1. **Data Loading**: MNIST dataset with normalization
2. **Noise Addition**: Random timestep selection and noise injection
3. **Denoising**: Model predicts noise to be removed
4. **Loss Calculation**: MSE loss between predicted and actual noise
5. **Optimization**: Adam optimizer with gradient updates

## 🎨 Sample Generation

The model can generate new MNIST digits by:
1. Starting with pure random noise
2. Iteratively denoising through all timesteps
3. Producing clean digit images

## 📁 Project Structure

```
deep learning project/
├── basic_diffusion_mnist.py    # Main implementation
├── data/                       # MNIST dataset (auto-downloaded)
│   └── MNIST/
└── README.md                   # This file
```

## 🔧 Key Components

### Noise Scheduling
```python
def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)
```

### Training Loop
- Random timestep selection
- Noise injection with proper scaling
- Model prediction and loss calculation
- Gradient updates

### Sampling Process
- Reverse diffusion process
- Iterative denoising
- Final image generation

## 📊 Expected Output

After training, the model will:
- Display training progress with loss values
- Generate 8 sample MNIST digits
- Show them in a matplotlib subplot

## 🎯 Use Cases

- **Educational**: Understanding diffusion model fundamentals
- **Research**: Baseline for more complex diffusion architectures
- **Experimentation**: Foundation for advanced generative models

## 🔮 Future Enhancements

- **Timestep Embeddings**: Add proper time conditioning to the UNet
- **Advanced Schedulers**: Implement cosine or other noise schedules
- **Larger Models**: Scale up architecture for better quality
- **Conditional Generation**: Add class conditioning for specific digits
- **Higher Resolution**: Extend to larger image sizes

## 📚 References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation!

## 📄 License

This project is open source and available under the MIT License.
