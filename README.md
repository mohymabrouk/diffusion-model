# Diffusion Model for Emoji Generation

## Overview
This project implements a diffusion model for generating emoji images based on text prompts. The model is trained on a dataset of emoji images from multiple platforms (Apple, Facebook, Google, JoyPixels, Samsung, Twitter, Windows) with associated text descriptions.

## Requirements

### Python Version
- Python (version shown in notebook)

### Dependencies
- tiktoken
- OpenCV (cv2)
- NumPy
- pandas
- matplotlib
- tqdm
- Pillow (PIL)
- TensorFlow
- scikit-learn

## Installation
```bash
pip install tiktoken
```

## Dataset

### Sources
The model uses emoji images from 7 different platforms:
- Apple
- Facebook
- Google
- JoyPixels
- Samsung
- Twitter
- Windows

### Data Structure
- Images are PNG files named numerically (1.png, 2.png, etc.)
- Text descriptions are provided in a CSV file (`full_emoji.csv`)
- 400 images are loaded from each platform

### Preprocessing
1. Images are resized to 64x64 pixels
2. Converted from BGR to RGB color space
3. Normalized to range [-1, 1]
4. Text prompts are tokenized using GPT-2 tokenizer

## Model Architecture

### Diffusion Process
- Forward process adds Gaussian noise gradually over 300 timesteps
- Beta schedule: linear from 0.0001 to 0.02
- Uses reparameterization trick for efficient sampling

### U-Net Model
The model consists of:
- Sinusoidal position embeddings for timesteps
- Text embeddings using GPT-2 tokenizer vocabulary
- Encoder-decoder architecture with residual connections
- 4 encoder blocks (64→128→256→512→1024 channels)
- 4 decoder blocks (1024→512→256→128→64 channels)
- GELU activation functions
- Batch normalization

### Key Components
1. **BlockUnet**: Custom U-Net block for down/up sampling
2. **SinusoidalPositionEmbeddings**: Time step embeddings
3. **GELU**: Gaussian Error Linear Unit activation

## Training

### Hyperparameters
- Batch size: 256
- Timesteps: 300
- Learning rate: 1e-4
- Epochs: 300
- Seed: 42

### Loss Function
Mean squared error between predicted noise and actual noise.

### Training Process
1. Sample random timesteps
2. Add noise to images according to timestep
3. Predict noise using model
4. Calculate MSE loss
5. Update parameters using Adam optimizer

## Usage

### Training
The training loop:
1. Processes batches of images and text prompts
2. Applies forward diffusion process
3. Updates model parameters
4. Tracks training and validation loss

### Generation
Use the `show_result()` function to generate emojis:
```python
show_result(prompt="Apple", nb_image=9)
```

### DDPM Sampling
The denoising process uses the DDPM algorithm to gradually remove noise from random initialization to generate final images.

## Output
- Generated images are saved as GIF files showing the denoising process
- Loss plots are generated and saved as "plot.png"
- Model is saved as "diffusion-model" directory
- Compressed model file: "file.zip"

## File Structure
- `values.npy`: Processed image data
- `targets.npy`: Tokenized text prompts
- `diffusion-model/`: Saved model directory
- `file.zip`: Compressed model file
- `ddpm_0*.gif`: Generated animation files
- `plot.png`: Training loss plot

## Note
This implementation supports conditional generation using text prompts, where text embeddings are incorporated into the diffusion process through the U-Net model.