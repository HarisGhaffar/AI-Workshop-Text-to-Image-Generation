### Text to Image Generation
!pip install -qq -U datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 ftfy==6.1.3 pyarrow==14.0.0
!pip install diffusers[training]==0.25.0
!pip install diffusers==0.25.0

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel, DDPMScheduler
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/ddpm-celebahq-256"
pipeline = DDPMPipeline.from_pretrained(model_id)
pipeline.to(device)

generated_image = pipeline().images[0]
generated_image


img = torchvision.transforms.ToTensor()(generated_image)
type(img)

img.shape

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

model = UNet2DModel.from_pretrained(model_id)
scheduler = DDIMScheduler.from_pretrained(model_id)
scheduler.set_timesteps(num_inference_steps=50)
scheduler.timesteps

torch.randn?

torch.manual_seed(1)

image = torch.randn(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
)
image.shape

show_images(image)

with torch.no_grad():
    noise_prediction = model(sample=image, timestep=980).sample

noise_prediction.shape

scheduler.step?

scheduler_output = scheduler.step(
    model_output=noise_prediction, timestep=960, sample=image
)

image = scheduler_output.prev_sample

image.shape

show_images(image)

#if we combine all of this, and this time just to save time, using a diffusion model with 50 steps

import tqdm
import PIL

model = UNet2DModel.from_pretrained(model_id)
scheduler = DDIMScheduler.from_pretrained(model_id)
scheduler.set_timesteps(num_inference_steps=50)
scheduler.timesteps

# We are starting off with a batch on just one image
image = torch.randn(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
)

model.to("cuda")
image = image.to("cuda")

for index, timestep in enumerate(tqdm.tqdm(scheduler.timesteps)):
  with torch.no_grad():
      noise_prediction = model(sample=image, timestep=timestep).sample

  scheduler_output = scheduler.step(
      model_output=noise_prediction, timestep=timestep, sample=image
  )

  image = scheduler_output.prev_sample

  if (index + 1) % 10 == 0:
      display_sample(image, index + 1)





## Scheduler

!pip install -qq -U datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 ftfy==6.1.3 pyarrow==14.0.0
!pip install diffusers[training]==0.25.0
!pip install diffusers==0.25.0

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel, DDPMScheduler
import torch.nn.functional as F

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

image_size = 32
batch_size = 64

preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

batch = next(iter(train_dataloader))["images"].to(device)[:8]
print("X shape:", batch.shape)
show_images(batch).resize((8 * 64, 64), resample=Image.NEAREST)

DDPMScheduler?

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise_scheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.005)
original_image = noise_scheduler.alphas_cumprod.cpu() ** 0.5
noise = (1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5

plt.title("Low values for beta_end")
plt.plot(original_image, label="Component of original image")
plt.plot(noise, label="Component of noise")
plt.legend(fontsize="medium", loc='center right')

timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(batch)
noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)
print("Noisy X shape", noisy_batch.shape)
show_images(noisy_batch).resize((8 * 64, 64), resample=Image.NEAREST)

#High values for beta_end
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.05)
original_image = noise_scheduler.alphas_cumprod.cpu() ** 0.5
noise = (1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5

plt.title("High values for beta_end")
plt.plot(original_image, label="Component of original image")
plt.plot(noise, label="Component of noise")
plt.legend(fontsize="medium", loc='center right')

timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(batch)
noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)
print("Noisy X shape", noisy_batch.shape)
show_images(noisy_batch).resize((8 * 64, 64), resample=Image.NEAREST)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
original_image = noise_scheduler.alphas_cumprod.cpu() ** 0.5
noise = (1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5

plt.title("Cosine scheduler")
plt.plot(original_image, label="Component of original image")
plt.plot(noise, label="Component of noise")
plt.legend(fontsize="medium", loc='center right')

timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(batch)
noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)
print("Noisy X shape", noisy_batch.shape)
show_images(noisy_batch).resize((8 * 64, 64), resample=Image.NEAREST)

#back to default
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

## U-Net Model

!pip install -qq -U datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 ftfy==6.1.3 pyarrow==14.0.0
!pip install diffusers[training]==0.25.0
!pip install diffusers==0.25.0

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel, DDPMScheduler
import torch.nn.functional as F

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

image_size = 32
batch_size = 64

preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

batch = next(iter(train_dataloader))["images"].to(device)[:8]
#print("X shape:", batch.shape)
#show_images(batch).resize((8 * 64, 64), resample=Image.NEAREST)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
original_image = noise_scheduler.alphas_cumprod.cpu() ** 0.5
noise = (1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5
timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(batch)
noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
model.to(device)

noisy_batch.shape

with torch.no_grad():
    model_prediction = model(noisy_batch, timesteps).sample

assert noisy_batch.shape == model_prediction.shape
print(f"Images are the same shape: {model_prediction.shape}")

## Train a model

!pip install -qq -U datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 ftfy==6.1.3 pyarrow==14.0.0
!pip install diffusers[training]==0.25.0
!pip install diffusers==0.25.0

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel, DDPMScheduler
import torch.nn.functional as F

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
model.to(device)

dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

image_size = 32 # smaller to reduce training time
batch_size = 64

preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

#build up to the for loop
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

batch = next(iter(train_dataloader))
print(batch)

batch["images"].shape

clean_images = batch["images"].to(device)
clean_images.shape

noise = torch.randn(clean_images.shape).to(device)
noise.shape

batch_size = clean_images.shape[0]
timesteps = torch.randint(
            low=0, high=noise_scheduler.num_train_timesteps, size=(batch_size,), device=device).long()
timesteps

noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
noisy_images.shape

lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

num_epochs = 20
losses = []

for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bs,), device=device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 2 == 0:
        loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
        print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

image_pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
sample_image = image_pipeline()
sample_image.images[0]

image_pipeline.save_pretrained("DDPM_pipeline")

!ls -la DDPM_pipeline/

!cat DDPM_pipeline/model_index.json

!ls -la DDPM_pipeline/unet

!cat DDPM_pipeline/unet/config.json

!ls -la DDPM_pipeline/scheduler

!cat DDPM_pipeline/scheduler/scheduler_config.json

## Challenge

- try another dataset? else butterfly with image sizes 64

!pip install -qq -U datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 ftfy==6.1.3 pyarrow==14.0.0
!pip install diffusers[training]==0.25.0
!pip install diffusers==0.25.0

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel, DDPMScheduler
import torch.nn.functional as F

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

image_size = 64 # smaller to reduce training time
batch_size = 64

preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
model.to(device)

num_epochs = 20
losses = []

for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bs,), device=device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 2 == 0:
        loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
        print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

## Challenge

- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

from datasets import load_dataset
cifar10 = load_dataset("cifar10")





## Solution

!pip install -qq -U datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 ftfy==6.1.3 pyarrow==14.0.0
!pip install diffusers[training]==0.25.0
!pip install diffusers==0.25.0

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel, DDPMScheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datasets import load_dataset
cifar10 = load_dataset("cifar10")

preprocess = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#        transforms.Pad(2),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image) for image in examples["img"]]
    return {"images": images, "labels": examples["label"]}


train_dataset = cifar10["train"].with_transform(transform)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True
)

cifar10

sample_size = 32

model = UNet2DModel(
    in_channels=3,  # 1 channel for grayscale images
    out_channels=3,
    sample_size=32,
    block_out_channels=(32, 64, 128, 256),
    num_class_embeds=10,  # Enable class conditioning
)


scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02
)
timesteps = torch.linspace(0, 999, 8).long()
batch = next(iter(train_dataloader))
x = batch["images"][0].expand([8, 3, sample_size, sample_size])
noise = torch.rand_like(x)
noised_x = scheduler.add_noise(x, noise, timesteps)
show_images((noised_x * 0.5 + 0.5).clip(0, 1))

from torch.nn import functional as F
from tqdm import tqdm

num_epochs = 3
lr = 3e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02
)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)
losses = []

for epoch in (tqdm(range(num_epochs))):
    for step, batch in (inner := tqdm(enumerate(train_dataloader), position=0, leave=True, total=len(train_dataloader))):

        clean_images = batch["images"].to(device)
        class_labels = batch["labels"].to(device)
        batch_size = clean_images.shape[0]

        noise = torch.randn(clean_images.shape).to(device)

        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (batch_size,), device=device
        ).long()

        noisy_images = scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, class_labels=class_labels, return_dict=False)[0]

        loss = F.mse_loss(noise_pred, noise)

        inner.set_postfix(loss=f"{loss.cpu().item():.3f}")

        losses.append(loss.item())

        loss.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

def generate_from_dataset(class_to_generate=0, n_samples=8):
    sample = torch.randn(n_samples, 3, 32, 32).to(device)
    class_labels = [class_to_generate] * n_samples
    class_labels = torch.tensor(class_labels).to(device)

    for _, t in tqdm(enumerate(scheduler.timesteps)):
        with torch.no_grad():
            noise_pred = model(sample, t, class_labels=class_labels).sample

        sample = scheduler.step(noise_pred, t, sample).prev_sample

    return sample.clip(-1, 1) * 0.5 + 0.5

images = generate_from_dataset(class_to_generate=6)
show_images(images)

| Label | Description |
|-------|-------------|
| 0     | airplane    |
| 1     | automobile  |
| 2     | bird        |
| 3     | cat         |
| 4     | deer        |
| 5     | dog         |
| 6     | frog        |
| 7     | horse       |
| 8     | ship        |
| 9     | truck       |




## Making improvements - Latent Diffusion

!pip install -qq -U datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 ftfy==6.1.3 pyarrow==14.0.0
!pip install diffusers[training]==0.25.0
!pip install diffusers==0.25.0

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel, DDPMScheduler
from diffusers import AutoencoderKL, StableDiffusionPipeline
import torch.nn.functional as F
from torch.utils.data import DataLoader
import requests

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url = "https://github.com/jonfernandes/images/raw/main/boat.png"

def download_image(url):
  return Image.open(requests.get(url, stream=True).raw).convert("RGB")

input_image = download_image(url).resize((512, 512))
input_image

- We want to go from image to latent so 3x512x512 to 1x4x64x64

transforms.ToTensor()(input_image).shape

transforms.ToTensor()(input_image).unsqueeze(0).shape

four_channels = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
vae.config

def image_to_latent(input_im):
    with torch.no_grad():
        latent = vae.encode(transforms.ToTensor()(input_im).unsqueeze(0).to(device)*2-1)
    return 0.18215 * latent.latent_dist.sample()

encoded = image_to_latent(input_image)
encoded.shape

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for c in range(4):
    axs[c].imshow(encoded[0][c].cpu(), cmap='Greys')

def latents_to_images(latents):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    all_images = [Image.fromarray(image) for image in images]
    return all_images

print(f"Latents dimension: {encoded.shape}")
latents_to_images(encoded)[0]

#original image
input_image

## Text encoder - CLIP Model

!pip install -qq -U datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 ftfy==6.1.3 pyarrow==14.0.0
!pip install diffusers[training]==0.25.0
!pip install diffusers==0.25.0

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel, DDPMScheduler
from diffusers import AutoencoderKL, StableDiffusionPipeline
import torch.nn.functional as F
from torch.utils.data import DataLoader
import requests
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url = "https://github.com/jonfernandes/images/raw/main/boat.png"

def download_image(url):
  return Image.open(requests.get(url, stream=True).raw).convert("RGB")

input_image = download_image(url).resize((512, 512))
input_image

model_id = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(model_id)
tokenizer = CLIPProcessor.from_pretrained(model_id)

photos = ["a photo of a boat", "a photo of a dolphin"]
inp = tokenizer(["a photo of a boat", "a photo of a dolphin"], images=input_image, return_tensors="pt")
inp

outp = model(**inp)
outp

outp.logits_per_image

torch.nn.functional.softmax(outp.logits_per_image, dim=-1)

photo = ["a photo of a boat", "a photo of a dolphin"]

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

tokenizer.vocab_size

tokenizer.get_vocab()

tokenizer.special_tokens_map

tokenizer.bos_token_id

tokenizer.eos_token_id

tokenizer.pad_token_id

prompt = "boat on the sea"
inp = tokenizer(prompt)

tokenizer.convert_ids_to_tokens(inp["input_ids"])

inp

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
text_encoder

text_encoder.text_model.embeddings

text_encoder.text_model.embeddings.token_embedding

tokenizer("boat")

text_encoder.text_model.embeddings.token_embedding(torch.tensor(4440, device=device))

text_encoder.text_model.embeddings.token_embedding(torch.tensor(4440, device=device)).shape

text_encoder.text_model.embeddings.token_embedding(torch.tensor(inp["input_ids"], device=device)).shape

# 77 is the maximum number of tokens in the text input
text_encoder.text_model.embeddings.position_embedding

## Putting the components together using Stable Diffusion

!pip install -qq -U datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 ftfy==6.1.3 pyarrow==14.0.0
!pip install diffusers[training]==0.25.0
!pip install diffusers==0.25.0

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel, DDPMScheduler
from diffusers import AutoencoderKL, StableDiffusionPipeline
import torch.nn.functional as F
from torch.utils.data import DataLoader
import requests
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16
).to(device)
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)


images = []
prompt = "A boat on the sea"
for guidance_scale in [1, 2, 4, 12]:
    torch.manual_seed(0)
    image = pipeline(prompt, guidance_scale=guidance_scale).images[0]
    images.append(image)

fig, axs = plt.subplots(1, 4, figsize=(12, 6))

for i in range(4):
  axs[i].imshow(images[i])
  axs[i].axis('off')

plt.tight_layout()

prompt = [
    "A boat on the sea"
]
height = 512
width = 512
num_inference_steps = 30
guidance_scale = 7.5
seed = 1

text_input = pipeline.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipeline.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

unconditioned_input = pipeline.tokenizer(
    "",
    padding="max_length",
    max_length=pipeline.tokenizer.model_max_length,
    return_tensors="pt",
)

with torch.no_grad():
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0]
    unconditioned_embeddings = pipeline.text_encoder(unconditioned_input.input_ids.to(device))[0]

text_embeddings = torch.cat([unconditioned_embeddings, text_embeddings])

pipeline.scheduler.set_timesteps(num_inference_steps)

latents = (
    torch.randn((1, pipeline.unet.config.in_channels, height // 8, width // 8)).to(device).half()
)
latents = latents * pipeline.scheduler.init_noise_sigma

for i, timestep in enumerate(pipeline.scheduler.timesteps):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timestep)

    with torch.no_grad():
        noise_pred = pipeline.unet(
            latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    latents = pipeline.scheduler.step(noise_pred, timestep, latents).prev_sample

latents = 1 / vae.config.scaling_factor * latents
with torch.no_grad():
    image = vae.decode(latents).sample
image = (image / 2 + 0.5).clamp(0, 1)

show_images(image[0])
