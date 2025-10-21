from dataset import dataloader, test_dl, dataset
from generator import *
from discriminator import *
from constants import *
from tqdm import tqdm
from torch import nn
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import time
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

print(f'Started training using device: {device}')

generator = ResidualAttentionUNet().to(device)
discriminator = Discriminator().to(device)

psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

d_opt = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
g_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))

loss_fn = nn.BCELoss()
recon_loss = nn.L1Loss()

fixed_noise = torch.randn(test_dl.batch_size, Z_DIM, device=device)
fixed_small_images, fixed_big_images = next(iter(test_dl))

plt.figure(figsize=(8, 8))
for i in range(16):
	plt.subplot(4, 4, i+1)
	plt.imshow(T.ToPILImage()(fixed_small_images[i]))
	plt.axis('off')
plt.show()

fixed_small_images = fixed_small_images.to(device)

g_losses = []
d_losses = []

start = time.time()
idx = 0
for epoch in range(EPOCHS):
	avg_d_loss = 0
	avg_g_loss = 0
	for small_batch, big_batch in tqdm(dataloader):
		small_batch = small_batch.to(device)
		big_batch = big_batch.to(device)
		b_size = small_batch.size(0)

		# Train on Feal
		discriminator.zero_grad()
		y_hat_real = discriminator(big_batch).view(-1)
		y_real = torch.ones_like(y_hat_real, device=device)
		real_loss = loss_fn(y_hat_real, y_real)
		real_loss.backward()

		# Train on Fake
		noise = torch.randn(b_size, Z_DIM, device=device)
		fake_images = generator(small_batch, noise)
		y_hat_fake = discriminator(fake_images.detach()).view(-1)
		y_fake = torch.zeros_like(y_hat_fake)
		fake_loss = loss_fn(y_hat_fake, y_fake)
		fake_loss.backward()
		d_opt.step()

		# Train generator
		generator.zero_grad()
		y_hat_fake = discriminator(fake_images)
		adversarial_loss = loss_fn(y_hat_fake, torch.ones_like(y_hat_fake))
		small_fake = dataset.small_resize(fake_images)
		reconstruction_loss = recon_loss(small_fake, small_batch)
		g_loss = 1e-3 * adversarial_loss + reconstruction_loss
		g_loss.backward()
		g_opt.step()

		avg_g_loss += adversarial_loss.item()
		avg_d_loss += (real_loss.item() + fake_loss.item()) / 2

		if idx % 10 == 0:
			avg_d_loss /= 10
			avg_g_loss /= 10
			d_losses.append(avg_d_loss)
			g_losses.append(avg_g_loss)
			avg_g_loss = 0
			avg_d_loss = 0
		idx += 1
	
	with torch.no_grad():
		predicted_images = generator(fixed_small_images, fixed_noise)[:16]

		real_images = fixed_big_images.to(device)

		psnr_val = psnr_metric(predicted_images, real_images).item()
		ssim_val = ssim_metric(predicted_images, real_images).item()
		
		print(f"[Epoch {epoch}] PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}")
	
	img = T.ToPILImage()(vutils.make_grid(predicted_images.to('cpu'), normalize=True, padding=2, nrow=4))
	img.save(f'progress/epoch_{epoch}.jpg')
	generator = generator.to('cpu')
	torch.save(generator.state_dict(), f'models/generator_epoch_{epoch}.pth')
	generator = generator.to(device)

plt.figure(figsize=(8, 5))
plt.plot(g_losses, label='G_Loss')
plt.plot(d_losses, label='D_Loss')
plt.title('Generator and Discriminator Loss')
plt.legend()
plt.show()

train_time = time.time() - start
print(f'Total training time: {train_time // 60} minutes')