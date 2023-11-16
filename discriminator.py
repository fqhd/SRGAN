import torch
from torch import nn
from constants import *

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		vgg = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
		in_channels = 3
		batch_norm = True

		layers = []
		for x in vgg:
			if type(x) == int:
				layers.append(nn.Conv2d(in_channels, x, 3, 1, 1))
				if batch_norm:
					layers.append(nn.BatchNorm2d(x))
				layers.append(nn.ReLU(inplace=True))
				in_channels = x
			elif x == 'M':
				layers.append(
					nn.MaxPool2d(2, 2)
				)
		layers.append(nn.Flatten())
		layers.append(nn.Linear(4096, 512))
		layers.append(nn.ReLU())
		layers.append(nn.Dropout(0.5))
		layers.append(nn.Linear(512, 512))
		layers.append(nn.ReLU())
		layers.append(nn.Dropout(0.5))
		layers.append(nn.Linear(512, 1))
		layers.append(nn.Sigmoid())
		self.main = nn.Sequential(*layers)

	def forward(self, input):
		return self.main(input)

if __name__ == '__main__':
	x = torch.randn(32, 3, 128, 128)

	model = Discriminator()

	trainable_params = sum(
		p.numel() for p in model.parameters() if p.requires_grad
	)
	
	print('Trainable Params:', trainable_params)

	out = model(x)

	print(out.shape)