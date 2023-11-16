import torch
import torchvision.transforms as T
from torchvision.io import read_image
import os

generator = torch.load('models/generator_v1.pkl')
generator.eval()

for n in os.listdir('in'):
	img = read_image(f'in/{n}')
	img = img[:3] / 255.0
	img = torch.unsqueeze(img, 0)
	noise = torch.randn(1, 128)
	with torch.no_grad():
		prediction = generator(img, noise)
	out = T.ToPILImage()(prediction[0])
	out.save(f'out/{n}')
