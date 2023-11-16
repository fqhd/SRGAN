import torch
from torch import nn
from constants import *
from modules import *

class ResidualAttentionUNet(nn.Module):
	def __init__(self):
		super().__init__()
		# self.project = nn.Linear(Z_DIM, 32*32)

		kernel_size = 3
		dropout = 0
		batch_norm = True
		n_features = 32

		self.pool = nn.MaxPool2d(2, 2)

		self.c32 = ResConvBlock(3, n_features * 4, kernel_size, dropout, batch_norm)
		self.c16 = ResConvBlock(n_features * 4, n_features * 8, kernel_size, dropout, batch_norm)
		self.c8 = ResConvBlock(n_features * 8, n_features * 16, kernel_size, dropout, batch_norm)

		self.u16 = ConvBlock(n_features * 16, n_features * 8, kernel_size, dropout, batch_norm)
		self.u32 = ConvBlock(n_features * 8, n_features * 4, kernel_size, dropout, batch_norm)
		self.u64 = ConvBlock(n_features * 2, n_features * 2, kernel_size, dropout, batch_norm)
		self.u128 = ConvBlock(n_features, n_features, kernel_size, dropout, batch_norm)

		self.att1 = AttentionBlock(n_features * 8, n_features * 8, n_features * 4, batch_norm)
		self.att2 = AttentionBlock(n_features * 4, n_features * 4, n_features * 2, batch_norm)

		self.u1 = up_conv(n_features * 16, n_features * 8, batch_norm)
		self.u2 = up_conv(n_features * 8, n_features * 4, batch_norm)
		self.u3 = up_conv(n_features * 4, n_features * 2, batch_norm)
		self.u4 = up_conv(n_features * 2, n_features, batch_norm)

		self.final = nn.Conv2d(n_features, 3, 3, 1, 1)
		
	def forward(self, x, z):
		"""
		b_size = x.shape[0]
		cond = self.project(z)
		cond = torch.reshape(cond, (b_size, 1, 32, 32))
		x = torch.concat((cond, x), dim=1)
		"""

		c3 = self.c32(x)
		c4 = self.c16(self.pool(c3))
		c5 = self.c8(self.pool(c4))

		g1 = self.u1(c5)
		s1 = self.att1(g1, c4)
		c6 = self.u16(torch.concat((s1, g1), dim=1))

		g2 = self.u2(c6)
		s2 = self.att2(g2, c3)
		c7 = self.u32(torch.concat((s2, g2), dim=1))

		c8 = self.u64(self.u3(c7))
		c9 = self.u128(self.u4(c8))

		final = self.final(c9)
		return torch.sigmoid(final)

if __name__ == '__main__':
	model = ResidualAttentionUNet()

	x = torch.randn(BATCH_SIZE, 3, 32, 32)
	z = torch.randn(BATCH_SIZE, Z_DIM)

	trainable_params = sum(
		p.numel() for p in model.parameters() if p.requires_grad
	)

	print('Trainable Params:', trainable_params)

	images = model(x, z)

	print(images.shape)
