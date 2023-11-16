import torch
from torch import nn

class up_conv(nn.Module):
	def __init__(self,ch_in,ch_out, batch_norm):
		super(up_conv,self).__init__()
		if batch_norm:
			self.up = nn.Sequential(
				nn.Upsample(scale_factor=2),
				nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
				nn.BatchNorm2d(ch_out),
				nn.ReLU(inplace=True)
			)
		else:
			self.up = nn.Sequential(
				nn.Upsample(scale_factor=2),
				nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
				nn.ReLU(inplace=True)
			)

	def forward(self,x):
		x = self.up(x)
		return x

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dropout, batch_norm):
		super().__init__()
		self.stack = []

		self.stack.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2))
		if batch_norm:
			self.stack.append(nn.BatchNorm2d(out_channels))
		self.stack.append(nn.ReLU())

		self.stack.append(nn.Conv2d(out_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2))
		if batch_norm:
			self.stack.append(nn.BatchNorm2d(out_channels))
		self.stack.append(nn.ReLU())

		if dropout > 0:
			self.stack.append(nn.Dropout(dropout))

		self.stack = nn.Sequential(*self.stack)
	
	def forward(self, x):
		return self.stack(x)
      
class ResConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dropout, batch_norm):
		super().__init__()
		self.stack = []

		self.stack.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2))
		if batch_norm:
			self.stack.append(nn.BatchNorm2d(out_channels))
		self.stack.append(nn.ReLU())

		self.stack.append(nn.Conv2d(out_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2))
		if batch_norm:
			self.stack.append(nn.BatchNorm2d(out_channels))

		if dropout > 0:
			self.stack.append(nn.Dropout(dropout))

		self.stack = nn.Sequential(*self.stack)

		if batch_norm:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1, 1, 0),
				nn.BatchNorm2d(out_channels)
			)
		else:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1, 1, 0)
			)
	
	def forward(self, x):
		conv = self.stack(x)
		residual = self.shortcut(x)
		return torch.relu(conv + residual)
	
class AttentionBlock(nn.Module):
	def __init__(self,F_g,F_l,F_int,batch_norm):
		super().__init__()
		if batch_norm:
			self.W_g = nn.Sequential(
				nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
				nn.BatchNorm2d(F_int)
			)
			
			self.W_x = nn.Sequential(
				nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
				nn.BatchNorm2d(F_int)
			)

			self.psi = nn.Sequential(
				nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
				nn.BatchNorm2d(1),
				nn.Sigmoid()
			)
		else:
			self.W_g = nn.Sequential(
				nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True)
			)
			
			self.W_x = nn.Sequential(
				nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True)
			)

			self.psi = nn.Sequential(
				nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
				nn.Sigmoid()
			)
		
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self,g,x):
		g1 = self.W_g(g)
		x1 = self.W_x(x)
		psi = self.relu(g1+x1)
		psi = self.psi(psi)

		return x*psi
