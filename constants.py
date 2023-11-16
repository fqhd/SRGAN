import torch

Z_DIM = 128
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
BETA_1 = 0.5
BETA_2 = 0.999

device = (
	'cuda' if torch.cuda.is_available() else
	'mps' if torch.backends.mps.is_available() else
	'cpu' 
)