import torch
from lpips import LPIPS
from tqdm import tqdm
from generator import ResidualAttentionUNet
from dataset import test_dl

generator = ResidualAttentionUNet()
generator.load_state_dict(torch.load('models/generator.pth', weights_only=True))
generator.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

generator = generator.to(device)

lpips_loss = LPIPS(net='vgg').to(device)

total_lpips = 0.0
count = 0

with torch.no_grad():
    for small_batch, big_batch in tqdm(test_dl):
        small_batch = small_batch.to(device)
        big_batch = big_batch.to(device)

        b_size = small_batch.size(0)
        fake_images = generator(small_batch, None)

        lp = lpips_loss(fake_images, big_batch).mean().item()

        total_lpips += lp
        count += 1

avg_lpips = total_lpips / count
print(f"Average LPIPS over dataset: {avg_lpips:.4f}")
