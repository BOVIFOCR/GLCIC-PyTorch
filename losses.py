from torch.nn.functional import mse_loss
import pytorch_ssim

def completion_network_loss(input, output, mask_less, mask):
    mask[mask_less == 1] = 0.0
    return mse_loss(output * mask, input * mask)


def ssim_loss(input, output, mask_less, mask):
    mask[mask_less == 1] = 0.0
    return 1 - pytorch_ssim.ssim(output * mask_less, input * mask)
