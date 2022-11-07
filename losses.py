from torch.nn.functional import mse_loss


def completion_network_loss(input, output, mask_less, mask):
    mask[mask_less == 1] = 0.0
    return mse_loss(output * mask, input * mask)
