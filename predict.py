
from torchvision.transforms.functional import crop

import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from models import CompletionNetwork
from utils import poisson_blend, read_mask, gen_input_mask
import random
import pytorch_ssim

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_img')
parser.add_argument('--mask_img', default=None)
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)


def main(args):

    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.input_img = os.path.expanduser(args.input_img)
    if args.mask_img is not None:
        args.mask_img = os.path.expanduser(args.mask_img)
    else:
        args.mask_img = None
    args.output_img = os.path.expanduser(args.output_img)

    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    img = Image.open(args.input_img)
    w, h = img.size
    img = transforms.Resize(args.img_size)(img)
    #img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)

    # create mask
    if args.mask_img is None:
        mask, _ = gen_input_mask(
            shape=(1, x.shape[2], x.shape[3]),
            bboxes=[[778, 512, 491, 33], [778, 512, 491, 33]],
            max_holes=args.max_holes,
        )
    else:
        mask = read_mask(args.mask_img)
        mask, mask_less = gen_input_mask((1, w, h), mask)

        mask = torch.unsqueeze(mask, dim=0) 
        mask_less = torch.unsqueeze(mask_less, dim=0) 
    # inpaint
    input_size = 512
    _, _, w, h = x.shape
    w_off = random.randint(0, abs(512 - w))
    h_off = random.randint(0, abs(512 - h))
    mask = crop(mask, w_off, h_off, 512, 512)
    mask_less = crop(mask_less, w_off, h_off, input_size, input_size)
    x = crop(x, w_off, h_off, input_size, input_size)



    model.eval()
    with torch.no_grad():
        #print(x.shape, mask.shape, mpv.shape)
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        #print(x_mask.shape, input.shape)
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)
        
        total = torch.numel(mask)
        n_sig = torch.count_nonzero(mask)
        n_blanks = total - n_sig

        metric = pytorch_ssim.ssim
        mask_less[mask == 1] = 0
        mm = metric(mask_less*x, mask_less*output).item()
        print(((mm*total - n_blanks)/n_sig).item())

        imgs = torch.cat((x, x_mask, inpainted), dim=0)
        save_image(imgs, args.output_img, nrow=3)
    #print('output img was saved as %s.' % args.output_img)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
