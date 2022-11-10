import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

def read_mask(mask_path):
    bboxes = []
    with open(mask_path, 'r') as fd:
        lines = fd.readlines()
    for line in lines:
        line = line.strip('\n')
        bbox = [int(x) for x in line.split(',')]
        bboxes.append(bbox)
    return bboxes

def gen_input_mask(
        shape, bboxes, hole_area=None, max_holes=1, max_size=512):
    """
    * inputs:
        - shape (sequence, required):
                Shape of a mask tensor to be generated.
                A sequence of length 4 (N, C, H, W) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 4 is provided,
                holes of size (W, H) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                ) are generated.
                All the pixel values within holes are filled with 1.0.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (X, Y) of the area,
                while hole_area[1] is its width and height (W, H).
                This area is used as the input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The number of holes is randomly chosen from [1, max_holes].
                The default value is 1.
    * returns:
            A mask tensor of shape [N, C, H, W] with holes.
            All the pixel values within holes are filled with 1.0,
            while the other pixel values are zeros.
    """
    #mask = torch.zeros(shape)
    #mask_loss = torch.zeros(shape)
    bsize, mask_h, mask_w = shape
    
    if mask_w > mask_h:
        max_h = max_size
        max_w = int(mask_w*max_size/mask_h)
    else: 
        max_w = max_size
        max_h = int(mask_h*max_size/mask_w)
    h_factor = max_h / mask_h
    w_factor = max_w / mask_w
    
    mask = torch.zeros((shape[0], max_w, max_h))
    mask_loss = torch.zeros((shape[0], max_w, max_h))
 
    #h_factor = w_factor = 1   
    for i in range(bsize):
        for idx,bbox in enumerate(bboxes):
            bbox = [int(bbox[0]*w_factor), int(bbox[1]*h_factor), 
                    int((bbox[0]+bbox[2])*w_factor), int((bbox[1]+bbox[3])*h_factor)]
            if idx % 2 == 0:
                mask[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0

            else:
                mask_loss[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
    
    #mask_loss[mask == 1] = 0.0

    """for i in range(bsize):
        n_holes = random.choice(list(range(1, max_holes+1)))
        for _ in range(n_holes):
            # choose patch width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_w = hole_size[0]

            # choose patch height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_h = hole_size[1]

            # choose offset upper-left coordinate
            if hole_area:
                harea_xmin, harea_ymin = hole_area[0]
                harea_w, harea_h = hole_area[1]
                offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
            else:
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = random.randint(0, mask_h - hole_h)
            mask[i, :, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = 1.0"""
    return mask, mask_loss


def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))

def split_holes(shape, mask):
    fake = np.array(shape)
    real = np.array(shape)

    regs = []
    for i, reg in mask:
        if i % 2 == 0:
            regs.append([reg])
        else:
            regs[-1].append(reg)

    random.shuffle(regs)
    fake_regs = regs[0]
    real_regs = regs[1]

    for i, reg in fake_regs:
        bbox = [int(bbox[0]*w_factor), int(bbox[1]*h_factor),
            int((bbox[0]+bbox[2])*w_factor), int((bbox[1]+bbox[3])*h_factor)] 
        if i % 2 == 0:
            fake[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
        else:
            fake[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0.0

    for i, reg in real_regs:
        bbox = [int(bbox[0]*w_factor), int(bbox[1]*h_factor),
            int((bbox[0]+bbox[2])*w_factor), int((bbox[1]+bbox[3])*h_factor)]        
        if i % 2 == 0:
            real[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
        else:
            real[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0.0

    return fake, real

def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (torch.Tensor, required)
                Same shape as x, acts as a binary mask.

        X area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    idx = area.nonzero()
    x_min = idx[:, 0].min()
    x_max = idx[:, 0].max()
    y_min = idx[:, 1].min()
    y_max = idx[:, 1].max()
    return x[:, :, x_min:x_max+1, y_min:y_max+1]
    
    #xmin, ymin = area[0]
    #w, h = area[1]
    #return x[:, :, ymin: ymin + h, xmin: xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    mask_batch = []
    mask_less_batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        tpl = dataset[index]
        x1 = torch.unsqueeze(tpl[0], dim=0)
        x2 = torch.unsqueeze(tpl[1], dim=0)
        x3 = torch.unsqueeze(tpl[2], dim=0)
        batch.append(x1)
        mask_less_batch.append(x2)
        mask_batch.append(x3)
    
    return torch.cat(batch, dim=0), torch.cat(mask_less_batch, dim=0), torch.cat(mask_batch, dim=0)


def poisson_blend(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network, whose shape = (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor of Completion Network, whose shape = (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
    * returns:
                Output image tensor of shape (N, 3, H, W) inpainted with poisson image editing method.
    """
    input = input.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format
    num_samples = input.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(input[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        # compute mask's center
        xs, ys = [], []
        for j in range(msk.shape[0]):
            for k in range(msk.shape[1]):
                if msk[j, k, 0] == 255:
                    ys.append(j)
                    xs.append(k)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        dstimg = cv2.inpaint(dstimg, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret
