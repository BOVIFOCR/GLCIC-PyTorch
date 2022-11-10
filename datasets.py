
import os
import imghdr
import random
import torch.utils.data as data
from utils import gen_input_mask, read_mask
from torchvision.transforms.functional import crop
from torchvision.transforms import Resize
from PIL import Image
import torch

class ImageDataset(data.Dataset):
    def __init__(self, data_dir, input_size, train=True, phase=1, transform=None, recursive_search=False):
        super(ImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        self.imgpaths = self.__load_imgpaths_from_dir(self.data_dir, walk=recursive_search)
        self.input_size = input_size
        self.train = train
        self.phase = phase

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, index, color_format='RGB'):
        img_path = self.imgpaths[index]
        mask_path = img_path[:-4] + ".jpg.txt"
        
        img = Image.open(img_path)
        w, h = img.size
        img = img.convert(color_format)
        
        if self.transform is not None:
            img = self.transform(img)

        mask = read_mask(mask_path)
        if self.phase == 1:
            if self.train:
                mask_less, mask = gen_input_mask((1, w, h), mask, max_size=self.input_size)
            else:
                mask, mask_less = gen_input_mask((1, w, h), mask, max_size=self.input_size)
        else: # mask_less == fake, mask == real
            mask_less, mask = split_holes((1, w, h), mask)

        #mask = Resize((self.input_size, self.input_size))
        #mask_less = Resize((self.input_size, self.input_size))
        
        _, w, h = img.shape
        w_off = random.randint(0, abs(self.input_size - w))
        h_off = random.randint(0, abs(self.input_size - h))
        mask = crop(mask, w_off, h_off, self.input_size, self.input_size)
        mask_less = crop(mask_less, w_off, h_off, self.input_size, self.input_size)
        img = crop(img, w_off, h_off, self.input_size, self.input_size)

        return img, mask_less, mask

    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath) and imghdr.what(filepath):
            return True
        return False

    def __load_imgpaths_from_dir(self, dirpath, walk=False):
        imgpaths = []
        dirpath = os.path.expanduser(dirpath)
        if walk:
            for (root, _, files) in os.walk(dirpath):
                for file in files:
                    file = os.path.join(root, file)
                    if self.__is_imgfile(file):
                        imgpaths.append(file)
        else:
            for path in os.listdir(dirpath):
                path = os.path.join(dirpath, path)
                if not self.__is_imgfile(path):
                    continue
                imgpaths.append(path)
        return imgpaths
