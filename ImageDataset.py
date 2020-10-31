from __future__ import print_function, division


import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import copy
from functools import cmp_to_key



def path_cmp(a, b):
    x = os.path.split(a[0])[1]
    y = os.path.split(b[0])[1]
    if x > y:
        return -1
    else:
        return 1
path_cmp_key = cmp_to_key(path_cmp)


def make_dataset(directory, class_to_idx, class_to_folders, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return datasets.folder.has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        for dir_name in class_to_folders[target_class]:
            target_dir = os.path.join(directory, dir_name)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
    return instances


class MyDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, detect_laying, to_sort=0, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(MyDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx,class_to_folders = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, class_to_folders, extensions=extensions, is_valid_file=is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        if to_sort < 0:
            self.samples.sort(key = path_cmp_key,reverse=True)
        elif to_sort > 0:
            self.samples.sort(key = path_cmp_key,reverse=False)


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        """
        if self.DETECT_LAY:
            classes = ['not_laying','laying']
            class_to_folders = {'not_laying':['not_laying'], 'laying':['not_nursing','nursing']}
        else:
            classes = ['nursing','not_nursing']
            class_to_folders = {'not_nursing':['not_nursing','not_laying'], 'nursing':['nursing']}
        
        folders = [d.name for d in os.scandir(dir) if d.is_dir()]
        #classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx, class_to_folders

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
    
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class TrainImageFolder(MyDatasetFolder): # https://discuss.pytorch.org/t/questions-about-imagefolder/774/11
    
    def __init__(self, root, detect_laying, to_sort=0,transform=None,loader=default_loader,is_valid_file=None):
        self.DETECT_LAY = detect_laying
        super(TrainImageFolder,self).__init__(root, loader, detect_laying, to_sort=to_sort,extensions=IMG_EXTENSIONS if is_valid_file is None else None,
                                              transform=transform)
        self.DETECT_LAY = detect_laying
        
