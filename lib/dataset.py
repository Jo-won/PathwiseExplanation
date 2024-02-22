from torch.utils.data import DataLoader
from torchvision import datasets

from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from lib.func import *


import os.path
import pickle
import torch
import torchvision 
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


class CIFAR10_custom(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        self.paths = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])
                self.paths.extend(entry["filenames"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, path = self.data[index], self.targets[index], self.paths[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
    

    
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def parse_gt_dict(gt_csv_path):
    gt_csv = pd.read_csv(gt_csv_path)
    gt_arr = np.asarray(gt_csv)

    name2gtidx = dict()
    for i in gt_arr:
        name2gtidx.update({i[0]: {"idx": i[1], "label": i[2]}})
            
    metaid2gtidx = dict()
    for i in gt_arr:
        metaid2gtidx.update({i[2]: {"idx": i[1], "label": i[2]}})

    return name2gtidx, metaid2gtidx

def parse_dataset(args):

    invTrans, test_loader, classes, \
        name2gtidx, folder_to_class, class_number, class_to_idx = \
            None, None, None, None, None, None, None


    if args.dataset == "cifar10":
        cifar_norm_mean = (0.4914, 0.4822, 0.4465)
        cifar_norm_std = (0.2023, 0.1994, 0.2010)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_norm_mean, cifar_norm_std),
        ])

        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = (1/np.array(cifar_norm_std)).tolist()),
                                    transforms.Normalize(mean = (-np.array(cifar_norm_mean)).tolist(),
                                                        std = [ 1., 1., 1. ]),
                                ])

        testset = CIFAR10_custom(
            root='pytorch_cifar/data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=4)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')
      
        name2gtidx = None
    elif args.dataset == "imagenet_select":
    
        folder_to_class = {"n02119022": "red fox",
                        "n02109961":"husky",
                        "n04037443": "racer/race car",
                        "n03272562": "electric locomotive",
                        "n02236044": "mantis",
                        "n02256656": "cicada",
                        "n02980441": "castle",
                        "n04005630": "prison",
                        "n07720875": "bell pepper",
                        "n07716358": "zucchini"
        }

        data_path = "./ILSVRC2012_selected_224/"
        class_number = 10
        name2gtidx = None

        imagenet_norm_mean = [0.485, 0.456, 0.406]
        imagenet_norm_std = [0.229, 0.224, 0.225]

        image_transforms = {
            'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_norm_mean, imagenet_norm_std)
            ]),
        }

        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = (1/np.array(imagenet_norm_std)).tolist()),
                                    transforms.Normalize(mean = (-np.array(imagenet_norm_mean)).tolist(),
                                                        std = [ 1., 1., 1. ]),
                                ])
        
        
        test_data_set = ImageFolderWithPaths(root=data_path+"val/", transform=image_transforms['test'])
        class_to_idx = test_data_set.class_to_idx

        test_loader = DataLoader(test_data_set, batch_size=1, shuffle=False, num_workers=4)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')


    return invTrans, test_loader, classes, \
        name2gtidx, folder_to_class, class_number, class_to_idx