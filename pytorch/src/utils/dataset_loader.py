import torch
import random
import torchvision

from torch.utils.data import DataLoader, Subset
from PIL import Image
from collections import defaultdict

from src.models import common as common
from src import definitions as defs
from torchvision import datasets, transforms
from torchvision.datasets import ImageNet #, ImageFolder

#import torchvision.datasets as tv_datasets
#import torchvision.transforms as tv_transforms

dataset = None
dataset_len = 0


# this is for the I-VIT models
#def build_transform(is_train, args):
def build_transform():
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from torchvision import transforms

    #resize_im = args.input_size > 32
    resize_im = True
    """
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform
    """
    t = []
    if resize_im:
        input_size = 224

        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def load_dataset_imagenet(batch_size=64):
    global dataset
    global dataset_len

    if defs.VIT: # using the same transform source code as the one in I-VIT
        transform = build_transform() 
    
    else:  # original transform for the CNNs
        # Pre-process the inputs: see https://docs.pytorch.org/vision/main/models.html
        weights_work = common.QUANTIZED_MODELS[defs.MODEL_NAME][0]
        transform = weights_work.transforms()
        
    """
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    """

    # Load the training and validation datasets
    #train_dataset = ImageNet(root=defs.PATH_IMAGENET, split`='train', transform=transform_train)
    #dataset = ImageNet(root=defs.PATH_IMAGENET, split='val', transform=transform)
    dataset = torchvision.datasets.ImageFolder(root=f"{defs.PATH_IMAGENET}/val", transform=transform)

    # Define data loaders
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4)

    #dataset = dataloader.dataset
    dataset_len = len(dataset)

    #print(f"Imagenet validation set size: {dataset_len}") # Imagenet validation set size: 50000
    # Example of iterating over the data loader
    #for images, labels in dl_imagenet_val:
        #pass

    # Class ID to list of sample indices
    """
    class_to_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        print(f"list {label} <- append ({idx})")
        class_to_indices[label].append(idx)

    print(class_to_indices)
    exit(0)
    """

    return dataloader


def load_dataset_mnist(normalize=True):
    # Define transform: convert to tensor and normalize to mean=0.1307, std=0.3081

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    else: # Sobel tests. DO NOT NORMALIZE the inputs
        transform = transforms.ToTensor()

    # Load training and test datasets
    #train_dataset = datasets.MNIST(root=defs.PATH_MNIST, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(
        root=defs.PATH_MNIST, 
        train=False, 
        download=True, 
        transform=transform)

    # Create data loaders
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return test_loader


def get_input_batch(indices):
    # Create a subset of the dataset
    subset = Subset(dataset, indices)

    # Create a DataLoader for the subset (batch_size=len(indices) to load all at once)
    subset_loader = DataLoader(subset, batch_size=len(indices), shuffle=False)

    # Get the batch (only one iteration needed)
    images, labels = next(iter(subset_loader))

    return images, labels


# returns a single input tensor + expected label
def get_input (idx):
    img, label = dataset[idx]  # (C,H,W)
    ten = img.unsqueeze(0)  # unsqueeze transforma para # (N,C,H,W)

    # Get the class name by looking up the index in `class_to_idx`
    """
    file_path, _ = dataset.samples[idx]

    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse the mapping
    class_name = idx_to_class[label_idx]
    
    print("Class index/name:", label_idx, class_name)
    print (f"The file is {file_path}")     # use xdg-open path/to/your/image.jpg 
    """
    return ten, label


def get_input_path(idx):
    file_path, _ = dataset.samples[idx]
    return file_path