import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)


class ZeroImageDataset(Dataset):
    def __init__(self, n_channels, h, w, n_samples, transform=None):
        self.n_channels = n_channels
        self.h = h
        self.w = w
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.ones(self.n_channels, self.h, self.w),
            torch.zeros(self.n_channels, self.h, self.w),
        )


# ============================================ #
class Sampler:
    def __init__(self, device="cpu"):
        self.device = device

    def sample(self, size=5):
        pass


# ====================== Paired Guided ====================== #
def paired_random_hflip(im1, im2):
    if np.random.rand() < 0.5:
        im1 = im1.transpose(Image.FLIP_LEFT_RIGHT)
        im2 = im2.transpose(Image.FLIP_LEFT_RIGHT)
    return im1, im2


def paired_random_vflip(im1, im2):
    if np.random.rand() < 0.5:
        im1 = im1.transpose(Image.FLIP_TOP_BOTTOM)
        im2 = im2.transpose(Image.FLIP_TOP_BOTTOM)
    return im1, im2


def paired_random_rotate(im1, im2):
    angle = np.random.rand() * 360
    im1 = im1.rotate(angle, fillcolor=(255, 255, 255))
    im2 = im2.rotate(angle, fillcolor=(255, 255, 255))
    return im1, im2


def paired_random_crop(im1, im2, size):
    assert im1.size == im2.size, "Images must have exactly the same size"
    assert size[0] <= im1.size[0]
    assert size[1] <= im1.size[1]

    x1 = np.random.randint(im1.size[0] - size[0])
    y1 = np.random.randint(im1.size[1] - size[1])

    im1 = im1.crop((x1, y1, x1 + size[0], y1 + size[1]))
    im2 = im2.crop((x1, y1, x1 + size[0], y1 + size[1]))

    return im1, im2


class PairedDataset(Dataset):
    def __init__(
        self,
        data_folder,
        labels_folder,
        transform=None,
        reverse=False,
        hflip=False,
        vflip=False,
        crop=None,
    ):
        self.transform = transform
        self.data_paths = sorted(
            [
                os.path.join(data_folder, file)
                for file in os.listdir(data_folder)
                if (
                    os.path.isfile(os.path.join(data_folder, file))
                    and file[-4:] in [".png", ".jpg"]
                )
            ]
        )
        self.labels_paths = sorted(
            [
                os.path.join(labels_folder, file)
                for file in os.listdir(labels_folder)
                if (
                    os.path.isfile(os.path.join(labels_folder, file))
                    and file[-4:] in [".png", ".jpg"]
                )
            ]
        )
        assert len(self.data_paths) == len(self.labels_paths)
        self.reverse = reverse
        self.hflip = hflip
        self.vflip = vflip
        self.crop = crop

    def __getitem__(self, index):
        x = Image.open(self.data_paths[index]).convert("RGB")
        y = Image.open(self.labels_paths[index]).convert("RGB")
        if self.crop is not None:
            x, y = paired_random_crop(x, y, size=self.crop)
        if self.hflip:
            x, y = paired_random_hflip(x, y)
        if self.vflip:
            x, y = paired_random_vflip(x, y)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return (x, y) if not self.reverse else (y, x)

    def __len__(self):
        return len(self.data_paths)


class PairedLoaderSampler(Sampler):
    def __init__(self, loader, device="cpu"):
        super(PairedLoaderSampler, self).__init__(device)
        # print("[Debug] PairedLoaderSampler: init")
        self.loader = loader
        # print("[Debug] PairedLoaderSampler: build iter")
        self.it = iter(self.loader)
        # print("[Debug] PairedLoaderSampler: init, OK")

    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch_X, batch_Y = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch_X) < size:
            return self.sample(size)

        return batch_X[:size].to(self.device), batch_Y[:size].to(self.device)


def get_paired_sampler(
    name,
    path,
    img_size=64,
    batch_size=64,
    device="cpu",
    reverse=False,
    load_ambient=False,
    num_workers=8,
):
    transform = Compose(
        [
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if name == "FS2K":
        source_folder, target_folder = "sketch", "photo"
        train_set = PairedDataset(
            os.path.join(path, "train", source_folder),
            os.path.join(path, "train", target_folder),
            transform=transform,
            reverse=reverse,
            hflip=True,
        )
        test_set = PairedDataset(
            os.path.join(path, "test", source_folder),
            os.path.join(path, "test", target_folder),
            transform=transform,
            reverse=reverse,
            hflip=True,
        )
    elif name in [
        "comic_faces",
        "comic_faces_v1",
        "celeba_mask",
        "aligned_anime_faces_sketch",
        "safebooru_sketch",
    ]:
        if name == "comic_faces":
            source_folder, target_folder = "faces", "comics"
        elif name == "comic_faces_v1":
            source_folder, target_folder = "face", "comics"
        elif name == "celeba_mask":
            source_folder, target_folder = "CelebAMask-HQ-mask-color", "CelebA-HQ-img"
        elif name == "safebooru_sketch":
            source_folder, target_folder = "safebooru_sketch", "safebooru_jpeg"
        else:
            source_folder, target_folder = "sketch", "image"
        # print("[Debug] data path, OK")
        dataset = PairedDataset(
            os.path.join(path, source_folder),
            os.path.join(path, target_folder),
            transform=transform,
            reverse=reverse,
            hflip=True,
        )
        # print("[Debug] dataset build, OK")
        idx = list(range(len(dataset)))
        test_ratio = 0.1
        test_size = int(len(idx) * test_ratio)
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]
        train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
        # print("[Debug] train/test partition, OK")
    else:
        raise Exception("Unknown dataset")

    train_sampler = PairedLoaderSampler(
        DataLoader(
            train_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
        ),
        device,
    )
    # print("[Debug] train sampler build, OK")
    test_sampler = PairedLoaderSampler(
        DataLoader(
            test_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
        ),
        device,
    )
    # print("[Debug] test sampler build, OK")
    return train_sampler, test_sampler


# ====================== Subset(Class) Guided ====================== #
def get_indicies_subset(dataset, new_labels={}, classes=4, subset_classes=None):
    labels_subset = []
    dataset_subset = []
    class_indicies = [[] for _ in range(classes)]
    i = 0
    for x, y in dataset:
        if y in subset_classes:
            if isinstance(y, int):
                class_indicies[new_labels[y]].append(i)
                labels_subset.append(new_labels[y])
            else:
                class_indicies[new_labels[y.item()]].append(i)
                labels_subset.append(new_labels[y.item()])
            dataset_subset.append(x)
            i += 1
    return dataset_subset, labels_subset, class_indicies


class SubsetGuidedDataset(Dataset):
    def __init__(
        self,
        dataset_in,
        dataset_out,
        num_labeled="all",
        in_indicies=None,
        out_indicies=None,
    ):
        super(SubsetGuidedDataset, self).__init__()
        self.dataset_in = dataset_in
        self.dataset_out = dataset_out
        assert len(in_indicies) == len(
            out_indicies
        )  # make sure in and out have same num of classes
        self.num_classes = len(in_indicies)
        self.subsets_in = in_indicies
        self.subsets_out = out_indicies
        if (
            num_labeled != "all"
        ):  # semi-supervision training: just using a less number of labeled samplers in each class
            assert isinstance(num_labeled, int)
            tmp_list = [
                np.random.choice(subset, num_labeled) for subset in self.subsets_out
            ]
            self.subsets_out = tmp_list
            # self.subset_out now is a list of index list, [[...],[...],...,[...]] , lenght is num_classes * num_labeled

    def get(self, class_, subsetsize):
        x_subset = []
        y_subset = []
        in_indexis = random.sample(list(self.subsets_in[class_]), subsetsize)
        out_indexis = random.sample(list(self.subsets_out[class_]), subsetsize)
        for x_i, y_i in zip(in_indexis, out_indexis):
            x, c1 = self.dataset_in[x_i]
            y, c2 = self.dataset_out[y_i]
            assert c1 == c2
            x_subset.append(x)
            y_subset.append(y)
        # shape=(subsetsize, sample/label)
        # sample.shape=(channels, height, width), label.shape=(1,)
        return torch.stack(x_subset), torch.stack(y_subset)

    def __len__(self):
        return len(self.dataset_in)


class SubsetGuidedSampler(Sampler):
    def __init__(self, loader, subsetsize=8, weight=None, device="cpu"):
        super(SubsetGuidedSampler, self).__init__(device)
        self.loader = loader
        self.subsetsize = subsetsize
        if weight is None:  # if no weight given, uniform probability for each class
            self.weight = [
                1 / self.loader.num_classes for _ in range(self.loader.num_classes)
            ]
        else:
            self.weight = weight

    def sample(self, num_selected_classes=5):
        classes = np.random.choice(
            self.loader.num_classes, num_selected_classes, p=self.weight
        )
        batch_X = []
        batch_Y = []
        batch_label = []
        with torch.no_grad():
            for class_ in classes:
                X, Y = self.loader.get(class_, self.subsetsize)
                batch_X.append(X.clone().to(self.device).float())
                batch_Y.append(Y.clone().to(self.device).float())
                for _ in range(self.subsetsize):
                    batch_label.append(class_)

        # shape=(num_selected_classes, subsetsize, sample/label)
        # sample.shape=(channels, height, width), label.shape=(1,)
        return (
            torch.stack(batch_X).to(self.device),
            torch.stack(batch_Y).to(self.device),
        )
