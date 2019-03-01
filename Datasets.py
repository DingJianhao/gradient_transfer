import numpy as np
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch

class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        torch.manual_seed(21)
        torch.cuda.manual_seed_all(21)
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            # print(self.train_data.size())
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets
            #每一张测试数据都给了一个triplet   e.g.【2,3211,3869】

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)

class TripletSVHN(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset,train=True):
        torch.manual_seed(21)
        torch.cuda.manual_seed_all(21)
        self.dataset = dataset
        self.train = train
        self.transform = self.dataset.transform

        self.labels = self.dataset.label
        self.data = self.dataset.data
        self.labels_set = set(self.labels.numpy())
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            random_state = np.random.RandomState(29)
            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.data))]
            self.test_triplets = triplets
            #每一张测试数据都给了一个triplet   e.g.【2,3211,3869】

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.data[index], self.labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.data[positive_index]
            img3 = self.data[negative_index]
        else:
            img1 = self.data[self.test_triplets[index][0]]
            img2 = self.data[self.test_triplets[index][1]]
            img3 = self.data[self.test_triplets[index][2]]

        img1 = Image.fromarray(np.transpose(img1, (1, 2, 0)))
        img2 = Image.fromarray(np.transpose(img2, (1, 2, 0)))
        img3 = Image.fromarray(np.transpose(img3, (1, 2, 0)))
        if self.transform is not None:
            img1 = img1.convert('L')
            img1 = self.transform(img1)
            img2 = img2.convert('L')
            img2 = self.transform(img2)
            img3 = img3.convert('L')
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)


class TripletMNIST_MINI(Dataset):
    '''
    only reduce training set
    not reduction of validation set

    sample_per_label: number of samples per class
    pro: probability of inverted samples
    '''
    def __init__(self, mnist_dataset, sample_per_label, pro):
        torch.manual_seed(21)
        torch.cuda.manual_seed_all(21)
        self.sample_per_label = sample_per_label
        self.pro = pro
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform
        random_state = np.random.RandomState(29)
        self.rs = random_state
        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data

            self.labels_set = set(self.train_labels.numpy())
            self._label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
            self.label_to_indices = {label:random_state.choice(self._label_to_indices[label],
                                                               self.sample_per_label)
                                                                for label in self.labels_set}
        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')

        r = self.rs.rand(3)
        if r[0]<=self.pro:
            img1 = ImageOps.invert(img1)
        if r[1]<=self.pro:
            img2 = ImageOps.invert(img2)
        if r[2]<=self.pro:
            img3 = ImageOps.invert(img3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        if self.train:
            return 10*self.sample_per_label
        else:
            return len(self.mnist_dataset)