from torchvision import datasets
import SVHNDataset
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import copy
from Datasets import TripletMNIST, TripletSVHN, TripletMNIST_MINI
from Losses import TripletLoss
from Networks import TripletNet, EmbeddingNet, MLP_Embedding, Generator, Discriminator
from tensorboardX import SummaryWriter, FileWriter
from datetime import datetime

# now = datetime.now()
# writer = SummaryWriter('./log/' + now.strftime("%Y%m%d-%H%M%S") + '/')
# print("Input these to startup Tensorboardx:\n\tcd {} \n\ttensorboard --logdir ./ --host=127.0.0.1".format('./log/' + now.strftime("%Y%m%d-%H%M%S") + '/'))


# 小数据训练为主框架：
# 迭代对于当前状态（当前参数配置）：
# 大数据一个epoch 在当前参数上backward 对所有的梯度平均 保存 两次参数差 为大数据梯度grad_B 但是不更新参数
# 小数据一个epoch 在当前参数上backward 对所有的梯度平均 保存 两次参数差 为小数据梯度grad_S 更新参数
# <一批大数据对一批小数据>


grad_B = []
grad_S = []
grads_B = []
grads_S = []

def hook_B(module, input, output):
    global grad_B
    grad_B = input[2].clone()


def hook_S(module, input, output):
    global grad_S
    grad_S = input[2].clone()


torch.manual_seed(21)
torch.cuda.manual_seed_all(21)

train_mnist_dataset = datasets.MNIST('./data/MNIST', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]))
test_mnist_dataset = datasets.MNIST('./data/MNIST', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))

cuda = torch.cuda.is_available()

mnist_triplet_train_dataset = TripletMNIST(train_mnist_dataset)
mnist_triplet_test_dataset = TripletMNIST(test_mnist_dataset)

mnist_mini_triplet_train_dataset = TripletMNIST_MINI(train_mnist_dataset, 1000, 0.1)
mnist_mini_triplet_test_dataset = TripletMNIST_MINI(test_mnist_dataset, 1000, 0.1)

batch_size = 128
gan_batch_size = 4

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
mnist_triplet_train_loader = torch.utils.data.DataLoader(mnist_triplet_train_dataset, batch_size=batch_size,
                                                         shuffle=True, **kwargs)
mnist_triplet_test_loader = torch.utils.data.DataLoader(mnist_triplet_test_dataset, batch_size=batch_size,
                                                        shuffle=False, **kwargs)
mnist_mini_triplet_train_loader = torch.utils.data.DataLoader(mnist_mini_triplet_train_dataset, batch_size=batch_size,
                                                              shuffle=True, **kwargs)
mnist_mini_triplet_test_loader = torch.utils.data.DataLoader(mnist_mini_triplet_test_dataset, batch_size=batch_size,
                                                             shuffle=False, **kwargs)

train_loader_B = mnist_triplet_train_loader
test_loader_B = mnist_triplet_test_loader
train_loader_S = mnist_mini_triplet_train_loader
test_loader_S = mnist_mini_triplet_test_loader

margin = 1.
embedding_net_B = MLP_Embedding()  # define network for big datasets
triplet_net_B = TripletNet(embedding_net_B)
embedding_net_S = MLP_Embedding()  # define network for small datasets
triplet_net_S = TripletNet(embedding_net_S)

layer_size = (256, 16)
G = Generator(layer_size)
D = Discriminator(layer_size)
# define hooks
# h_B = embedding_net_B.fc2.register_backward_hook(hook_B)
# h_S = embedding_net_S.fc2.register_backward_hook(hook_S)
if cuda:
    triplet_net_S.cuda()
    triplet_net_B.cuda()
    G.cuda()
    D.cuda()

loss_fn_S = TripletLoss(margin)
loss_fn_B = TripletLoss(margin)
lr = 1e-3
optim_B = optim.Adam(triplet_net_B.parameters(), lr=lr)
optim_S = optim.Adam(triplet_net_S.parameters(), lr=lr)
optim_G = optim.Adam(G.parameters(), lr=0.001)
optim_D = optim.Adam(D.parameters(), lr=0.001)

GAN_criterion = torch.nn.BCELoss()

n_epochs_S = 16
gan_data_dim = 8
n_epochs_G = 1000
# log_interval = 100

print(len(train_loader_S))
print(len(train_loader_B))

losses_B = []
losses_S = []
total_loss_B = 0
total_loss_S = 0
param_B_1 = []
param_B_2 = []
param_S_1 = []
param_S_2 = []
for epoch_S in range(0, n_epochs_S): # 以小数据集同步
    total_loss_B = 0
    total_loss_S = 0
    if (epoch_S + 1) % gan_data_dim == 1:
        grads_B = []
        grads_S = []
    # Big Data model Train
    triplet_net_B.train()
    triplet_net_B.load_state_dict(copy.deepcopy(triplet_net_S.state_dict()))
    for name,param in triplet_net_B.named_parameters():
        if "fc2" in name and "weight" in name:
            param_B_1 = param.clone()
    for batch_idx_B, (data_B, _) in enumerate(train_loader_B):
        if cuda:
            data_B = tuple(d.cuda() for d in data_B)
        optim_B.zero_grad()
        triplet_embed_B = triplet_net_B(*data_B)
        loss_inputs_B = triplet_embed_B
        loss_outputs_B = loss_fn_B(*loss_inputs_B)
        loss_B = loss_outputs_B.item()
        losses_B.append(loss_B)
        total_loss_B += loss_outputs_B.item()
        loss_outputs_B.backward()
        optim_B.step()
    total_loss_B /= len(train_loader_B)
    print('[Big   Dataset] Epoch: {}/{}. Train loss: {:.4f}'.format(epoch_S + 1, n_epochs_S, total_loss_B))
    for name,param in triplet_net_B.named_parameters():
        if "fc2" in name and "weight" in name:
            param_B_2 = param.clone()
    grad_B = param_B_2-param_B_1

    # Small Data model Train
    for name,param in triplet_net_S.named_parameters():
        if "fc2" in name and "weight" in name:
            param_S_1 = param.clone()
    for batch_idx_S, (data_S, _) in enumerate(train_loader_S):
        if cuda:
            data_S = tuple(d.cuda() for d in data_S)
        optim_S.zero_grad()
        triplet_embed_S = triplet_net_S(*data_S)
        loss_inputs_S = triplet_embed_S
        loss_outputs_S = loss_fn_S(*loss_inputs_S)
        loss_S = loss_outputs_S.item()
        losses_S.append(loss_S)
        total_loss_S += loss_outputs_S.item()
        loss_outputs_S.backward()
        optim_S.step()
        pass
    total_loss_S /= len(train_loader_S)
    print('[Small Dataset] Epoch: {}/{}. Train loss: {:.4f}'.format(epoch_S + 1, n_epochs_S, total_loss_S))
    for name,param in triplet_net_S.named_parameters():
        if "fc2" in name and "weight" in name:
            param_S_2 = param.clone()
    grad_S = param_S_2 - param_S_1

    grads_B.append(grad_B)
    grads_S.append(grad_S)

    if (epoch_S+1) % gan_data_dim == 0 and False:
        grads_B = torch.stack(grads_B)
        grads_S = torch.stack(grads_S)
        grads_B_loader = torch.utils.data.DataLoader(grads_B, batch_size=gan_batch_size)
        grads_S_loader = torch.utils.data.DataLoader(grads_S, batch_size=gan_batch_size)
        for epoch_G in range(0, n_epochs_G):
            grads_S_loader_list = list(enumerate(grads_S_loader))
            total_fake = 0.0
            total_real = 0.0
            for i, data_fake in enumerate(grads_B_loader):
                _, data_real = grads_S_loader_list[i]

                real_label = torch.ones(data_real.size(0), 1)
                fake_label = torch.zeros(data_fake.size(0), 1)
                if cuda:
                    real_label = real_label.cuda()
                    fake_label = fake_label.cuda()
                gen_grads = G(data_fake)
                out_fake = D(gen_grads)
                d_fake_loss = GAN_criterion(out_fake, fake_label)

                out_real = D(data_real)
                d_real_loss = GAN_criterion(out_real, real_label)
                d_loss = d_fake_loss + d_real_loss
                total_fake+=out_fake.data.mean().item()
                total_real+=out_real.data.mean().item()
                optim_D.zero_grad()
                d_loss.backward()
                optim_D.step()

                data_fake_clone = data_fake.clone()
                gen_grads2 = G(data_fake_clone)
                out_fake2 = D(gen_grads2)
                g_loss = GAN_criterion(out_fake2, real_label)

                optim_G.zero_grad()
                g_loss.backward()
                optim_G.step()
            print("  Epoch:{}/{} Fake: {:.4f}  Real: {:.4f}".format(epoch_G + 1, n_epochs_G,
                                                                             total_fake/len(grads_B_loader),
                                                                             total_real / len(grads_B_loader)
                  ))
    pass
