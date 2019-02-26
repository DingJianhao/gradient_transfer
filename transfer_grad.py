from torchvision import datasets
import SVHNDataset
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np

from Datasets import TripletMNIST, TripletSVHN, TripletMNIST_MINI
from Losses import TripletLoss
from Networks import TripletNet, EmbeddingNet, MLP_Embedding, Generator, Discriminator
from tensorboardX import SummaryWriter, FileWriter
from datetime import datetime

now = datetime.now()
writer = SummaryWriter('./log/' + now.strftime("%Y%m%d-%H%M%S") + '/')
print("Input these to startup Tensorboardx:\n\tcd {} \n\ttensorboard --logdir ./ --host=127.0.0.1".format('./log/' + now.strftime("%Y%m%d-%H%M%S") + '/'))

grad_B = 0
grad_S = 0
grads_B = []
grads_S = []


def hook_B(module, input, output):
    global grad_B
    grad_B = input[2].clone()
    # grad_B=grad_B.unsqueeze(0)


def hook_S(module, input, output):
    global grad_S
    grad_S = input[2].clone()
    # grad_S = grad_S.unsqueeze(0)


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

mnist_mini_triplet_train_dataset = TripletMNIST_MINI(train_mnist_dataset, 100, 0.5)
mnist_mini_triplet_test_dataset = TripletMNIST_MINI(test_mnist_dataset, 100, 0.5)

batch_size = 64
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
train_loader_S_list = list(enumerate(train_loader_S))

margin = 1.
embedding_net_B = MLP_Embedding()  # define network for big datasets
triplet_net_B = TripletNet(embedding_net_B)
embedding_net_S = MLP_Embedding()  # define network for small datasets
triplet_net_S = TripletNet(embedding_net_S)

layer_size = (256, 16)
G = Generator(layer_size)
D = Discriminator(layer_size)
# define hooks
h_B = embedding_net_B.fc2.register_backward_hook(hook_B)
h_S = embedding_net_S.fc2.register_backward_hook(hook_S)
if cuda:
    triplet_net_B.cuda()
    triplet_net_S.cuda()
    G.cuda()
    D.cuda()

loss_fn_B = TripletLoss(margin)
loss_fn_S = TripletLoss(margin)
lr = 1e-3
optim_B = optim.Adam(triplet_net_B.parameters(), lr=lr)
optim_S = optim.Adam(triplet_net_S.parameters(), lr=lr)
optim_G = optim.SGD(G.parameters(), lr=0.001)
optim_D = optim.SGD(D.parameters(), lr=0.001)
# scheduler_B = lr_scheduler.StepLR(optim_B, 8, gamma=0.1, last_epoch=-1)
# scheduler_S = lr_scheduler.StepLR(optim_S, 8, gamma=0.1, last_epoch=-1)

GAN_criterion = torch.nn.BCELoss()

n_epochs_B = 50
n_epochs_S = n_epochs_B * len(train_loader_B) // len(train_loader_S) + 1 if n_epochs_B * len(train_loader_B) % len(
    train_loader_S) else n_epochs_B * len(train_loader_B) // len(train_loader_S)
log_interval = 100

cnt = 0
global batch_idx_B, batch_idx_S
losses_B = []
losses_S = []
total_loss_B = 0
total_loss_S = 0
epoch_S = 0
for epoch_B in range(0, n_epochs_B):
    for batch_idx_B, (data_B, _) in enumerate(train_loader_B):
        # Batch samples from Big Data and Small Data
        if cnt == 0:
            train_loader_S_list = list(enumerate(train_loader_S))
        batch_idx_S, (data_S, _) = train_loader_S_list[cnt]
        cnt += 1
        cnt %= len(train_loader_S_list)

        if batch_idx_B == 0:
            # Big Data model Prepare
            # scheduler_B.step()
            triplet_net_B.train()
            losses_B = []
            total_loss_B = 0

        if batch_idx_S == 0:
            # Small Data model Prepare
            # scheduler_S.step()
            triplet_net_S.train()
            losses_S = []
            total_loss_S = 0
            grads_B = []
            grads_S = []

        # Big Data model Train
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
        '''
        if batch_idx_B % log_interval == 0:
            print('\tTrain: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx_B * len(data_B[0]), len(train_loader_B.dataset),
                100. * batch_idx_B / len(train_loader_B), np.mean(losses_B)))
            losses_B = []
        '''
        if batch_idx_B == len(train_loader_B) - 1:
            total_loss_B /= len(train_loader_B)
            print('[Big   Dataset] Epoch: {}/{}. Train      set: Average loss: {:.4f}'.format(epoch_B + 1, n_epochs_B,
                                                                                              total_loss_B))
            writer.add_scalar('Big Dataset/Train/Loss', total_loss_B, epoch_B + 1)

        # Small Data model Train
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
        '''
        if batch_idx_S % log_interval == 0:
            print('\tTrain: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx_S * len(data_S[0]), len(train_loader_S.dataset),
                100. * batch_idx_S / len(train_loader_S), np.mean(losses_S)))
            losses_S = []
        '''
        if batch_idx_S == len(train_loader_S) - 1:
            epoch_S += 1
            total_loss_S /= len(train_loader_S)
            print('[Small Dataset] Epoch: {}/{}. Train      set: Average loss: {:.4f}'.format(epoch_S, n_epochs_S,
                                                                                              total_loss_S))
            writer.add_scalar('Small Dataset/Train/Loss', total_loss_S, epoch_S)
        # collect GAN data

        grads_B.append(grad_B)
        grads_S.append(grad_S)

        # GAN Train
        '''
        if batch_idx_S == len(train_loader_S) - 1:
            grads_B = torch.stack(grads_B)
            grads_S = torch.stack(grads_S)
            grads_B_loader = torch.utils.data.DataLoader(grads_B, batch_size=16)
            grads_S_loader = torch.utils.data.DataLoader(grads_S, batch_size=16)
            n_epochs_G = 400
            for epoch_G in range(0, n_epochs_G):
                grads_S_loader_list = list(enumerate(grads_S_loader))
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
                    print("  Epoch:{}/{} Batch:{} Fake: {:.4f}  Real: {:.4f}".format(epoch_G + 1, n_epochs_G, i,
                                                                                     out_fake.data.mean().item(),
                                                                                     out_real.data.mean().item()))
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
            # exit(0)
            '''

        # Big Data model Validate
        if batch_idx_B == len(train_loader_B) - 1:
            with torch.no_grad():
                triplet_net_B.eval()
                val_loss = 0
                for batch_idx, (data, _) in enumerate(test_loader_B):
                    if cuda:
                        data = tuple(d.cuda() for d in data)
                    outputs = triplet_net_B(*data)
                    loss_inputs = outputs
                    loss_outputs = loss_fn_B(*loss_inputs)
                    val_loss += loss_outputs.item()
                val_loss /= len(test_loader_B)
                print(
                    '[Big   Dataset] Epoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch_B + 1, n_epochs_B,
                                                                                                val_loss))
                writer.add_scalar('Big Dataset/Val/Loss', val_loss, epoch_B + 1)

        # Small Data model Validate
        if batch_idx_S == len(train_loader_S) - 1:
            with torch.no_grad():
                triplet_net_S.eval()
                val_loss = 0
                for batch_idx, (data, _) in enumerate(test_loader_S):
                    if cuda:
                        data = tuple(d.cuda() for d in data)
                    outputs = triplet_net_S(*data)
                    loss_inputs = outputs
                    loss_outputs = loss_fn_S(*loss_inputs)
                    val_loss += loss_outputs.item()
                val_loss /= len(test_loader_B)
                print(
                    '[Small Dataset] Epoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch_S + 1, n_epochs_S,
                                                                                                val_loss))
                writer.add_scalar('Small Dataset/Val/Loss', val_loss, epoch_S + 1)

