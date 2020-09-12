import time
import scipy.io as iso
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
batch_size = 30


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)


def get_data_label(input_file="s02"):
    # print("loading ", dataset_dir + input_file, ".mat")
    dataset_dir = "./data/1D_dataset_move/"
    data_file = iso.loadmat(dataset_dir + "DE_"+input_file + ".mat")
    data = data_file["data"]  ## N* 160
    # print("data.shape", data.shape)
    # data = np.expand_dims(data, axis= 1)
    labels = data_file["valence_labels"]
    labels = np.squeeze(labels)
    # print("labels", labels.shape)
    return data, labels


def source_loader(source_dataset_name):
    source_data, source_label = get_data_label(source_dataset_name)
    dataloader = torch.utils.data.DataLoader(
    dataset=MyDataset(source_data, source_label),
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_workers=8)
    return dataloader


def target_loader(target_dataset_name):
    target_data ,target_label = get_data_label(target_dataset_name)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=MyDataset(target_data, target_label),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8)
    return dataloader_target



# #return MNIST dataloader
# def mnist_dataloader(batch_size=256,train=True):

#     dataloader=DataLoader(
#     datasets.MNIST('./data/mnist', train=train, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize([0.5],[0.5])
#                    ])),
#     batch_size=batch_size, shuffle=True)

#     return dataloader

# def svhn_dataloader(batch_size=4,train=True):
#     dataloader = DataLoader(
#         datasets.SVHN('./data/SVHN', split=('train' if train else 'test'), download=False,
#                        transform=transforms.Compose([
#                            transforms.Resize((28,28)),
#                            transforms.Grayscale(),
#                            transforms.ToTensor(),
#                            transforms.Normalize([0.5], [0.5])
#                        ])),
#         batch_size=batch_size, shuffle=False)

#     return dataloader


def sample_data(source_dataset_name):  ### 打乱数据

    data, label = get_data_label(source_dataset_name)
    n = len(data)
    X = torch.Tensor(n, 160)
    Y = torch.LongTensor(n)

    inds = torch.randperm(len(data))    ##
    for i, index in enumerate(inds):
        x, y = torch.tensor(data[index]), torch.tensor(label[index])
        X[i] = x
        Y[i] = y
    return X, Y

def create_target_samples(n, source_dataset_name):
    # dataset=datasets.SVHN('./data/SVHN', split='train', download=True,
    #                    transform=transforms.Compose([
    #                        transforms.Resize((28,28)),
    #                        transforms.Grayscale(),
    #                        transforms.ToTensor(),
    #                        transforms.Normalize([0.5], [0.5])
    #                    ]))

    data, label = get_data_label(source_dataset_name)
    print(label.shape)
    print('sum_label', sum(label))

    X, Y = [], []
    classes = 2 * [n]  ####  ten one
    i = 0
    while True:
        if len(X) == n * 2:  ## 当 n是10的时候
            break
        x, y = data[i], label[i]  ##  骚操作，留下每一种一个标签
        if classes[int(y)] > 0:
            X.append(x)
            Y.append(y)
            classes[int(y)] -= 1
        i += 1

    assert (len(X) == n * 2)
    aa = torch.from_numpy(np.array(X))
    bb = torch.from_numpy(np.array(Y))
    print(bb.shape, aa.shape)
    print("type(aa)", type(aa))
    print("type(bb)", type(bb))
    return aa, bb

"""
G1: a pair of pic comes from same domain ,same class
G3: a pair of pic comes from same domain, different classes

G2: a pair of pic comes from different domain,same class
G4: a pair of pic comes from different domain, different classes
"""
def create_groups(X_s, Y_s, X_t, Y_t, seed=1):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)
    n= X_t.shape[0]   #10*shot
    #shuffle order
    classes = torch.unique(Y_t)  ##  其中 是否相等
    classes = classes[torch.randperm(len(classes))]  ## 0 和1

    class_num = classes.shape[0]
    shot = n//class_num

    def s_idxs(c):
        idx = torch.nonzero(Y_s.eq(int(c))) ## 返回非零数据的位置
        return idx[torch.randperm(len(idx))][:shot*2].squeeze()

    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix = torch.stack(source_idxs)  ## 
    target_matrix = torch.stack(target_idxs)


    G1, G2, G3, G4, G5, G6 = [], [] , [] , [], [],[]
    Y1, Y2 , Y3 , Y4, Y5, Y6 = [], [] ,[] ,[], [], []


    for i in range(2):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]], X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]], Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))

            G3.append((X_s[source_matrix[i % 2][j]], X_s[source_matrix[(i+1) % 2][j]]))
            Y3.append((Y_s[source_matrix[i % 2][j]], Y_s[source_matrix[(i + 1) % 2][j]]))
            G4.append((X_s[source_matrix[i % 2][j]], X_t[target_matrix[(i+ 1) % 2][j]]))
            Y4.append((Y_s[source_matrix[i % 2][j]], Y_t[target_matrix[(i + 1) % 2][j]]))

    for i in range(class_num):
        for j in range(shot):
            G5.append((X_t[target_matrix[i][j]], X_t[target_matrix[i][int((j+1)%shot)]]))
            Y5.append((Y_t[target_matrix[i][j]], Y_t[target_matrix[i][int((j+1)%shot)]]))
            if i == 0:
                G6.append((X_t[target_matrix[i][j]], X_t[target_matrix[(i+1) % 2][j]]))
                Y6.append((Y_t[target_matrix[i][j]], Y_t[target_matrix[(i+1) % 2][j]]))
            else:
                G6.append((X_t[target_matrix[i][j]], X_t[target_matrix[(i + 1) % 2][int((j + 1) % shot)]]))
                Y6.append((Y_t[target_matrix[i][j]], Y_t[target_matrix[(i + 1) % 2][int((j + 1) % shot)]]))

    groups=[G1, G2, G3, G4, G5, G6]
    groups_y=[Y1, Y2, Y3, Y4, Y5, Y6]


    #make sure we sampled enough samples
    for g in groups:
        assert(len(g)==n)
    # print("over")
    return groups,groups_y




def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):
    print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)


