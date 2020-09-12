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


def get_data_label(input_file="2"):
    # print("loading ", dataset_dir + input_file, ".mat")
    dataset_dir = "./data/time_one/"
    # student_data_paths = os.listdir(dataset_dir)
    # i = 0
    # for student_data_path in student_data_paths:
    # print(student_data_path)
    dd = np.load(dataset_dir + '/' + str(input_file) + '.npz', allow_pickle=True)
    print("over")

    # for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]:#0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
    # for i in [0]:
    # print(i)
    # dd = np.load('C:/Users/ying/Desktop/EEG/EEG/data/{}.npy'.format(i))
    # dd = np.load('../data/0.npy')
    train_data = dd['train_data']  ####2010, 62, 5
    train_labels = dd['train_label'] + 1
    test_data = dd['test_data']
    test_label = dd['test_label'] + 1

    val0=np.mean(train_data[:,:,0])
    val1=np.mean(train_data[:,:,1])
    val2=np.mean(train_data[:,:,2])
    val3=np.mean(train_data[:,:,3])
    val4=np.mean(train_data[:,:,4])

    train_data[:,:,0]=train_data[:,:,0]-val0   ### 标准化的减去均值
    train_data[:,:,1]=train_data[:,:,1]-val1
    train_data[:,:,2]=train_data[:,:,2]-val2
    train_data[:,:,3]=train_data[:,:,3]-val3
    train_data[:,:,4]=train_data[:,:,4]-val4

    train_data[:,:,0]=2*train_data[:,:,0]/val0  ###  两倍内容除以均值
    train_data[:,:,1]=2*train_data[:,:,1]/val1
    train_data[:,:,2]=2*train_data[:,:,2]/val2
    train_data[:,:,3]=2*train_data[:,:,3]/val3
    train_data[:,:,4]=2*train_data[:,:,4]/val4

    # train_labels=labels[0:2010]
    # test_data=DE[2010:3394,:,:]
    # test_data=np.reshape(test_data,[1384,62,5])
    #test_data=test_data-val
    #test_data=2*test_data/val

    val00=np.mean(test_data[:,:,0])
    val10=np.mean(test_data[:,:,1])
    val20=np.mean(test_data[:,:,2])
    val30=np.mean(test_data[:,:,3])
    val40=np.mean(test_data[:,:,4])

    test_data[:,:,0]=test_data[:,:,0]-val00
    test_data[:,:,1]=test_data[:,:,1]-val10
    test_data[:,:,2]=test_data[:,:,2]-val20
    test_data[:,:,3]=test_data[:,:,3]-val30
    test_data[:,:,4]=test_data[:,:,4]-val40

    test_data[:,:,0]=2*test_data[:,:,0]/val00
    test_data[:,:,1]=2*test_data[:,:,1]/val10
    test_data[:,:,2]=2*test_data[:,:,2]/val20
    test_data[:,:,3]=2*test_data[:,:,3]/val30
    test_data[:,:,4]=2*test_data[:,:,4]/val40


    print("train_data.shape", train_data.shape)

    # print("data.shape", data.shape)
    # data = np.vstack(())
    data = np.vstack((train_data, test_data))
    data = np.reshape(data, (data.shape[0], -1))
    train_data = np.reshape(train_data, (train_data.shape[0], -1))
    test_data = np.reshape(test_data, (test_data.shape[0], -1))
    labels = np.hstack((train_labels, test_label))
    print("labels", labels.shape)
    print("data", data.shape)
    return data, labels
    # return data, labels, train_data, train_labels, test_data, test_label




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

# def all_loader(source_dataset_name):
#     source_data, source_label, _, _, _, _ = get_data_label(source_dataset_name)
#     dataloader = torch.utils.data.DataLoader(
#     dataset=MyDataset(source_data, source_label),
#     batch_size=batch_size,
#     drop_last=True,
#     shuffle=True,
#     num_workers=8)
#     return dataloader
#
#
# def train_loader(target_dataset_name):
#     _, _, target_data ,target_label, _, _  = get_data_label(target_dataset_name)
#     dataloader_target = torch.utils.data.DataLoader(
#         dataset=MyDataset(target_data, target_label),
#         batch_size=batch_size,
#         drop_last=True,
#         shuffle=True,
#         num_workers=8)
#     return dataloader_target
#
# def test_loader(target_dataset_name):
#     _, _, _, _, target_data, target_label  = get_data_label(target_dataset_name)
#     dataloader_target = torch.utils.data.DataLoader(
#         dataset=MyDataset(target_data, target_label),
#         batch_size=batch_size,
#         drop_last=True,
#         shuffle=True,
#         num_workers=8)
#     return dataloader_target



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
    X = torch.Tensor(n, 310)
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
    # print(label.shape)  ##  3394
    # print('sum_label', sum(label))

    X, Y = [], []
    classes = 3 * [n]  ####  ten one [180, 180, 180]
    # print("classes", classes)
    i = 0
    while True:
        if len(X) == n * 3:  ## 当 n是10的时候
            break
        x, y = data[i], label[i]  ##  骚操作，留下每一种一个标签
        if classes[int(y)] > 0:
            X.append(x)
            Y.append(y)
            classes[int(y)] -= 1
        i += 1

    assert (len(X) == n * 3)
    aa = torch.from_numpy(np.array(X))
    bb = torch.from_numpy(np.array(Y))
    # print(bb.shape, aa.shape)
    # print("type(aa)", type(aa))
    # print("type(bb)", type(bb))
    print("bb",  bb[10:20])
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
    n= X_t.shape[0]   #10*shot #
    # print("n", n)  ## 540 = 180 * 3
    #shuffle order
    classes = torch.unique(Y_t)  ##  其中 是否相等
    classes = classes[torch.randperm(len(classes))]

    class_num = classes.shape[0]
    # print("classes", class_num)
    shot = n//class_num
    # print("shot", shot)

    def s_idxs(c):
        idx = torch.nonzero(Y_s.eq(int(c)))
        return idx[torch.randperm(len(idx))][:shot*2].squeeze()

    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix = torch.stack(source_idxs)
    target_matrix = torch.stack(target_idxs)


    G1, G2, G3, G4, G5, G6 = [], [] , [] , [], [],[]
    Y1, Y2 , Y3 , Y4, Y5, Y6 = [], [] ,[] ,[], [], []


    for i in range(class_num):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]], X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]], Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))

            G3.append((X_s[source_matrix[i % 2][j]], X_s[source_matrix[(i+1) % 2][j]]))
            Y3.append((Y_s[source_matrix[i % 2][j]], Y_s[source_matrix[(i + 1) % 2][j]]))
            G4.append((X_s[source_matrix[i % 2][j]], X_t[target_matrix[(i+1) % 2][j]]))
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
        # print(len(g))
        assert(len(g) == n)
    # print("over")
    return groups, groups_y




def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):
    print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)


