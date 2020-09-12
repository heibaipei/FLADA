import argparse
import torch
import dataloader_seed_6
from models import main_models_seed_6
import numpy as np
import os
from sklearn import metrics

from sklearn.preprocessing import label_binarize
# python main1.py 5 120 50 's05' 's01' 112 64
parser=argparse.ArgumentParser()
parser.add_argument('n_epoches_1',type=int,default=50)
parser.add_argument('n_epoches_2', type=int, default=120)
parser.add_argument('n_epoches_3', type=int, default=50)
parser.add_argument('original', type = str, default= 's05')
parser.add_argument('target', type = str, default= 's01')
parser.add_argument('n_target_samples', type=int, default=240) ###
parser.add_argument('batch_size', type=int, default=30)

opt=vars(parser.parse_args())

use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

batch_size = 30
dropout = 0.5
output_size = 64
hidden_size = 32
embed_dim = 32
bidirectional = True
attention_size = 16
sequence_length = 128
o_data = opt['original']
bb= opt['target']
train_dataloader = dataloader_seed_6.source_loader(o_data)
# train_dataloader = dataloader.mnist_dataloader(batch_size = opt['batch_size'], train=True, )
test_dataloader = dataloader_seed_6.source_loader(bb)

classifier = main_models_seed_6.Classifier()
encoder = main_models_seed_6.Encoder()
discriminator = main_models_seed_6.DCD(input_features=128)

classifier.to(device)
encoder.to(device)
discriminator.to(device)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()), lr=0.001)


if __name__=='__main__':
    accuracy_all = []
    for epoch in range(opt['n_epoches_1']):
        for data, labels in train_dataloader:
            # print(data.shape)
            data = data.to(device)
            labels = (labels.long()).to(device)
            optimizer.zero_grad()
            y_pred = classifier(encoder(data))
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

        acc = 0
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc = acc + (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

        accuracy = round(acc / float(len(test_dataloader)), 3)
        accuracy_all.append(accuracy)

        print("step1----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, opt['n_epoches_1'], accuracy))


    # -------------------------------------------------------------------

    X_s, Y_s = dataloader_seed_6.sample_data(o_data)  ##
    # print("X_s.shape", X_s.shape)
    X_t, Y_t = dataloader_seed_6.create_target_samples(opt['n_target_samples'], bb)  ## 112
    # print("X_t.shape" , X_t.shape)

    # -----------------train DCD for step 2--------------------------------


    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(opt['n_epoches_2']):
        # data
        groups, aa = dataloader_seed_6.sample_groups(X_s, Y_s, X_t, Y_t, seed=epoch)  ##  六个group
        n_iters = 6 * len(groups[1])  ##  same class from source
        # print('n_iters', n_iters)  ##960  240 *4
        index_list = torch.randperm(n_iters)
        mini_batch_size = 30  # use mini_batch train can be more stable

        loss_mean = []

        X1 = []
        X2 = []
        ground_truths = []
        for index in range(n_iters):
            ground_truth = index_list[index] // len(groups[1]) ##  整除  多少倍就是哪种类型
            x1, x2 = groups[ground_truth][index_list[index] - len(groups[1]) * ground_truth]
            # print('x1.shape, x2.shape', x1.shape, x2.shape)

            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            # select data for a mini-batch to train
            if (index + 1) % mini_batch_size == 0:
                X1 = np.stack(X1)
                X2 = np.stack(X2)
                # print('X1.shape,X2.shape after', X1.shape,  X2.shape)
                ground_truths = torch.LongTensor(ground_truths)
                X1 = torch.tensor(X1).to(device)
                X2 = torch.tensor(X2).to(device)
                ground_truths = ground_truths.to(device)

                optimizer_D.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                optimizer_D.step()
                loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []

        print("step2----Epoch %d/%d loss:%.3f" % (epoch + 1, opt['n_epoches_2'], np.mean(loss_mean)))

    # ----------------------------------------------------------------------

    # -------------------training for step 3-------------------
    optimizer_g_h = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.0001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    test_dataloader = dataloader_seed_6.target_loader(bb)
    accuracy_all2 = []
    auc_all = []

    path = './/seed_6_90//' + str(bb)
    isExists = os.path.exists(path)
    if isExists:
        pass
    else:
        os.makedirs(path)
    path2 = './/seed_6_90//' + str(bb) + "//save_model"
    isExists2 = os.path.exists(path2)
    if isExists2:
        pass
    else:
        os.makedirs(path2)
    max = 0
    for epoch in range(opt['n_epoches_3']):
        # ---training g and h , DCD is frozen

        groups, groups_y = dataloader_seed_6.sample_groups(X_s, Y_s, X_t, Y_t, seed=opt['n_epoches_2'] + epoch)
        G1, G2, G3, G4, G5, G6 = groups
        Y1, Y2, Y3, Y4, Y5, Y6 = groups_y
        groups_2 = [G2, G4, G5, G6]
        # print("Y2[40:50]", Y2[40:50])
        # print("Y4[40:50]",Y2[40:50] )
        # print("Y5[40:50]",Y5[40:50] )
        # print("Y6[40:50]", Y6[40:50])  ##  都是tensor 类型

        groups_y_2 = [Y2, Y4, Y5, Y6]
        n_iters = 4 * len(G2)
        index_list = torch.randperm(n_iters)

        n_iters_dcd = 6 * len(G2)
        index_list_dcd = torch.randperm(n_iters_dcd)

        mini_batch_size_g_h = 30  # data only contains G2 and G4 ,so decrease mini_batch
        mini_batch_size_dcd = 30  # data contains G1,G2,G3,G4 so use 40 as mini_batch
        X1 = []
        X2 = []
        ground_truths_y1 = []
        ground_truths_y2 = []
        dcd_labels = []
        for index in range(n_iters):
            ground_truth = index_list[index] // len(G2)
            x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]


            dcd_label = 0 if ground_truth == 0 or ground_truth ==2 else 2
            X1.append(x1)
            X2.append(x2)
            ground_truths_y1.append(y1)
            ground_truths_y2.append(y2)
            dcd_labels.append(dcd_label)

            if (index + 1) % mini_batch_size_g_h == 0:
                # X1 = torch.tensor(X1).to(device)
                # X2 = torch.tensor(X2).to(device)
                X1 = torch.stack([tmp.float() for tmp in X1])
                X2 = torch.stack([tmp.float() for tmp in X2])
                # print(ground_truths_y1)
                # print("dcd_labels",  dcd_labels)
                # print(ground_truths_y1)
                # print(type(ground_truths_y1))
                # ground_truths_y11 = torch.from_numpy(np.array(ground_truths_y1))
                # ground_truths_y21 = torch.from_numpy(np.array(ground_truths_y2))
                ground_truths_y1 = torch.as_tensor(ground_truths_y1).long()
                ground_truths_y2 = torch.as_tensor(ground_truths_y2).long()
                dcd_labels = torch.LongTensor(dcd_labels)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths_y1 = ground_truths_y1.to(device)
                ground_truths_y2 = ground_truths_y2.to(device)
                dcd_labels = dcd_labels.to(device)
                optimizer_g_h.zero_grad()
                encoder_X1 = encoder(X1)
                encoder_X2 = encoder(X2)

                X_cat = torch.cat([encoder_X1, encoder_X2], 1)
                y_pred_X1 = classifier(encoder_X1)
                y_pred_X2 = classifier(encoder_X2)
                y_pred_dcd = discriminator(X_cat)

                loss_X1 = loss_fn(y_pred_X1, ground_truths_y1)
                loss_X2 = loss_fn(y_pred_X2, ground_truths_y2)
                # print("dcd_labels", set(dcd_labels))
                loss_dcd = loss_fn(y_pred_dcd, dcd_labels)

                loss_sum = loss_X1 + loss_X2 + 1 * loss_dcd

                loss_sum.backward()
                optimizer_g_h.step()

                X1 = []
                X2 = []
                ground_truths_y1 = []
                ground_truths_y2 = []
                dcd_labels = []


        # ----training dcd ,g and h frozen
        X1 = []
        X2 = []
        ground_truths = []

        for index in range(n_iters_dcd):

            ground_truth = index_list_dcd[index] // len(groups[1])  ## 分为六组
            x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
            X1.append(x1)
            X2.append(x2)  ####  w错误的地方
            ground_truths.append(ground_truth)

            if (index + 1) % mini_batch_size_dcd == 0:
                X1 = torch.stack([tmp.float() for tmp in X1])
                X2 = torch.stack([tmp.float() for tmp in X2])
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_d.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                optimizer_d.step()
                # loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []  ##   两个轮流训练

        from sklearn.preprocessing import label_binarize        # testing
        acc = 0
        auc = 0
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
            y_one_hot = labels.cpu().detach().numpy()
            y_one_hot = label_binarize(y_one_hot, np.arange(3))
            a = y_test_pred.cpu().detach().numpy()
            auc += metrics.roc_auc_score(y_one_hot, a, average='micro')

            # myscore = make_scorer(metrics.roc_auc_score, multi_class='ovo', needs_proba=True)
            # auc += metrics.roc_auc_score(labels.cpu(), torch.max(y_test_pred, 1)[1].cpu(), )

        accuracy = round(acc / float(len(test_dataloader)), 3)
        auc_temp = round(auc/ float(len(test_dataloader)), 3)

        if accuracy > max:
            max = accuracy
            torch.save(encoder, path2 + '//encode_%s.pth' % (opt['original']))
            torch.save(classifier, path2 + '//classifier_%s.pth' % (opt['original']))
        accuracy_all2.append(accuracy)
        auc_all.append(auc_temp)


        print("step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, opt['n_epoches_3'], accuracy))

    # path = './/' + str(bb)
    # isExists = os.path.exists(path)
    # if isExists:
    #     pass
    # else:
    #     os.makedirs(path)
    # path2 = './/' + str(bb)+ "//save_model"
    # isExists2 = os.path.exists(path2)
    # if isExists2:
    #     pass
    # else:
    #     os.makedirs(path2)
    # torch.save(encoder, path2 + '//encode_%s.pth' % (opt['original']))
    # torch.save(classifier, path2 + '//classifier_%s.pth' % (opt['original']))

    save_dir = path + '//' + str(o_data) + '.npz'
    np.savez(save_dir, accuracy_all = accuracy_all , accuracy_all2 = accuracy_all2, auc_all = auc_all)






















