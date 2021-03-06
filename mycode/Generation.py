import torch
import dataloader_seed
import argparse
from models import main_models
import numpy as np
from sklearn import metrics
temple_dir = "./seed_6_210/"

parser=argparse.ArgumentParser()
parser.add_argument('target',type=str, default = '2')
parser.add_argument('number',type=str, default = '1')
parser.add_argument('nargs',type=list, nargs='+', default=['s01', 's29', 's20', 's24', 's13' ])
parser.add_argument('batch',type=int, default=30)
opt = parser.parse_args()

path =  temple_dir + opt.target +"/save_model/"
path1 = temple_dir + opt.target + "/"
# path = "./s04/save_model/"
# path1 = "./s04/"
use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
####读取准确率#####找出结果中最好的三个
import numpy as np
final = []
name = []
or_all = []
auc_all = []

target = opt.target
print("target", target )
number = opt.number
all =  ['1','10','13','2','24','26','3','31','4','42','44','6','7','8','9']
all.remove(str(number))
# print(all)


# def get_name(num):
#     if num <10:
#         name = "s0" + str(num)
#     else:
#         name = "s" + str(num)
#
#     return name


for i in set(all):
    # if i <10:
    #     dir = './' + target + "/s0" + str(i) + '.npz'
    # else:
    #     dir = './' + target + "/s" + str(i) + '.npz'
    dir = temple_dir + target + '/'+ str(i) + '.npz'
    a = np.load(dir)
    # print(a.keys())
    data = a['accuracy_all2']
    or_data = a['accuracy_all']
    auc = a['auc_all']
    # print(data.shape)
    temp = np.max(data)
    auc_temp = np.max(auc)
    or_all.append(np.mean(or_data))
    final.append(temp)  ## 每个人对应source迁移最好的结果
    auc_all.append(auc_temp)  ##  每个人对应source中auc 最好的结果
    name.append(i)
sort_list = np.argsort((-1) * np.array(or_all))  ##  返回的是排序后的坐标
print(or_all)
print(sort_list)
k = 14
a = sort_list  ##  代表从大到小的作为排序
# print(a)
index = a[0:k]
# print(index)
max_value = np.max(final)  ##  是
print(np.max(final))  ##  十四个人中最好的结果
temp = final[a[0]]  ##  对应最好的
print("corres", temp)  ##  对应最好的结果
aa = []
for i in range(len(index)):
    aa.append(name[index[i]])

test_list = []
final_k = 5
for i in range(final_k):
    test_list.append(aa[i])
print(test_list)

# save_dir = path1 = "./" + opt.target  + '.xlsx'
# import xlsxwriter
# workbook = xlsxwriter.Workbook(save_dir)
# worksheet = workbook.add_worksheet()
# aa = final.__len__()
# for i in range(int(aa)):
#     worksheet.write(i, 0, name[i])  # 第i行0列
#     worksheet.write(i, 2, final[i])  # 第i行1列
# workbook.close()


def get_model(file):
    encode = torch.load(path+"encode_%s.pth"%(file))
    classifier = torch.load(path + "classifier_%s.pth" % (file))
    return encode,classifier
# def get_model(file):
#     encode = torch.load(path+"encode_%s.pth"%(file))
#     classifier = torch.load(path + "classifier_%s.pth" % (file))
#     model = classifier(encode)
#     return model
from sklearn.preprocessing import label_binarize
test_dataloader = dataloader_seed.source_loader(opt.target)
opt.nargs = test_list
def Caculate(weight):
    Encode=[]
    Classifier=[]
    for file in opt.nargs:

        Ez, Cz=get_model(file)
        # Ez = Ez.to(device)
        # Cz = Cz.to(device)
        # model_temp.to(device)

        Encode.append(Ez)
        Classifier.append(Cz)
    acc = 0
    auc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = (labels.long()).to(device)
        Y = torch.zeros(opt.batch, 3).to(device)
        for i in range(len(Encode)):
            Encode[i].eval()
            Classifier[i].eval()
            Y+=Classifier[i](Encode[i](data))*weight[i]
        acc = acc + (torch.max(Y, 1)[1] == labels).float().mean().item()
        y_one_hot = labels.cpu().detach().numpy()
        y_one_hot = label_binarize(y_one_hot, np.arange(3))
        a = Y.cpu().detach().numpy()
        auc += metrics.roc_auc_score(y_one_hot, a, average='micro')
        # auc += metrics.roc_auc_score(labels.cpu(), torch.max(Y, 1)[1].cpu())
    accuracy = round(acc / float(len(test_dataloader)), 3)
    auc_temp = round(auc / float(len(test_dataloader)), 3)
    print("accuracy: %.3f " % (accuracy))
    return accuracy, auc_temp
print("len(opt.nargs)", len(opt.nargs))

weight=[1/len(opt.nargs)]*len(opt.nargs)

save_accuracy, save_auc = Caculate(weight)
save_dir = path1 = temple_dir + opt.target + '.xlsx'
import xlsxwriter
workbook = xlsxwriter.Workbook(save_dir)
worksheet = workbook.add_worksheet()
aa = final.__len__()
for i in range(int(aa)):
    worksheet.write(i, 0, name[i])  # 第i行0列
    worksheet.write(i, 1, final[i])  # 第i行1列
    worksheet.write(i, 8, auc_all[i])  # 第i行1列
bb = or_all.__len__()
for j in range(bb):
    worksheet.write(j, 5, or_all[j] )  # 第i行0列
    # worksheet.write(i, 1, final[i])  # 第i行1列

worksheet.write(0, 2, "max")
worksheet.write(1, 2, max_value)
worksheet.write(0, 3, "mean")
worksheet.write(1, 3, save_accuracy)
worksheet.write(0, 4, "k")
worksheet.write(1, 4, final_k)
worksheet.write(0, 6, test_list[0])
# worksheet.write(1, 6, test_list[1])
# worksheet.write(2, 6, test_list[2])
# worksheet.write(3, 6, test_list[3])
# worksheet.write(4, 6, test_list[4])

worksheet.write(0, 7, "or_data")
worksheet.write(1, 7, temp)

worksheet.write(0, 9, "mean_auc")
worksheet.write(1, 9, save_auc)


workbook.close()
print("over")


# if __name__ =='__main__':
#     weight=[1/len(opt.nargs)]*len(opt.nargs)
#     Caculate(weight)

