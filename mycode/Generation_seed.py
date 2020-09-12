import torch
import dataloader
import argparse
from models import main_models
import numpy as np
from sklearn import metrics
###@  just for DEAP test  ######
change = "deap_6/"
parser=argparse.ArgumentParser()
parser.add_argument('target',type=str, default = "s01")
parser.add_argument('number',type=int, default = 1)
parser.add_argument('nargs',type=list, nargs='+', default=['s01', 's29', 's20', 's24', 's13' ])
parser.add_argument('batch',type=int, default=30)
opt = parser.parse_args()

path = "./"+ change + opt.target +"/save_model/"
path1 = "./" + change + opt.target + "/"
# path = "./s04/save_model/"
# path1 = "./s04/"
use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
####读取准确率#####找出结果中最好的三个
import numpy as np
final = []
name = []
all = []
or_all = []
auc_all = []
target = opt.target
number = opt.number
for i in range(1, 33):
    all.append(i)
all.pop(number-1)  ##  list 去掉指定位置数据
all = set(all)

def get_name(num):
    if num <10:
        name = "s0" + str(num)
    else:
        name = "s" + str(num)

    return name


for i in all:
    if i <10:
        dir = './' + change + target + "/s0" + str(i) + '.npz'
    else:
        dir = './' + change + target + "/s" + str(i) + '.npz'
    a = np.load(dir)
    # print(a.keys())
    data = a['accuracy_all2']
    or_data = a['accuracy_all']
    auc = a['auc_all']
    # print(data.shape)
    temp = np.max(data)
    auc_temp = np.max(auc)
    or_all.append(or_data[-1])   ###
    final.append(temp)
    auc_all.append(auc_temp)
    name.append(i)
sort_list = np.argsort((-1) * np.array(or_all))  ##  返回的是排序后的坐标
print(or_all)
print(sort_list)
k = 31
a = sort_list  ##  代表从大到小的作为排序
# print(a)
index = a[0:k]
# print(index)
max_value = np.max(final)
print(np.max(final))
temp = final[a[0]]
print(temp)
aa = []
for i in range(len(index)):
    aa.append(name[index[i]])

test_list = []
final_k = 6
for i in range(final_k):
    test_list.append(get_name(aa[i]))
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
    # model = classifier(encode)
    return encode, classifier

test_dataloader = dataloader.source_loader(opt.target)
opt.nargs = test_list
def Caculate(weight):
    Encode=[]
    Classifier=[]
    for file in opt.nargs:

        Ez, Cz =get_model(file)
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
        Y = torch.zeros(opt.batch, 2).to(device)
        for i in range(len(Encode)):
            Encode[i].eval()
            Classifier[i].eval()
            Y+=Classifier[i](Encode[i](data))*weight[i]
        acc = acc + (torch.max(Y, 1)[1] == labels).float().mean().item()
        auc += metrics.roc_auc_score(labels.cpu(), torch.max(Y, 1)[1].cpu())
    accuracy = round(acc / float(len(test_dataloader)), 3)
    auc_temp = round(auc / float(len(test_dataloader)), 3)
    print("accuracy: %.3f " % (accuracy))
    return accuracy, auc_temp


def model_test(net, testloader):
    net.eval()
    output_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            output_labels+=predicted

    return output_labels

weight=[1/len(opt.nargs)]*len(opt.nargs)
save_accuracy, save_auc = Caculate(weight)
# path1 = "./" + change + opt.target + "/"
save_dir = "./" + change + opt.target  + '.xlsx'
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

