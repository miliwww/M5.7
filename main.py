import os
import numpy as np
import random
import math

import numpy.random
from scipy.io import savemat
import spectral
import torch
import torch.utils.data as dataf
import torch.nn as nn
#import matplotlib.pyplot as plt
from scipy import io
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import gc
import time
import torch.nn.functional as F
from pymodel import pyCNN
from data_prepare import data_load, nor_pca, border_inter, con_data, getIndex, con_data1, con_data2, con_data_even, \
    con_data2_even

# 1. two branches share parameters; 2. use summation feature fusion;
# 3. weighted summation in the decision level, the weights are determined by their accuracies.

# setting parameters
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

batchsize = 64
EPOCH = 200
LR = 0.001
dataset_name = "Houston"
#dataset_name = "MUUFL"
#dataset_name = "Augsburg"

gc.collect()
torch.cuda.empty_cache()

# load data
# Data,Data2,TrLabel,TsLabel,spData,spTrLabel,spData2,spTrLabel2 = data_load(name=dataset_name,split_percent=0.2)
(Data,Data2,TrLabel,TsLabel,additive_data,deadlines_data,
 kernal_data,poisson_data,salt_pepper_data,stripes_data,zmguass_data)= data_load(name=dataset_name)
# TrLabel = small_sample(TrLabel, radito=0.2)
img_row = len(Data2)
img_col = len(Data2[0])

# normalization method 1: map to [0, 1]
[m, n, l] = Data.shape

PC,Data2,NC = nor_pca(Data,Data2,ispca=True)

# boundary interpolation
x, x2 = border_inter(PC,Data2,NC)
# construct the training and testing set of HSI

TrainPatch,TestPatch,TrainPatch2,TestPatch2,TrainLabel,TestLabel,TrainLabel2,TestLabel2 = con_data(x,x2,TrLabel,TsLabel,NC)

input_data = [additive_data, deadlines_data, kernal_data, stripes_data,poisson_data, salt_pepper_data,  zmguass_data]
noise_data = []

for data in input_data:
    PC, _ ,NC= nor_pca(data,Data2,ispca=True)
    data, _ = border_inter(PC,Data2,NC)
    result = con_data2(data,TsLabel,NC)
    noise_data.append(result)

noise_name = ['additive_data', 'deadlines_data', 'kernal_data','stripes_data', 'poisson_data', 'salt_pepper_data',  'zmguass_data']


# step3: change data to the input type of PyTorch (tensor)
TrainPatch1 = torch.from_numpy(TrainPatch)
TrainLabel1 = torch.from_numpy(TrainLabel)-1
TrainLabel1 = TrainLabel1.long()#torch.long 是 64 位有符号整数类型

TestPatch1 = torch.from_numpy(TestPatch)
TestLabel1 = torch.from_numpy(TestLabel)-1
TestLabel1 = TestLabel1.long()
Classes = len(np.unique(TrainLabel))

TrainPatch2 = torch.from_numpy(TrainPatch2)
TrainLabel2 = torch.from_numpy(TrainLabel2)-1
TrainLabel2 = TrainLabel2.long()

dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel2)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)
TestPatch2 = torch.from_numpy(TestPatch2)
TestLabel2 = torch.from_numpy(TestLabel2)-1
TestLabel2 = TestLabel2.long()


para_tune = False
FM = 64
if dataset_name == "Houston":
    para_tune = True


# 定义可学习权重（单个参数）
weight_raw = nn.Parameter(torch.tensor([0.5, 0.5], requires_grad=True))


class infonce_loss(nn.Module):
    def __init__(self,temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def NT_XentLoss(self,z1, z2, temperature=0.07):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        #device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2 * N, dtype=torch.bool)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

        negatives = similarity_matrix[~diag].view(2 * N, -1)

        logits = torch.cat([positives, negatives], dim=1).to('cuda')
        logits /= temperature

        labels = torch.zeros(2 * N, dtype=torch.int64).to('cuda')

        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)

    def forward(self, z1,z2):

        # 计算损失
        loss = self.NT_XentLoss(z1, z2)
        #print(f"InfoNCE Loss: {loss.item()}")
        return loss

# def reset_model_parameters(model):
#     for layer in model.modules():
#         if hasattr(layer, 'reset_parameters'):
#             layer.reset_parameters()

# cnn = CNN()
cnn = pyCNN(FM=FM,NC=NC,Classes=Classes,para_tune=para_tune)
# reset_model_parameters(cnn)
# move model to GPU
cnn.cuda()

# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters

optimizer = torch.optim.Adam([    {'params': cnn.parameters()},    {'params': [weight_raw]} ], lr=LR)

loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
infonce_loss = infonce_loss()

BestAcc = 0
pred_img = TsLabel
torch.cuda.synchronize()
start = time.time()

# train and test the designed model
for epoch in range(EPOCH):
    for step, (b_x1, b_x2, b_y) in enumerate(train_loader):

        # move train data to GPU
        b_x1 = b_x1.cuda()
        b_x2 = b_x2.cuda()
        b_y = b_y.cuda()


        out1, out2, out3 = cnn(b_x1, b_x2)
        # loss1 = loss_func(out1, b_y)
        # loss2 = loss_func(out2, b_y)
        loss1 = infonce_loss(out1, out2)
        loss3 = loss_func(out3, b_y)

        weight_celoss, weight_infonce = torch.softmax(weight_raw, dim=0)
        loss = weight_celoss * loss3 + weight_infonce * loss1

        #loss = weight_celoss *loss1 + weight_infonce * loss2 + weight * loss3
        #loss = loss1  + loss3
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        #print(f"weight_celoss: {weight_celoss.item(), loss3.item()},weight_infonce: {weight_infonce.item(), loss1.item()}")
        del out1, out2, out3

        if step % 50 == 0:
            cnn.eval()
            del b_x1, b_x2, b_y
            temp1 = TrainPatch1
            temp1 = temp1.cuda()
            temp2 = TrainPatch2
            temp2 = temp2.cuda()
            temp3, temp4, temp5 = cnn(temp1, temp2)
            Classes = np.unique(TrainLabel1)
            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 3000
            for i in range(number):
                temp = TestPatch1[i * 3000:(i + 1) * 3000, :, :, :]
                temp = temp.cuda()
                temp1 = TestPatch2[i * 3000:(i + 1) * 3000, :, :, :]
                temp1 = temp1.cuda()
                #temp2 = cnn(temp, temp1)[2] + cnn(temp, temp1)[1] + cnn(temp, temp1)[0]
                temp2 = cnn(temp,temp1)[2]
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 3000:(i + 1) * 3000] = temp3.cpu()
                del temp, temp1, temp2, temp3

            if (i + 1) * 3000 < len(TestLabel):
                temp = TestPatch1[(i + 1) * 3000:len(TestLabel), :, :, :]
                temp = temp.cuda()
                temp1 = TestPatch2[(i + 1) * 3000:len(TestLabel), :, :, :]
                temp1 = temp1.cuda()
                #temp2 = cnn(temp, temp1)[2] + cnn(temp, temp1)[1] + cnn(temp, temp1)[0]
                temp2 = cnn(temp, temp1)[2]
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 3000:len(TestLabel)] = temp3.cpu()
                del temp, temp1, temp2, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.6f' % accuracy,'| ')

            # save the parameters in network
            if accuracy > BestAcc:
                torch.save(cnn.state_dict(), 'BestAcc.pkl')
                BestAcc = accuracy

            del pred_y, accuracy
            cnn.train()

print('Best test acc:',BestAcc)
torch.cuda.synchronize()
end = time.time()
#print(end - start)
Train_time = end - start

# # test each class accuracy
# # divide test set into many subsets

# load the saved parameters
cnn.load_state_dict(torch.load('BestAcc.pkl', weights_only=True))
cnn.eval()
torch.cuda.synchronize()


print('-'*30,'Test','-'*30)
start = time.time()

pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel)//3000
for i in range(number):
    temp = TestPatch1[i*3000:(i+1)*3000, :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[i*3000:(i+1)*3000, :, :]
    temp1 = temp1.cuda()
    #temp2 =  1*cnn(temp, temp1)[2] +  0.01*cnn(temp, temp1)[1] +  0.01*cnn(temp, temp1)[0]
    temp2 = cnn(temp, temp1)[2]
    temp2_p = temp2.data
    # temp2 = cnn(temp, temp1)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i*3000:(i+1)*3000] = temp3.cpu()
    del temp, temp2, temp3

if (i+1)*3000 < len(TestLabel):
    temp = TestPatch1[(i+1)*3000:len(TestLabel), :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[(i+1)*3000:len(TestLabel), :, :]
    temp1 = temp1.cuda()
    #temp2 = 1*cnn(temp, temp1)[2] + 0.01*cnn(temp, temp1)[1] + 0.01*cnn(temp, temp1)[0]
    temp2 = cnn(temp, temp1)[2]
    temp2_p = temp2.data
    # temp2 = cnn(temp, temp1)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i+1)*3000:len(TestLabel)] = temp3.cpu()
    del temp, temp2, temp3

pred_y = torch.from_numpy(pred_y).long()
OA = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
oa = OA.numpy()

Classes = np.unique(TestLabel1)
EachAcc = np.empty(len(Classes))
pe = 0
for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestLabel1)):
        if TestLabel1[j] == cla:
            sum += 1
        if TestLabel1[j] == cla and pred_y[j] == cla:
            right += 1
    pe += sum*right
    EachAcc[i] = right.__float__()/sum.__float__()

AA = np.sum(EachAcc)/len(Classes)
pe = pe / math.pow(TestLabel1.size(0), 2)
kappa = (oa-pe)/(1-pe)
print(dataset_name)
print("OA:  ", OA)
print("oa:  ", oa)
print("EachAcc:  ", EachAcc)
print("AA:  ", AA)
print("kappa:  ", kappa)
torch.cuda.synchronize()
end = time.time()
#print(end - start)
Test_time = end - start
print('The Training time is: ', Train_time)
print('The Test time is: ', Test_time)

# savemat("./png/predHouston.mat", {'pred':pred_y1})
# savemat("./png/indexHouston.mat", {'index':index})
print()

print('Training size and testing size of HSI are:', TrainPatch.shape, 'and', TestPatch.shape)
print('Training size and testing size of LiDAR are:', TrainPatch2.shape, 'and', TestPatch2.shape)
print('Noise data size is:',noise_data[1].shape)


for i,TestPatch1 in enumerate(noise_data):
    name = noise_name[i]
    print('-' * 30, '%s %s data test'%(dataset_name,name), '-' * 30)
    start = time.time()
    TestPatch1 = torch.from_numpy(TestPatch1)

    pred_y = np.empty((len(TestLabel)), dtype='float32')
    number = len(TestLabel)//3000
    for i in range(number):
        temp = TestPatch1[i*3000:(i+1)*3000, :, :]
        temp = temp.cuda()
        temp1 = TestPatch2[i*3000:(i+1)*3000, :, :]
        temp1 = temp1.cuda()
        #temp2 =  1*cnn(temp, temp1)[2] +  0.01*cnn(temp, temp1)[1] +  0.01*cnn(temp, temp1)[0]
        temp2 = cnn(temp, temp1)[2]
        temp2_p = temp2.data
        # temp2 = cnn(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i*3000:(i+1)*3000] = temp3.cpu()
        del temp, temp2, temp3

    if (i+1)*5000 < len(TestLabel):
        temp = TestPatch1[(i+1)*3000:len(TestLabel), :, :]
        temp = temp.cuda()
        temp1 = TestPatch2[(i+1)*3000:len(TestLabel), :, :]
        temp1 = temp1.cuda()
        #temp2 = 1*cnn(temp, temp1)[2] + 0.01*cnn(temp, temp1)[1] + 0.01*cnn(temp, temp1)[0]
        temp2 = cnn(temp, temp1)[2]
        temp2_p = temp2.data
        # temp2 = cnn(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i+1)*3000:len(TestLabel)] = temp3.cpu()
        del temp, temp2, temp3

    pred_y = torch.from_numpy(pred_y).long()
    OA = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
    oa = OA.numpy()

    Classes = np.unique(TestLabel1)
    EachAcc = np.empty(len(Classes))
    pe = 0
    for i in range(len(Classes)):
        cla = Classes[i]
        right = 0
        sum = 0

        for j in range(len(TestLabel1)):
            if TestLabel1[j] == cla:
                sum += 1
            if TestLabel1[j] == cla and pred_y[j] == cla:
                right += 1
        pe += sum*right
        EachAcc[i] = right.__float__()/sum.__float__()

    AA = np.sum(EachAcc)/len(Classes)
    pe = pe / math.pow(TestLabel1.size(0), 2)
    kappa = (oa-pe)/(1-pe)
    #print(dataset_name)
    print("OA:  ", OA)
    print("oa:  ", oa)
    print("EachAcc:  ", EachAcc)
    print("AA:  ", AA)
    print("kappa:  ", kappa)
    torch.cuda.synchronize()
    end = time.time()
    #print(end - start)
    Test_time = end - start
    print('The Test time is: ', Test_time)

    # savemat("./png/predHouston.mat", {'pred':pred_y1})
    # savemat("./png/indexHouston.mat", {'index':index})
    print()

import winsound
winsound.Beep(1000, 2000)