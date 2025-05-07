from scipy import io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

patchsize1 = 33
patchsize2 = 33
batchsize = 8
EPOCH = 200
LR = 0.001
pad_width = np.floor(patchsize1 / 2)
pad_width = np.int32(pad_width)
pad_width2 = np.floor(patchsize2 / 2)
pad_width2 = np.int32(pad_width2)



def data_load(name="Trento", split_percent=0.2):

    if name == "Trento":
        DataPath1 = './dataset/Trento/HSI.mat'
        DataPath2 = './dataset/Trento/LiDAR.mat'
        TRPath = './dataset/Trento/TRLabel.mat'
        TSPath = './dataset/Trento/TSLabel.mat'

        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)
        TrLabel = TrLabel['TRLabel']
        TsLabel = TsLabel['TSLabel']

        Data = io.loadmat(DataPath1)
        Data = Data['HSI']
        Data = Data.astype(np.float32)

        Data2 = io.loadmat(DataPath2)
        Data2 = Data2['LiDAR']
        Data2 = Data2.astype(np.float32)

    # elif name == "Augsburg":
    #     DataPath1 = './dataset/Augsburg/data_DSM.mat'
    #     DataPath2 = './dataset/Augsburg/data_HS_LR.mat'
    #     TRPath = './dataset/Augsburg/TrainImage.mat'
    #     TSPath = './dataset/Augsburg/TestImage.mat'
    #     TrLabel = io.loadmat(TRPath)
    #     TsLabel = io.loadmat(TSPath)
    #     TrLabel = TrLabel['TrainImage']
    #     TsLabel = TsLabel['TestImage']
    #
    #     Data2 = io.loadmat(DataPath1)
    #     Data2 = Data2['data_DSM']
    #     Data2 = Data2.astype(np.float32)
    #
    #     Data = io.loadmat(DataPath2)
    #     Data = Data['data_HS_LR']
    #     Data = Data.astype(np.float32)

    # elif name == "Houston":
    #     DataPath1 = './dataset/Houston2013/Houston_HS_HR.mat'
    #     DataPath2 = './dataset/Houston2013/Houston_DSM_HR.mat'
    #     TRPath = './dataset/Houston2013/Houston_train.mat'
    #     TSPath = './dataset/Houston2013/Houston_test.mat'
    #     TrLabel = io.loadmat(TRPath)
    #     TsLabel = io.loadmat(TSPath)
    #     print("Keys in TrLabel:", TrLabel.keys())
    #     print("Keys in TrLabel:", TsLabel.keys())
    #     TrLabel = TrLabel['TrainImage']
    #     TsLabel = TsLabel['TestImage']
    #
    #     Data = io.loadmat(DataPath1)
    #     print("Keys in Data:", Data.keys())
    #     Data = Data['data_HS_HR']
    #     Data = Data.astype(np.float32)# 转换为浮点数类型
    #
    #     Data2 = io.loadmat(DataPath2)
    #     print("Keys in Data2:", Data2.keys())
    #     Data2 = Data2['DSM']
    #     Data2 = Data2.astype(np.float32)

    elif name == "Houston":
        DataPath1 = "dataset\Houston\Houston_0.2_split.mat"
        DataPath2 = "dataset\\noise_Houston"
        data = io.loadmat(DataPath1)

        TrLabel = data['TR']
        TsLabel = data['TE']

        Data = data['HSI']
        Data = Data.astype(np.float32)# 转换为浮点数类型

        Data2 = data['LiDAR']
        Data2 = Data2.astype(np.float32)
        # 加载噪声图像
        additive_data = io.loadmat(DataPath2 + "/additive.mat")['data'].astype(np.float32)
        deadlines_data = io.loadmat(DataPath2 + "/deadlines.mat")['data'].astype(np.float32)
        kernal_data = io.loadmat(DataPath2 + "/kernal.mat")['data'].astype(np.float32)
        poisson_data = io.loadmat(DataPath2 + "/poisson.mat")['data'].astype(np.float32)
        salt_pepper_data = io.loadmat(DataPath2 + "/salt_pepper.mat")['data'].astype(np.float32)
        stripes_data = io.loadmat(DataPath2 + "/stripes.mat")['data'].astype(np.float32)
        zmguass_data = io.loadmat(DataPath2 + "/zmguass.mat")['data'].astype(np.float32)

    elif name == "MUUFL":
        DataPath1 = "E:\Green\dataset\MUUFL\MUUFL_0.2_split.mat"
        DataPath2 = "E:\Green\dataset\\noise_MUUFL"
        data = io.loadmat(DataPath1)

        TrLabel = data['TR']
        TsLabel = data['TE']

        Data = data['HSI']
        Data = Data.astype(np.float32)# 转换为浮点数类型

        Data2 = data['LiDAR']
        Data2 = Data2.astype(np.float32)
        # 加载噪声图像
        additive_data = io.loadmat(DataPath2 + "/additive.mat")['data'].astype(np.float32)
        deadlines_data = io.loadmat(DataPath2 + "/deadlines.mat")['data'].astype(np.float32)
        kernal_data = io.loadmat(DataPath2 + "/kernal.mat")['data'].astype(np.float32)
        poisson_data = io.loadmat(DataPath2 + "/poisson.mat")['data'].astype(np.float32)
        salt_pepper_data = io.loadmat(DataPath2 + "/salt_pepper.mat")['data'].astype(np.float32)
        stripes_data = io.loadmat(DataPath2 + "/stripes.mat")['data'].astype(np.float32)
        zmguass_data = io.loadmat(DataPath2 + "/zmguass.mat")['data'].astype(np.float32)

    elif name == "Augsburg":
        DataPath1 = "E:\Green\dataset\Augsburg\Augsburg_0.2_split.mat"
        DataPath2 = "E:\Green\dataset\\noise_Augsburg"
        data = io.loadmat(DataPath1)

        TrLabel = data['TR']
        TsLabel = data['TE']

        Data = data['HSI']
        Data = Data.astype(np.float32)# 转换为浮点数类型

        Data2 = data['LiDAR']
        Data2 = Data2.astype(np.float32)
        # 加载噪声图像
        additive_data = io.loadmat(DataPath2 + "/additive.mat")['data'].astype(np.float32)
        deadlines_data = io.loadmat(DataPath2 + "/deadlines.mat")['data'].astype(np.float32)
        kernal_data = io.loadmat(DataPath2 + "/kernal.mat")['data'].astype(np.float32)
        poisson_data = io.loadmat(DataPath2 + "/poisson.mat")['data'].astype(np.float32)
        salt_pepper_data = io.loadmat(DataPath2 + "/salt_pepper.mat")['data'].astype(np.float32)
        stripes_data = io.loadmat(DataPath2 + "/stripes.mat")['data'].astype(np.float32)
        zmguass_data = io.loadmat(DataPath2 + "/zmguass.mat")['data'].astype(np.float32)

    #spData, a, spTrLabel, b = train_test_split(Data, TrLabel, test_size=(1-split_percent), random_state=3, shuffle=False)# 划分训练集和测试集
    #spData2, a, spTrLabel2, b = train_test_split(Data2, TrLabel, test_size=(1-split_percent), random_state=3, shuffle=False)

    return (Data,Data2,TrLabel,TsLabel,additive_data,deadlines_data,
            kernal_data,poisson_data,salt_pepper_data,stripes_data,zmguass_data)


def nor_pca(Data,Data2,ispca=True):
    [m, n, l] = Data.shape
    for i in range(l):
        minimal = Data[:, :, i].min()
        maximal = Data[:, :, i].max()
        Data[:, :, i] = (Data[:, :, i] - minimal) / (maximal - minimal)

    minimal = Data2.min()
    maximal = Data2.max()
    Data2 = (Data2 - minimal) / (maximal - minimal)

    if ispca is True:
        NC = 20
        PC = np.reshape(Data, (m * n, l))
        pca = PCA(n_components=NC, copy=True, whiten=False)
        PC = pca.fit_transform(PC)
        PC = np.reshape(PC, (m, n, NC))
    else:
        NC = l
        PC = Data

    return PC,Data2,NC #349*1905*20,349*1905,20

#padding
def border_inter(PC,Data2,NC):
    temp = PC[:, :, 0]
    pad_width = np.floor(patchsize1 / 2)
    pad_width = np.int32(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')#359*1915
    [m2, n2] = temp2.shape
    x = np.empty((m2, n2, NC), dtype='float32')

    for i in range(NC):
        temp = PC[:, :, i]
        pad_width = np.floor(patchsize1 / 2)
        pad_width = np.int32(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x[:, :, i] = temp2

    x2 = Data2
    pad_width2 = np.floor(patchsize2 / 2)
    pad_width2 = np.int32(pad_width2)
    temp2 = np.pad(x2, pad_width2, 'symmetric')
    x2 = temp2
    return x, x2


def con_data(x,x2,TrLabel,TsLabel,NC):
    [ind1, ind2] = np.where(TrLabel != 0)# 返回非零元素的索引
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]#取patch 11*11*20
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))#121*20
        patch = np.transpose(patch)#20*121
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TrainPatch[i, :, :, :] = patch
        patchlabel = TrLabel[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch
        patchlabel = TsLabel[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel

    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch2 = np.empty((TrainNum, 1, patchsize2, patchsize2), dtype='float32')
    TrainLabel2 = np.empty(TrainNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TrainPatch2[i, :, :, :] = patch
        patchlabel2 = TrLabel[ind1[i], ind2[i]]
        TrainLabel2[i] = patchlabel2

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TestPatch2[i, :, :, :] = patch
        patchlabel2 = TsLabel[ind1[i], ind2[i]]
        TestLabel2[i] = patchlabel2

    return TrainPatch,TestPatch,TrainPatch2,TestPatch2,TrainLabel,TestLabel,TrainLabel2,TestLabel2

def con_data_even(x, x2, TrLabel, TsLabel, NC):
    # ============================== 第一部分：处理x（多通道数据） ==============================
    # 对输入图像进行反射填充（处理边界）
    x_padded = np.pad(x, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')

    # 获取训练集非零标签的索引
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)

    # 提取训练集图像块（基于填充后的坐标）
    for i in range(TrainNum):
        # 在填充后的图像上直接截取块（无需调整索引）
        row_start = ind1[i]
        row_end = ind1[i] + patchsize1  # 偶数尺寸，直接+patchsize1
        col_start = ind2[i]
        col_end = ind2[i] + patchsize1

        patch = x_padded[row_start:row_end, col_start:col_end, :]  # 形状：(10,10,NC)
        patch = np.transpose(patch, (2, 0, 1))  # 直接转置为 (NC,10,10)
        TrainPatch[i] = patch
        TrainLabel[i] = TrLabel[ind1[i], ind2[i]]

    # 处理测试集（逻辑与训练集一致）
    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    for i in range(TestNum):
        patch = x_padded[ind1[i]:ind1[i] + patchsize1, ind2[i]:ind2[i] + patchsize1, :]
        TestPatch[i] = np.transpose(patch, (2, 0, 1))
        TestLabel[i] = TsLabel[ind1[i], ind2[i]]

    # ============================== 第二部分：处理x2（单通道数据） ==============================
    # 对输入图像进行反射填充
    x2_padded = np.pad(x2, ((pad_width2, pad_width2), (pad_width2, pad_width2)), mode='reflect')

    # 处理训练集
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch2 = np.empty((TrainNum, 1, patchsize2, patchsize2), dtype='float32')
    TrainLabel2 = np.empty(TrainNum)
    for i in range(TrainNum):
        patch = x2_padded[ind1[i]:ind1[i] + patchsize2, ind2[i]:ind2[i] + patchsize2]  # (6,6)
        TrainPatch2[i, 0] = patch  # 直接赋值，无需reshape
        TrainLabel2[i] = TrLabel[ind1[i], ind2[i]]

    # 处理测试集
    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)
    for i in range(TestNum):
        patch = x2_padded[ind1[i]:ind1[i] + patchsize2, ind2[i]:ind2[i] + patchsize2]
        TestPatch2[i, 0] = patch
        TestLabel2[i] = TsLabel[ind1[i], ind2[i]]

    return TrainPatch, TestPatch, TrainPatch2, TestPatch2, TrainLabel, TestLabel, TrainLabel2, TestLabel2


def con_data1(x,x2,AllLabel,NC):

    [ind1, ind2] = np.where(AllLabel != 0)
    TestNum = len(ind1)
    Allpatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        Allpatch[i, :, :, :] = patch
        patchlabel = AllLabel[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel


    [ind1, ind2] = np.where(AllLabel != 0)
    TestNum = len(ind1)
    Allpatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        Allpatch2[i, :, :, :] = patch
        patchlabel2 = AllLabel[ind1[i], ind2[i]]
        TestLabel2[i] = patchlabel2

    return Allpatch,Allpatch2

def con_data2(x,TsLabel,NC):

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch

    return TestPatch

def con_data2_even(x, TsLabel, NC):

    # 对输入图像进行反射填充（处理边界）
    x_padded = np.pad(x, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')

    # 获取测试集非零标签的索引
    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')

    # 提取测试集图像块（基于填充后的坐标）
    for i in range(TestNum):
        # 在填充后的图像上直接截取块（无需调整索引）
        row_start = ind1[i]
        row_end = ind1[i] + patchsize1  # 偶数尺寸，直接+patchsize1
        col_start = ind2[i]
        col_end = ind2[i] + patchsize1

        # 提取块并调整维度
        patch = x_padded[row_start:row_end, col_start:col_end, :]  # 形状：(10,10,NC)
        patch = np.transpose(patch, (2, 0, 1))  # 直接转置为 (NC,10,10)
        TestPatch[i] = patch

    return TestPatch

def getIndex(TestLabel, temp):
    index = np.empty(shape=(2,temp), dtype=int)
    k = 0
    for i in range(len(TestLabel)):
        for j in range(len(TestLabel[0])):
            if TestLabel[i][j] != 0:
                index[0][k] = i+1
                index[1][k] = j+1
                k += 1

    return index