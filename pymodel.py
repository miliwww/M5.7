import torch
import torch.nn as nn
import random
from torchsummary import summary

import torch.nn.functional as F
from CrosAttention import SA,SE
import math


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v1= self.max_pool(x)
        v = v+v1
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.size()
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        y = torch.ones(size=(b,c,h,w),dtype=torch.float32).cuda()
        z = torch.zeros(size=(b,c,h,w),dtype=torch.float32).cuda()
        beta = 0.2
        # change the value of beta to acquire best results
        out = torch.where(out.data>=beta,out,z)
        # print(out.grad)

        return out


class EFR(nn.Module):
    def __init__(self, channel):
        super(EFR, self).__init__()
        self.spatial_attention = SpatialAttentionModule()
        self.eca = ECABlock(channel)

    def forward(self, x):
        out = self.eca(x)
        out = self.spatial_attention(out) * out
        return out


# class MixStyle_F(nn.Module):
#     """MixStyle.
#     Reference:
#       Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
#     """
#
#     def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
#         """
#         Args:
#           p (float): probability of using MixStyle.
#           alpha (float): parameter of the Beta distribution.
#           eps (float): scaling parameter to avoid numerical issues.
#           mix (str): how to mix.
#         """
#         super().__init__()
#         self.p = p
#         self.beta = torch.distributions.Beta(alpha, alpha)
#         self.eps = eps
#         self.alpha = alpha
#         self.mix = mix
#         self._activated = True
#
#     def __repr__(self):
#         return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'
#
#     def set_activation_status(self, status=True):
#         self._activated = status
#
#     def update_mix_method(self, mix='random'):
#         self.mix = mix
#
#     def forward(self, x1,x2):
#         if not self.training or not self._activated:
#             return x1
#
#         # if random.random() > self.p: # 随机数大于p，不使用MixStyle
#         #     return x1
#
#         B = x1.size(0)
#
#         mu1 = x1.mean(dim=[2, 3], keepdim=True)# 计算均值
#         var1 = x1.var(dim=[2, 3], keepdim=True)# 计算方差
#         sig1 = (var1 + self.eps).sqrt()# 计算标准差
#         mu1, sig1 = mu1.detach(), sig1.detach()#禁止梯度传播
#         x_normed = (x1-mu1) / sig1
#
#         mu2 = x2.mean(dim=[2, 3], keepdim=True)
#         var2 = x2.var(dim=[2, 3], keepdim=True)
#         sig2 = (var2 + self.eps).sqrt()
#         mu2, sig2 = mu2.detach(), sig2.detach()
#
#         lmda = self.beta.sample((B, 1, 1, 1))
#         lmda = lmda.to(x1.device)
#         #
#         # if self.mix == 'random':
#         #     # random shuffle
#         #     perm = torch.randperm(B)
#         #
#         # elif self.mix == 'crossdomain':
#         #     # split into two halves and swap the order
#         #     perm = torch.arange(B - 1, -1, -1) # inverse index
#         #     perm_b, perm_a = perm.chunk(2)
#         #     perm_b = perm_b[torch.randperm(B // 2)]
#         #     perm_a = perm_a[torch.randperm(B // 2)]
#         #     perm = torch.cat([perm_b, perm_a], 0)
#         #
#         # else:
#         #     raise NotImplementedError
#
#         # mu2, sig2 = mu[perm], sig[perm]#使用 perm 索引获取其他样本的统计量 mu2 和 sig2。
#         mu_mix = mu1*lmda + mu2 * (1-lmda)
#         sig_mix = sig1*lmda + sig2 * (1-lmda)
#         output = x_normed*sig_mix + mu_mix
#
#         return output

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='crossdomain'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x  #64*64*5*5

        P = random.random()
        if P > self.p:
            return x

        B = x.size(0) #64

        mu = x.mean(dim=[2, 3], keepdim=True) #64*64*1*1
        var = x.var(dim=[2, 3], keepdim=True) #64*64*1*1 方差
        sig = (var + self.eps).sqrt() #64*64*1*1 标准差
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig #64*64*5*5 归一化

        lmda = self.beta.sample((B, 1, 1, 1)) #64*1*1*1 从Beta分布中采样一个参数lmda，用于控制两个样本之间的混合程度。
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B) #随机打乱索引 0-64

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            size1 = B // 2
            size2 = B - size1
            perm_b = perm[:size1]
            perm_a = perm[size1:]
            perm_b = perm_b[torch.randperm(size1)]
            perm_a = perm_a[torch.randperm(size2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm] #使用 perm 索引获取其他样本的统计量 mu2 和 sig2。
        mu_mix = mu*lmda + mu2 * (1-lmda) #64*64*1*1
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix #64*64*5*5

class MixStyle2(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='crossdomain'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x  #64*64*5*5

        P = random.random()
        if P > self.p:
            return x

        #B = x.size(0) #64
        B, C, H, W = x.shape

        mu = x.mean(dim=[2, 3], keepdim=True) #64*64*1*1
        var = x.var(dim=[2, 3], keepdim=True) #64*64*1*1 方差
        sig = (var + self.eps).sqrt().detach()#64*64*1*1 标准差
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig  # 64*64*5*5 归一化
        del mu, var, sig

        median_h = x.median(dim=2, keepdim=True)[0]
        median = median_h.median(dim=3, keepdim=True)[0].detach()

        # 计算Gram矩阵并添加扰动
        flat_x = x .view(B, C, -1) #64*64*25
        gram = torch.bmm(flat_x, flat_x.transpose(1, 2)) / (H * W + self.eps) #64*64*64 B,C,C
        gram = gram.detach()
        gram = gram + 1e-2 * torch.eye(C, device=x.device).unsqueeze(0) + 1e-6 *  torch.eye(C, device=x.device).unsqueeze(0)

        lmda = self.beta.sample((B, 1, 1, 1)) #64*1*1*1 从Beta分布中采样一个参数lmda，用于控制两个样本之间的混合程度。
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B) #随机打乱索引 0-64

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            size1 = B // 2
            size2 = B - size1
            perm_b = perm[:size1]
            perm_a = perm[size1:]
            perm_b = perm_b[torch.randperm(size1)]
            perm_a = perm_a[torch.randperm(size2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        #mu2, sig2 = mu[perm], sig[perm] #使用 perm 索引获取其他样本的统计量 mu2 和 sig2。
        #mu_mix = mu*lmda + mu2 * (1-lmda) #64*64*1*1
        #sig_mix = sig*lmda + sig2 * (1-lmda)

        median_mix = median * lmda + median[perm] * (1 - lmda)# (B, C, 1, 1)

        # 对 Gram 矩阵进行特征分解
        eigvals, eigvecs = torch.linalg.eigh(gram)  # (64, 64), (64, 64, 64)
        eigvals = eigvals + self.eps  # 避免除零

        # 计算白化矩阵: W = V * diag(1/sqrt(λ)) * V^T
        whitening = eigvecs @ torch.diag_embed(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(1, 2)  # (64, 64, 64)

        # 应用白化变换
        x_whitened = torch.bmm(whitening, x.view(B, C, -1))  #64*64*25 (B, C, H*W)
        x_whitened = x_whitened.view(B, C, H, W)  #64*64*5*5 (B, C, H, W)

        gram_mix = x_whitened * lmda + x_whitened[perm] * (1 - lmda)  # 64*64*5*5
        del x_whitened, whitening, eigvals, eigvecs, flat_x, gram

        return x_normed*gram_mix + median_mix #64*64*5*5

class HSI_MixStyle(nn.Module):
    def __init__(self,groups=[[0,16],[16,32],[32,64]],p=0.5,alpha = 0.1):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.groups = groups
        self._activated = True
    def forward(self, x):
        if not self.training or not self._activated:
            return x  #64*64*5*5
        B,C,H,W = x.size()
        out = []
        for start,end in self.groups:
            x_g =x[:,start:end,:,:]
            mu = x_g.mean(dim=[2,3],keepdim=True)
            sig = x_g.std(dim=[2,3],keepdim=True)
            x_normed = (x_g - mu)/(sig+1e-6)
            perm = torch.randperm(B)
            mu2,sig2 = mu[perm],sig[perm]
            lam=self.beta.sample((B,1,1,1)).to(x.device)
            mu_mix = lam*mu+(1-lam)*mu2
            sig_mix = lam*sig+(1-lam)*sig2
            out.append(x_normed*sig_mix+mu_mix)
        return torch.cat(out,dim=1)

class MixStyle3(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='crossdomain'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x  #64*64*5*5

        P = random.random()
        if P > self.p:
            return x

        B = x.size(0) #64

        mu = x.mean(dim=[1, 2], keepdim=True) #64*1*1*5
        var = x.var(dim=[1, 2], keepdim=True) #64*1*1*5 方差
        sig = (var + self.eps).sqrt() #64*1*1*5 标准差
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig #64*64*5*5 归一化

        lmda = self.beta.sample((B, 1, 1, 1)) #64*1*1*1 从Beta分布中采样一个参数lmda，用于控制两个样本之间的混合程度。
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B) #随机打乱索引 0-64

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            size1 = B // 2
            size2 = B - size1
            perm_b = perm[:size1]
            perm_a = perm[size1:]
            perm_b = perm_b[torch.randperm(size1)]
            perm_a = perm_a[torch.randperm(size2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm] #使用 perm 索引获取其他样本的统计量 mu2 和 sig2。
        mu_mix = mu*lmda + mu2 * (1-lmda) #64*1*1*5
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix #64*64*5*5


class ChannelFusion(nn.Module):
    def __init__(self,FM):
        super().__init__()
        self.conv1x1 = nn.Conv2d(FM*8, FM*4, kernel_size=1)
        #self.norm = nn.LayerNorm(FM * 4)
        self.BN = nn.BatchNorm2d(FM * 4)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def process_tensor(self, x):
        b, c, h, w = x.size()
        x_unfold = x.unfold(2, 2, 2).unfold(3, 2, 2)
        x_processed = x_unfold.contiguous().view(b, c, h//2, w//2, -1).permute(0, 1, 4, 2, 3).contiguous().view(b, -1, h//2, w//2)
        return x_processed

    def forward(self, x1, x2):
        x1_processed = self.process_tensor(x1)  # (64, 256, 4, 4)
        x2_processed = self.process_tensor(x2)  # (64, 256, 4, 4)
        del x1, x2
        b, c, h, w = x1_processed.shape
        chunk_size = 4  # 每块4个通道

        # 将通道维度拆分为块 [b, num_chunks, chunk_size, h, w]
        x1_chunks = x1_processed.view(b, -1, chunk_size, h, w)  # num_chunks = (FM*4)/4 = FM
        x2_chunks = x2_processed.view(b, -1, chunk_size, h, w)

        # 交替拼接 x1 和 x2 的块 [b, num_chunks*2, chunk_size, h, w]
        combined = torch.stack([x1_chunks, x2_chunks], dim=2).view(b, -1, h, w)  # 通道数 FM*8
        del  x1_processed, x2_processed, x1_chunks, x2_chunks
        output = self.conv1x1(combined)  # (64, 256, 4, 4)
        del combined

        # 调整形状以适配 LayerNorm (对每个空间位置的通道归一化)
        # output = output.permute(0, 2, 3, 1)  # [B, H, W, C]
        # output = self.norm(output)  # 归一化最后一维 (C)
        # output = output.permute(0, 3, 1, 2)  # 恢复形状 [B, C, H, W]
        output = self.BN(output)
        output = self.leaky_relu(output)
        output = self.dropout(output)

        return output  # (64, 256, 4, 4)



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)



class PyConv(nn.Module):

    def __init__(self, inplans, planes,  pyconv_kernels=[1, 3, 5], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])

        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])
        self.channelspatialselayer1 = EFR(channel=64)
        self.channelspatialselayer2 = EFR(channel=64)
        self.channelspatialselayer3 = EFR(channel=128)
    def forward(self, x):
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x1 = self.channelspatialselayer1(x1)
        x2 = self.channelspatialselayer2(x2)
        x3 = self.channelspatialselayer3(x3)
        return torch.cat((x1,x2,x3), dim=1)

#
# def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
#     return PyConv(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_channels, output_channels, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.d_head = output_channels // num_heads

        # 定义线性投影层
        self.W_q = nn.Linear(input_channels, output_channels)
        self.W_k = nn.Linear(input_channels, output_channels)
        self.W_v = nn.Linear(input_channels, output_channels)
        self.W_o = nn.Linear(output_channels, output_channels)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 输入形状: (batch_size, input_channels, height, width)
        batch_size, _, height, width = x.size()
        # 将输入转换为二维张量 (batch_size, seq_len, input_channels)
        x = x.view(batch_size, self.input_channels, -1).transpose(1, 2)

        # 线性投影
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 分割为多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))

        # 应用 Softmax 得到注意力权重
        attn_weights = self.softmax(scores)

        # 计算加权和
        output = torch.matmul(attn_weights, V)

        # 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.output_channels)

        # 线性变换
        output = self.W_o(output)

        # 将输出转换回四维张量 (batch_size, output_channels, 1, 1)
        output = output.transpose(1, 2).view(batch_size, self.output_channels, height, width)

        return output


def get_MultiHeadSelfAttention(input_channels, output_channels, num_heads):
    return MultiHeadSelfAttention(input_channels=input_channels, output_channels=output_channels, num_heads=num_heads)


class MLP(nn.Module):
    def __init__(self, dim,hidden_size,projection_size):#hidden_size = 2048
        super().__init__()
        self.layer = nn.Sequential(
            #nn.Linear(dim, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            #nn.ReLU(inplace=True),
            #nn.Linear(hidden_size, projection_size)
            nn.Linear(dim, projection_size),
            nn.BatchNorm1d(projection_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 输入 x1 的形状为 (batch_size, channels, height, width)，这里假设为 (64, 256, 1, 1)
        batch_size = x.size(0)
        # 先将输入张量从 (batch_size, channels, height, width) 调整为 (batch_size, channels)
        x = x.view(batch_size, -1)
        return self.layer(x)



class pyCNN(nn.Module):
    def __init__(self,NC,Classes,FM=32,para_tune=True):
        super(pyCNN, self).__init__()
        #self.sa = SA(in_channels=FM*2)
        #self.se = SE(in_channels=FM*2,para_tune=para_tune)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = NC,out_channels = FM,kernel_size = 3,stride = 2,padding = 0),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            #nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM*4, int(FM/2), 4, 2, 1),
            nn.BatchNorm2d(int(FM/2)),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            #get_pyconv(inplans=FM*2, planes=FM*4, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
            get_MultiHeadSelfAttention(input_channels=FM*2, output_channels=FM*4, num_heads=8),
            nn.BatchNorm2d(FM*4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, FM, 3, 2, 0, ),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(FM*4, int(FM/2), 4, 2, 1),
            nn.BatchNorm2d(int(FM/2)),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv6 = nn.Sequential(
            #get_pyconv(inplans=FM * 2, planes=FM * 4, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
            get_MultiHeadSelfAttention(input_channels=FM*2, output_channels=FM * 4, num_heads=8),
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )

        # self.FusionLayer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=FM*4,
        #         out_channels=FM*2,
        #         kernel_size=1,
        #     ),
        #     nn.BatchNorm2d(FM*2),
        #     nn.LeakyReLU(),
        # )
        #self.out1 = nn.Linear(FM*4,Classes)
        #self.out2 = nn.Linear(FM*4,Classes)
        self.out3 = nn.Linear(FM*4,Classes)
        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(FM*4, Classes)
        self.projection_head1 = MLP(FM*4,FM*64,FM*4)
        self.projection_head2 = MLP(FM*4,FM*64,FM*4)
        self.MixStyle1 = MixStyle2()
        self.MixStyle2 = MixStyle2()
        self.downsample1 = nn.PixelUnshuffle(2)
        self.downsample2 = nn.PixelUnshuffle(2)
        self.CF1 = ChannelFusion(FM)
        self.CF2 = ChannelFusion(FM//2)
        self.HSI_MS1 = HSI_MixStyle(groups=[[0,16],[16,32]])

    def forward(self, x1, x2):
        # print(x1.shape)
        # print(x2.shape)
        # print("")
        x1 = self.conv1(x1)
        x2 = self.conv4(x2)
        # print(x1.shape)
        # print(x2.shape)
        # print("")

        x1 = self.MixStyle1(x1)

        x1 = self.CF1(x1,x2)
        x2 = self.downsample1(x2)
        # print(x1.shape)
        # print(x2.shape)
        # print("")

        x1 = self.MixStyle2(x1)

        x1 = self.conv2(x1)
        x2 = self.conv5(x2)
        # print(x1.shape)
        # print(x2.shape)
        # print("")

        x1 = self.HSI_MS1(x1)

        x1 = self.CF2(x1,x2)
        x2 = self.downsample2(x2)
        # print(x1.shape)
        # print(x2.shape)
        # print("")

        #x1 = self.sa(x1,x2)
        #x2 = self.se(x2,x1)
        # print(x1.shape)
        # print(x2.shape)
        # print("")

        x1 = self.conv3(x1)
        x2 = self.conv6(x2)

        x1 = self.projection_head1(x1)
        x2 = self.projection_head2(x2)

        x = x1 + x2
        out3 = self.out3(x)

        return x1, x2, out3




# cnn = pyCNN(NC=40,Classes=13,FM=64)
# a = torch.randn(size=(64,40,11,11))
# b = torch.randn(size=(64,1,11,11))
#
# c,d,e = cnn(a,b)
# print(c)
# print()

# py = get_pyconv(inplans=128,planes=256,pyconv_kernels=[3,5,7],stride=1,pyconv_groups=[1,4,8])
# a = torch.randn(size=(64,128,2,2))
# b = py(a)

# pe = ChannelSpatialSELayer2D(num_channels=64,reduction_ratio=2)
# sp = SpatialSELayer2D(num_channels=64)
#
# a = torch.randn(size=(64,64,11,11))
# b = pe(a)
# c = sp(a)
# print(b)

# print()
