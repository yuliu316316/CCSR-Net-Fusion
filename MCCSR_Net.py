import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def pst(x, s, lam):
    """implement a piece-wise soft threshold function """
    lam = lam.cuda()
    s = s.cuda()
    x = x.cuda()
    x1 = x + 2 * lam
    x2 = torch.zeros(x.shape).cuda()
    x5 = x - 2 * lam
    c1 = (((s >= 0) & (x < -2 * lam)) | ((s < 0) & (x < s - 2 * lam)))
    c2 = (((s >= 0) & (x <= 0) & (x >= -2 * lam)) | ((s < 0) & (x <= 2 * lam) & (x >= 0)))
    c3 = (((s >= 0) & (x > 0) & (x < s)) | ((s < 0) & (x < 0) & (x > s)))
    c4 = (((s >= 0) & (x >= s) & (x <= s + 2 * lam)) | ((s < 0) & (x >= s - 2 * lam) & (x <= s)))
    return torch.where(c1, x1,
                       torch.where(c2, x2,
                                   torch.where(c3, x,
                                               torch.where(c4, s, x5))))


class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilate):
        super(BasicUnit, self).__init__()
        self.basic_unit = nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilate, padding_mode='reflect', dilation=dilate, bias=False)

    def forward(self, input):
        return self.basic_unit(input)


class Lambda(nn.Module):
    def __init__(self):
        super(Lambda, self).__init__()
        self.lam = nn.Parameter(torch.Tensor([0.01]))

    def forward(self):
        return self.lam


class Prediction(nn.Module):
    def __init__(self, num_channels, kernel_size,  dilation):
        super(Prediction, self).__init__()
        self.in_channel = num_channels
        self.num_filters = 64
        self.kernel_size_c = kernel_size
        self.dilate = dilation
        self.layer_in_u = BasicUnit(self.in_channel, self.num_filters, self.kernel_size_c, self.dilate)
        self.layer_in_s = BasicUnit(self.in_channel, self.num_filters, self.kernel_size_c, self.dilate)
        self.lam = nn.Parameter(torch.Tensor([0.01]))
        self.layer_e_u = BasicUnit(self.in_channel, self.num_filters, self.kernel_size_c, self.dilate)
        self.layer_d_u = BasicUnit(self.num_filters, self.in_channel, self.kernel_size_c, self.dilate)
        self.layer_e_s = BasicUnit(self.in_channel, self.num_filters, self.kernel_size_c, self.dilate)
        self.layer_d_s = BasicUnit(self.num_filters, self.in_channel, self.kernel_size_c, self.dilate)

    def forward(self, i, tensor_u, tensor_s, mod_1, mod_2):
        p1_u = self.layer_in_u(mod_1)
        q1_s = self.layer_in_s(mod_2)
        if i == 0:
            tensor_u = torch.zeros(p1_u.shape).cuda()
            tensor_s = torch.zeros(p1_u.shape).cuda()
        p3_u = self.layer_d_u(tensor_u)
        p3_u = self.layer_e_u(p3_u)
        p3_u = tensor_u - p3_u
        p3_u.add_(p1_u)
        tensor_u = pst(p3_u, tensor_s, self.lam)
        q3_s = self.layer_d_s(tensor_s)
        q3_s = self.layer_e_s(q3_s)
        q3_s = tensor_s - q3_s
        q3_s.add_(q1_s)
        tensor_s = pst(q3_s, tensor_u, self.lam)
        return tensor_u, tensor_s


class MCCSR_Net(nn.Module):
    def __init__(self):
        super(MCCSR_Net, self).__init__()
        self.channel = 1
        self.out_channel = 1
        self.filters = 64
        self.kernel_sizes_b = 3
        self.kernel_sizes_d = 3
        self.num_t = 4
        self.dilation_b = 3
        self.dilation_d = 1
        self.conv_u = nn.ModuleList([BasicUnit(self.filters, self.channel, self.kernel_sizes_b, self.dilation_b) for i in range(self.num_t)])
        self.conv_s = nn.ModuleList([BasicUnit(self.filters, self.channel, self.kernel_sizes_b, self.dilation_b) for i in range(self.num_t)])
        self.conv_v = nn.ModuleList([BasicUnit(self.filters, self.channel, self.kernel_sizes_d, self.dilation_d) for i in range(self.num_t)])
        self.conv_t = nn.ModuleList([BasicUnit(self.filters, self.channel, self.kernel_sizes_d, self.dilation_d) for i in range(self.num_t)])
        self.net_c = nn.ModuleList([Prediction(num_channels=self.channel, kernel_size=self.kernel_sizes_b, dilation=self.dilation_b) for i in range(self.num_t)])  # BSMP
        self.net_t = nn.ModuleList([Prediction(num_channels=self.channel, kernel_size=self.kernel_sizes_d, dilation=self.dilation_d) for i in range(self.num_t)])  # DSMP
        self.mean_pooling_b = torch.nn.AvgPool2d(19, stride=1, padding=9)
        self.mean_pooling_d = torch.nn.AvgPool2d(7, stride=1, padding=3)
        self.conv_r1 = BasicUnit(self.out_channel * 2, self.out_channel, self.kernel_sizes_d, self.dilation_d)
        self.conv_r2 = BasicUnit(self.out_channel * 2, self.out_channel, self.kernel_sizes_d, self.dilation_d)
        self.conv_r3 = nn.Conv2d(self.out_channel * 2, self.out_channel, 1, padding=0, bias=False)
        self.conv_1 = BasicUnit(self.filters, self.channel, self.kernel_sizes_b, self.dilation_b)
        self.conv_2 = BasicUnit(self.filters, self.channel, self.kernel_sizes_b, self.dilation_b)
        self.conv_3 = BasicUnit(self.filters, self.channel, self.kernel_sizes_d, self.dilation_d)
        self.conv_4 = BasicUnit(self.filters, self.channel, self.kernel_sizes_d, self.dilation_d)

    def forward(self, x, y):
        u = 0
        s = 0
        v = 0
        t = 0
        u, s = self.net_c[0](0, u, s, x, y)
        p_v = x - self.conv_u[0](u)
        p_t = y - self.conv_s[0](s)
        v, t = self.net_t[0](0, v, t, p_v, p_t)
        p_u = x - self.conv_v[0](v)
        p_s = y - self.conv_t[0](t)
        for i in range(1, self.num_t):
            u, s = self.net_c[i](i, u, s, p_u, p_s)
            p_v = x - self.conv_u[i](u)
            p_t = y - self.conv_s[i](s)
            v, t = self.net_t[i](i, v, t, p_v, p_t)
            p_u = x - self.conv_v[i](v)
            p_s = y - self.conv_t[i](t)

        a1 = torch.abs(u).sum(dim=1, keepdim=True)
        a2 = torch.abs(s).sum(dim=1, keepdim=True)
        aa1 = self.mean_pooling_b(a1)
        aa2 = self.mean_pooling_b(a2)

        b1 = torch.abs(v).sum(dim=1, keepdim=True)
        b2 = torch.abs(t).sum(dim=1, keepdim=True)
        bb1 = self.mean_pooling_d(b1)
        bb2 = self.mean_pooling_d(b2)

        x_low = self.conv_1(u)
        y_low = self.conv_2(s)
        x_high = self.conv_3(v)
        y_high = self.conv_4(t)
        m_b = self.conv_r1(torch.cat((aa1, aa2), dim=1))
        m_d = self.conv_r2(torch.cat((bb1, bb2), dim=1))
        m = self.conv_r3(torch.cat((m_b, m_d), dim=1))
        return x_low, y_low, x_high, y_high, m_b, m_d, m


if __name__ == '__main__':
    net = MCCSR_Net()
    print(net)