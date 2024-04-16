import os
import torch
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


def data_normal(X):
    min = X.min()
    max = X.max()
    X_norm = (X - min)/(max - min)
    return X_norm


class Prediction(nn.Module):
    def __init__(self, num_channels):
        super(Prediction, self).__init__()
        self.num_layers = 3
        self.in_channel = num_channels
        self.kernel_size = 3
        self.num_filters = 64
        self.layer_in_u = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                    kernel_size=self.kernel_size, padding=1, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_in_u.weight.data)

        self.layer_in_v = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                    kernel_size=self.kernel_size, padding=1, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_in_v.weight.data)

        self.lam_in = nn.Parameter(torch.full([self.num_filters, 1, 1], 0.01))  # 阈值

        self.lam_i = []
        self.layer_down_u = []
        self.layer_down_v = []
        self.layer_up_u = []
        self.layer_up_v = []
        for i in range(self.num_layers):
            down_conv_u = 'down_conv_u_{}'.format(i)
            up_conv_u = 'up_conv_u_{}'.format(i)
            down_conv_v = 'down_conv_v_{}'.format(i)
            up_conv_v = 'up_conv_v_{}'.format(i)
            lam_id = 'lam_{}'.format(i)
            layer_1_u = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,
                                  kernel_size=self.kernel_size, padding=1, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_1_u.weight.data)
            setattr(self, down_conv_u, layer_1_u)
            self.layer_down_u.append(getattr(self, down_conv_u))
            layer_1_v = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,
                                  kernel_size=self.kernel_size, padding=1, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_1_v.weight.data)
            setattr(self, down_conv_v, layer_1_v)
            self.layer_down_v.append(getattr(self, down_conv_v))

            layer_2_u = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                  kernel_size=self.kernel_size, padding=1, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_2_u.weight.data)
            setattr(self, up_conv_u, layer_2_u)
            self.layer_up_u.append(getattr(self, up_conv_u))
            layer_2_v = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                  kernel_size=self.kernel_size, padding=1, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_2_v.weight.data)
            setattr(self, up_conv_v, layer_2_v)
            self.layer_up_v.append(getattr(self, up_conv_v))

            lam_ = nn.Parameter(torch.full([self.num_filters, 1, 1], 0.01))
            setattr(self, lam_id, lam_)
            self.lam_i.append(getattr(self, lam_id))

    def forward(self, mod_1, mod_2):
        p1 = self.layer_in_u(mod_1)
        q1 = self.layer_in_v(mod_2)
        tensor_v = torch.zeros(p1.shape)
        tensor_u = pst(p1, tensor_v, self.lam_in)
        tensor_v = pst(q1, tensor_u, self.lam_in)
        for i in range(self.num_layers):
            p3 = self.layer_down_u[i](tensor_u)
            p3 = self.layer_up_u[i](p3)
            p3 = tensor_u - p3
            p3.add_(p1)
            tensor_u = pst(p3, tensor_v, self.lam_i[i])
            q3 = self.layer_down_v[i](tensor_v)
            q3 = self.layer_up_v[i](q3)
            q3 = tensor_v - q3
            q3.add_(q1)
            tensor_v = pst(q3, tensor_u, self.lam_i[i])
        return tensor_u, tensor_v


class CCSR_Net(nn.Module):
    def __init__(self):
        super(CCSR_Net, self).__init__()
        self.channel = 3
        self.kernel_size = 3
        self.out_channel = 1
        self.filters = 64
        self.mean_pooling = torch.nn.AvgPool2d(19, stride=1, padding=9)
        self.conv_1 = nn.Conv2d(in_channels=self.out_channel * 2, out_channels= self.out_channel, kernel_size=self.kernel_size,
                                stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv_1.weight.data)
        self.net = Prediction(num_channels=self.channel)

    def forward(self, x, y):
        u, v = self.net(x, y)
        a1 = torch.abs(u).sum(dim=1, keepdim=True)
        a2 = torch.abs(v).sum(dim=1, keepdim=True)
        aa1 = self.mean_pooling(a1)
        aa2 = self.mean_pooling(a2)
        m = self.conv_1(torch.cat((aa1, aa2), dim=1))
        return m


if __name__ == '__main__':
    net = CCSR_Net()
    print(net)
