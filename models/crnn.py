from torch import nn
from torchvision import ops
import torch


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels))

    def forward(self, x):
        residual = self.resblock(x)
        return x + residual


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # padding_mode='zeros'
        self.conv2d_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, stride, padding)
        self.deformconv2d = ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # self.init_offset()
        self.reset_parameters()

    # def init_offset(self):
    #     torch.nn.init.zeros_(self.conv2d_offset.weight)
    #     torch.nn.init.zeros_(self.conv2d_offset.bias)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.conv2d_offset.weight)
        torch.nn.init.zeros_(self.conv2d_offset.bias)

    def forward(self, x):
        return self.deformconv2d(x, self.conv2d_offset(x))


class CNN(nn.Module):
    def __init__(self, channel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

    def forward(self, x):
        conv = self.cnn(x)
        return conv

class CNN_regularized_do(nn.Module):
    def __init__(self, channel_size):
        super(CNN_regularized_do, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Dropout(0.2),
            nn.Conv2d(512, 512, 2, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return self.cnn(x)


class DeformCNN(nn.Module):
    def __init__(self, channel_size):
        super(DeformCNN, self).__init__()
        self.cnn = nn.Sequential(
            DeformConv2d(channel_size, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            DeformConv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            DeformConv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            DeformConv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            DeformConv2d(512, 512, 2, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.cnn(x)


class CNND5L(nn.Module):
    def __init__(self, channel_size):
        super(CNND5L, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(48, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 80, 3, 1, 1, bias=False), nn.BatchNorm2d(80), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        conv = self.cnn(x)
        return conv


class CNN5L(nn.Module):
    def __init__(self, channel_size):
        super(CNN5L, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 80, 3, 1, 1, bias=False), nn.BatchNorm2d(80), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        conv = self.cnn(x)
        return conv


class CNN5LNOBN(nn.Module):
    def __init__(self, channel_size):
        super(CNN5LNOBN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 16, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 64, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 80, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        conv = self.cnn(x)
        return conv


class DeformCNN5LNOBN(nn.Module):
    def __init__(self, channel_size):
        super(DeformCNN5LNOBN, self).__init__()
        self.cnn = nn.Sequential(
            DeformConv2d(channel_size, 16, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(16, 32, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(32, 48, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(48, 64, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True),
            DeformConv2d(64, 80, 3, 1, 1, bias=False), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        conv = self.cnn(x)
        return conv


class DeformCNN5L(nn.Module):
    def __init__(self, channel_size):
        super(DeformCNN5L, self).__init__()
        self.cnn = nn.Sequential(
            DeformConv2d(channel_size, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(16, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(32, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(48, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True),
            DeformConv2d(64, 80, 3, 1, 1, bias=False), nn.BatchNorm2d(80), nn.LeakyReLU(inplace=True))
    def forward(self, x):
        conv = self.cnn(x)
        return conv


class DeformCNND5L(nn.Module):
    def __init__(self, channel_size):
        super(DeformCNND5L, self).__init__()
        self.cnn = nn.Sequential(
            DeformConv2d(channel_size, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(16, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            DeformConv2d(32, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            DeformConv2d(48, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            DeformConv2d(64, 80, 3, 1, 1, bias=False), nn.BatchNorm2d(80), nn.LeakyReLU(inplace=True))
    def forward(self, x):
        conv = self.cnn(x)
        return conv


class DeformCNN_regularized(nn.Module):
    def __init__(self, channel_size):
        super(DeformCNN_regularized, self).__init__()
        self.cnn = nn.Sequential(
            DeformConv2d(channel_size, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            DeformConv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            DeformConv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            DeformConv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            DeformConv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            DeformConv2d(512, 512, 2, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return self.cnn(x)

class DeformCNN_regularized_do(nn.Module):
    def __init__(self, channel_size):
        super(DeformCNN_regularized_do, self).__init__()
        self.cnn = nn.Sequential(
            DeformConv2d(channel_size, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            DeformConv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            DeformConv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            DeformConv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Dropout(0.2),
            DeformConv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            DeformConv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Dropout(0.2),
            DeformConv2d(512, 512, 2, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return self.cnn(x)


class BLSTMD5L(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BLSTMD5L, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=0.5, num_layers=5)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.embedding = nn.Linear(hidden_size * 2, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
    def forward(self, x):
        x = self.dropout1(x)
        recurrent, _ = self.rnn(x)
        recurrent = self.dropout2(recurrent)
        output = self.embedding(recurrent)
        output = self.logsoftmax(output)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=False, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        output = self.logsoftmax(output)
        return output


class BLSTM4L(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BLSTM4L, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=0.5, num_layers=4)
        self.embedding = nn.Linear(hidden_size * 2, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        output = self.logsoftmax(output)
        return output


class BLSTM3L(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BLSTM3L, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=0.5, num_layers=3)
        self.embedding = nn.Linear(hidden_size * 2, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        output = self.logsoftmax(output)
        return output


class BGRU(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(BGRU, self).__init__()

        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(n_hidden * 2, n_out)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        output = self.logsoftmax(output)
        return output


class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(hidden_size * 2, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        output = self.logsoftmax(output)
        return output


class CRNN(nn.Module):
    def __init__(self, cnn, rnn):
        super(CRNN, self).__init__()
        self.cnn = cnn
        self.rnn = rnn
    def forward(self, x):
        conv = self.cnn(x)
        b, w = conv.shape[0], conv.shape[3]
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output
