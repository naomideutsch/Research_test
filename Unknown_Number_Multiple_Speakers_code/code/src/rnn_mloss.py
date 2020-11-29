import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
dbstop = pdb.set_trace
import sys

from utils import overlap_and_add
from torch.autograd import Variable

EPS = 1e-8


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*self.num_direction, input_size)

    def forward(self, input):

        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)

        return rnn_output


class GatedRNN(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=True):
        super(GatedRNN, self).__init__()

        self.rnn = SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True)
        self.gate_rnn = SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True)
        self.proj = nn.Linear(input_size*2, input_size)
        self.input_size = input_size

    def forward(self, input):

        output = self.rnn(input) * self.gate_rnn(input)
        output = torch.cat([output, input], 2)
        output = self.proj(output.contiguous().view(-1, output.shape[2])).view([input.shape[0], input.shape[1], self.input_size])

        return output

# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """
    def __init__(self, rnn_type, input_size, hidden_size, output_size, nspk,
                 dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):

            self.row_rnn.append(GatedRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.col_rnn.append(GatedRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size*nspk, 1))


    def forward(self, input, f1s2):

        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        batch_size, _, dim1, dim2 = input.shape
        output = input
        output_all = []

        for i in range(len(self.row_rnn)):

            row_input = output.permute(0,3,2,1).contiguous().view(batch_size*dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0,3,2,1).contiguous()  # B, N, dim1, dim2
            output = output + row_output


            col_input = output.permute(0,2,3,1).contiguous().view(batch_size*dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0,3,1,2).contiguous()  # B, N, dim1, dim2
            output = output + col_output

            output_a = self.output(output)
            output_all.append(output_a)

        output = self.output(output)

        return output, output_all

# base module for deep DPRNN
class DPRNN_base(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, num_spk=2,
                 layer=4, segment_size=100, bidirectional=True, model_type='DPRNN',
                 rnn_type='LSTM'):
        super(DPRNN_base, self).__init__()

        assert model_type in ['DPRNN', 'DPRNN_TAC'], "model_type can only be 'DPRNN' or 'DPRNN_TAC'."

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.model_type = model_type
        self.eps = 1e-8

        # DPRNN model
        self.DPRNN = getattr(sys.modules[__name__], model_type)(rnn_type, self.feature_dim, self.hidden_dim, self.feature_dim, self.num_spk,
                                         num_layers=layer, bidirectional=bidirectional)

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:,:,:-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:,:,segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size*2)  # B, N, K, L

        input1 = input[:,:,:,:segment_size].contiguous().view(batch_size, dim, -1)[:,:,segment_stride:]
        input2 = input[:,:,:,segment_size:].contiguous().view(batch_size, dim, -1)[:,:,:-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:,:,:-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):
        pass

# DPRNN for beamforming filter estimation
class BF_module(DPRNN_base):
    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)

    def forward(self, input, f1s2, num_mic=1):

        if self.model_type == 'DPRNN':
            # input: (B, N, T)
            batch_size, N, seq_length = input.shape
            ch = 1
        elif self.model_type == 'DPRNN_TAC':
            # input: (B, ch, N, T)
            batch_size, ch, N, seq_length = input.shape

        enc_feature = input.view(batch_size*ch, N, seq_length)  # B*ch, N, T

        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)  # B*ch, N, L, K

        # pass to DPRNN
        if self.model_type == 'DPRNN':
            output, output_all = self.DPRNN(enc_segments, f1s2)

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B*ch*nspk, N, T

        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_feature(output_all[ii], enc_rest)  # B*ch*nspk, N, T
            output_all_wav.append(output_ii)

        return output, output_all_wav

class GateRNNNet(nn.Module):
    def __init__(self, N, L, B, H, P, X, R, C, sr, segment, rnn_b_layer, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(GateRNNNet, self).__init__()
        # Hyper-parameter
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C, self.sr, self.segment = N, L, B, H, P, X, R, C, sr, segment
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.rnn_b_layer = rnn_b_layer

        # Components
        self.encoder = Encoder(L, N)
        self.decoder = Decoder(N, L)

        self.win_len = segment 
        self.sr = sr 
        self.context_len = 16
        self.window = int(self.sr * self.win_len / 1000)
        self.context = int(self.sr * self.context_len / 1000)
        self.stride = self.window // 2
        self.enc_dim = self.N 
        self.feature_dim = self.N 
        self.hidden_dim = 128
        self.layer = self.rnn_b_layer 
        self.filter_dim = self.context*2+1 
        self.num_spk = self.C
        self.segment_size = int(np.sqrt(2*self.sr*self.win_len/(self.L/2))) # sqrt(2*L) 
        self.ref_BF = BF_module(self.filter_dim+self.enc_dim, self.feature_dim, self.hidden_dim, self.filter_dim, self.num_spk, self.layer, self.segment_size, model_type='DPRNN')

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """

        # tas-convnet without enc/mul/dec
        mixture_w = self.encoder(mixture)  
        est_mask, output_all = self.ref_BF(mixture_w, 1)
        est_mask = est_mask.view(mixture.shape[0], self.C, self.N, mixture_w.shape[2])
        est_source = self.decoder(est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))

        est_mask_f_all = []
        for ii in range(len(output_all)):
            est_mask_f_ii = output_all[ii].view(mixture.shape[0], self.C, self.N, mixture_w.shape[2])
            est_mask_f_ii = self.decoder(est_mask_f_ii)
            est_mask_f_ii = F.pad(est_mask_f_ii, (0, T_origin - T_conv))
            est_mask_f_all.append(est_mask_f_ii)


        return est_source, est_mask_f_all

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['N'], package['L'], package['B'], package['H'],
                    package['P'], package['X'], package['R'], package['C'], package['sr'], package['segment'],
                    norm_type=package['norm_type'], causal=package['causal'], rnn_b_layer=package['rnn_b_layer'],
                    mask_nonlinear=package['mask_nonlinear'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'N': model.N, 'L': model.L, 'B': model.B, 'H': model.H,
            'P': model.P, 'X': model.X, 'R': model.R, 'C': model.C, 'sr': model.sr, 'segment': model.segment,
            'norm_type': model.norm_type, 'causal': model.causal, 'rnn_b_layer': model.rnn_b_layer,
            'mask_nonlinear': model.mask_nonlinear,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w

class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.N, self.L = N, L
        # Components
        # self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """

        est_source = torch.transpose(est_mask, 2, 3)
        est_source = nn.AvgPool2d((1, self.L))(est_source)
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T

        return est_source



def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    else: # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y

