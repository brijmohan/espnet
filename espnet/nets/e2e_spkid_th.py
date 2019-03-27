#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math
import sys

from argparse import Namespace
import editdistance

import chainer
import numpy as np
import random
import six
import torch
import torch.nn.functional as F
import warpctc_pytorch as warp_ctc

from chainer import reporter
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.e2e_asr_common import label_smoothing_dist

from torch.autograd import Function

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5

# Gradient reversal alpha
GRL_ALPHA = 0.5

# ------------- Utility functions --------------------------------------------------------------------------------------
def to_cuda(m, x):
    """Function to send tensor into corresponding device

    :param torch.nn.Module m: torch module
    :param torch.Tensor x: torch tensor
    :return: torch tensor located in the same place as torch module
    :rtype: torch.Tensor
    """
    assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)


def pad_list(xs, pad_value):
    """Function to pad values

    :param list xs: list of torch.Tensor [(L_1, D), (L_2, D), ..., (L_B, D)]
    :param float pad_value: value for padding
    :return: padded tensor (B, Lmax, D)
    :rtype: torch.Tensor
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths):
    """Function to make mask tensor containing indices of padded part

    e.g.: lengths = [5, 3, 2]
          mask = [[0, 0, 0, 0 ,0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1]]

    :param list lengths: list of lengths (B)
    :return: mask tensor containing indices of padded part (B, Tmax)
    :rtype: torch.Tensor
    """
    bs = int(len(lengths))
    maxlen = int(max(lengths))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    return seq_range_expand >= seq_length_expand


def mask_by_length(xs, length, fill=0):
    assert xs.size(0) == len(length)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(length):
        ret[i, :l] = xs[i, :l]
    return ret


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Function to calculate accuracy

    :param torch.Tensor pad_outputs: prediction tensors (B*Lmax, D)
    :param torch.Tensor pad_targets: target tensors (B, Lmax, D)
    :param int ignore_label: ignore label id
    :retrun: accuracy value (0.0 - 1.0)
    :rtype: float
    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0),
        pad_targets.size(1),
        pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def get_last_yseq(exp_yseq):
    last = []
    for y_seq in exp_yseq:
        last.append(y_seq[-1])
    return last


def append_ids(yseq, ids):
    if isinstance(ids, list):
        for i, j in enumerate(ids):
            yseq[i].append(j)
    else:
        for i in range(len(yseq)):
            yseq[i].append(ids)
    return yseq


def expand_yseq(yseqs, next_ids):
    new_yseq = []
    for yseq in yseqs:
        for next_id in next_ids:
            new_yseq.append(yseq[:])
            new_yseq[-1].append(next_id)
    return new_yseq


def index_select_list(yseq, lst):
    new_yseq = []
    for l in lst:
        new_yseq.append(yseq[l][:])
    return new_yseq


def index_select_lm_state(rnnlm_state, dim, vidx):
    if isinstance(rnnlm_state, dict):
        new_state = {}
        for k, v in rnnlm_state.items():
            new_state[k] = [torch.index_select(vi, dim, vidx) for vi in v]
    elif isinstance(rnnlm_state, list):
        new_state = []
        for i in vidx:
            new_state.append(rnnlm_state[int(i)][:])
    return new_state


class Reporter(chainer.Chain):
    def report(self, loss_adv, acc_adv):
        reporter.report({'loss_adv': loss_adv}, self)
        reporter.report({'acc_adv': acc_adv}, self)


# TODO(watanabe) merge Loss and E2E: there is no need to make these separately
class Loss(torch.nn.Module):
    """Multi-task learning loss module

    :param torch.nn.Module predictor: E2E model instance
    :param float mtlalpha: mtl coefficient value (0.0 ~ 1.0)
    """

    def __init__(self, predictor):
        super(Loss, self).__init__()
        self.predictor = predictor
        self.reporter = Reporter()

    def forward(self, xs_pad, ilens, y_adv):
        '''Multi-task learning loss forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_adv: batch of speaker ids (B, z_i)
        :return: loss value
        :rtype: torch.Tensor
        '''
        loss_adv, acc_adv = self.predictor(xs_pad, ilens, y_adv)
        loss_adv_data = float(loss_adv.item())

        if not math.isnan(loss_adv_data):
            self.reporter.report(loss_adv_data, acc_adv)
        else:
            logging.warning('loss (=%f) is not correct', loss_adv_data)

        return loss_adv


class E2E(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param int odim_adv: dimension of outputs for adversarial class
    :param namespace args: argument namespace containing options
    """

    def __init__(self, idim, odim_adv, args):
        super(E2E, self).__init__()
        self.verbose = args.verbose
        self.outdir = args.outdir

        # Adversarial branch
        self.adv = SpeakerAdv(odim_adv, idim, args.adv_units,
                                args.adv_layers,
                                dropout_rate=args.dropout_rate)

        # weight initialization
        self.init_like_chainer()

        self.logzero = -10000000000.0

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        #self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        #for l in six.moves.range(len(self.dec.decoder)):
        #    set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, y_adv):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        # 4. Adversarial loss
        logging.info("Computing speaker loss")
        #rev_hs_pad = ReverseLayerF.apply(hs_pad, GRL_ALPHA)
        loss_adv, acc_adv = self.adv(xs_pad, ilens, y_adv)

        return loss_adv, acc_adv

    def recognize(self, x):
        prev = self.training
        self.eval()
        ilen = [x.shape[0]]

        y = self.adv.predict(x, ilen)

        if prev:
            self.train()

        return y

    def recognize_batch(self, x):
        prev = self.training
        self.eval()

        ilens = np.fromiter((xx.shape[0] for xx in x), dtype=np.int64)
        hs = [to_cuda(self, torch.from_numpy(np.array(xx, dtype=np.float32)))
              for xx in x]

        xpad = pad_list(hs, 0.0)

        y = self.adv.predict_batch(xpad, ilens)

        if prev:
            self.train()

        return y



#-------------------- Adversarial Network ------------------------------------
#Brij: Added the gradient reversal layer
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

# Brij: Added to classify speakers from encoder projections
class SpeakerAdv(torch.nn.Module):
    """ Speaker adversarial module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    """

    def __init__(self, odim, eprojs, advunits, advlayers, dropout_rate=0.2):
        super(SpeakerAdv, self).__init__()
        self.advunits = advunits
        self.advlayers = advlayers
        #self.advnet = torch.nn.LSTM(eprojs, advunits, self.advlayers,
        #                            batch_first=True, dropout=dropout_rate)
        '''
        linears = [torch.nn.Linear(eprojs, advunits), torch.nn.ReLU(),
                   torch.nn.Dropout(p=dropout_rate)]
        for l in six.moves.range(1, self.advlayers):
            linears.extend([torch.nn.Linear(advunits, advunits),
                            torch.nn.ReLU(), torch.nn.Dropout(p=dropout_rate)])
        self.advnet = torch.nn.Sequential(*linears)
        '''
        self.vgg = VGG2L(1)
        layer_arr = [torch.nn.Linear(get_vgg2l_odim(eprojs, in_channel=1),
                                          advunits), torch.nn.ReLU()]
        for l in six.moves.range(1, self.advlayers):
            layer_arr.extend([torch.nn.Linear(advunits, advunits),
                            torch.nn.ReLU(), torch.nn.Dropout(p=dropout_rate)])
        self.advnet = torch.nn.Sequential(*layer_arr)
        self.output = torch.nn.Linear(advunits, odim)

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(self.advlayers, hs_pad.size(0), self.advunits)

    def forward(self, hs_pad, hlens, y_adv):
        '''Adversarial branch forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor y_adv: batch of speaker class (B, #Speakers)
        :return: loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        '''

        # initialization
        #logging.info("initializing hidden states for LSTM")
        #h_0 = self.zero_state(hs_pad)
        #c_0 = self.zero_state(hs_pad)

        logging.info("Passing encoder output through advnet %s",
                     str(hs_pad.shape))

        #self.advnet.flatten_parameters()
        #out_x, (h_0, c_0) = self.advnet(hs_pad, (h_0, c_0))
        vgg_x, _ = self.vgg(hs_pad, hlens)
        out_x = self.advnet(vgg_x)

        logging.info("vgg output size = %s", str(vgg_x.shape))
        logging.info("advnet output size = %s", str(out_x.shape))
        logging.info("speaker target size = %s", str(y_adv.shape))
        
        y_hat = self.output(out_x)

        # Create labels tensor by replicating speaker label
        batch_size, avg_seq_len, out_dim = y_hat.size()

        labels = torch.zeros([batch_size, avg_seq_len], dtype=torch.int64)
        for ix in range(batch_size):
            labels[ix, :] = y_adv[ix]

        # Mean over sequence length
        #y_hat = torch.mean(y_hat, 1)
        #h_0.detach_()
        #c_0.detach_()

        # Convert tensors to desired shape
        y_hat = y_hat.view((-1, out_dim))
        labels = labels.contiguous().view(-1)
        labels = to_cuda(self, labels.long())
        logging.info("speaker output size = %s", str(y_hat.shape))
        logging.info("artificial label size = %s", str(labels.shape))

        loss = F.cross_entropy(y_hat, labels, size_average=False)
        logging.info("Speaker loss = %f", loss.item())
        acc = th_accuracy(y_hat, labels.unsqueeze(0), -1)
        logging.info("Speaker accuracy = %f", acc)

        return loss, acc

    def predict_batch(self, hs_pad, hlens):
        '''Adversarial branch forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor y_adv: batch of speaker class (B, #Speakers)
        :return: loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        '''

        logging.info("Passing encoder output through advnet %s",
                     str(hs_pad.shape))

        #self.advnet.flatten_parameters()
        #out_x, (h_0, c_0) = self.advnet(hs_pad, (h_0, c_0))
        vgg_x, _ = self.vgg(hs_pad, hlens)
        out_x = self.advnet(vgg_x)

        logging.info("vgg output size = %s", str(vgg_x.shape))
        logging.info("advnet output size = %s", str(out_x.shape))
        logging.info("speaker target size = %s", str(y_adv.shape))
        
        y_hat = self.output(out_x)
        # Create labels tensor by replicating speaker label
        batch_size, avg_seq_len, out_dim = y_hat.size()
        logging.info("speaker output size = %s", str(y_hat.shape))

        pred = y_hat.argmax(2)

        return pred.tolist()

class VGG2L(torch.nn.Module):
    """VGG-like module

    :param int in_channel: number of input channels
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        '''VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128)
        :rtype: torch.Tensor
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens
