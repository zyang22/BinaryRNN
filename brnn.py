import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from utils import smoothBinerazer


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert (len(weight) == total_layers)
        next_hidden = []
        if lstm:
            hidden = list(zip(*hidden))
        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j
                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)
            input = torch.cat(all_output, input.dim() - 1)
            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)
        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())
        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)
        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        return hidden, output

    return forward


def variable_recurrent_factory(batch_sizes):
    def fac(inner, reverse=False):
        if reverse:
            return VariableRecurrentReverse(batch_sizes, inner)
        else:
            return VariableRecurrent(batch_sizes, inner)

    return fac


def VariableRecurrent(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size
            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size
            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()
        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)
        return hidden, output

    return forward


def VariableRecurrentReverse(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size
            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])
        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward


def AutogradRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False,
                dropout=0, train=True, bidirectional=False, batch_sizes=None,
                dropout_state=None):
    if mode == 'LSTM':
        cell = None
    elif mode == 'GRU':
        cell = GRUCellOrig
    else:
        raise Exception('Unknown mode: {}'.format(mode))
    if batch_sizes is None:
        rec_factory = Recurrent
    else:
        rec_factory = variable_recurrent_factory(batch_sizes)
    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)
    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM'),
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)
        nexth, output = func(input, hidden, weight)
        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)
        return output, nexth

    return forward


def RNN(*args, **kwargs):
    def forward(input, *fargs, **fkwargs):
        func = AutogradRNN(*args, **kwargs)
        return func(input, *fargs, **fkwargs)

    return forward


class BRNN(nn.RNNBase):
    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_())
            if self.mode == 'LSTM':
                hx = (hx, hx)
        func = RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state
        )
        bin_weights = [
            [smoothBinerazer(this_weight, is_embedding=True) if (len(this_weight.size()) == 2) else this_weight for
             this_weight in self.all_weights[0]]]  # TODO handle multilayer case
        output, hidden = func(input, bin_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden


def blinear(x, w, b=None):
    if b is None:
        return F.linear(x, smoothBinerazer(w, True))
    else:
        return F.linear(x, smoothBinerazer(w, True), b)


def GRUCellOrig(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(smoothBinerazer(hidden, True), w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)
    reset_gate = F.sigmoid(i_r + h_r)
    input_gate = F.sigmoid(i_i + h_i)
    new_gate = F.tanh(i_n + reset_gate * h_n)
    hy = new_gate + input_gate * (hidden - new_gate)
    return hy


class BLinear(nn.Linear):
    def forward(self, input):
        return blinear(input, self.weight, self.bias)
