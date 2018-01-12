import torch
import torch.nn as nn
from brnn import BRNN, BLinear
from utils import smoothBinerazer

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers=1, dropout=0., tie_weights=True, use_cuda=True):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.bdecoder = BLinear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.bdecoder.weight = self.decoder.weight
        self.bdecoder.bias = self.decoder.bias

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        hidden0 = torch.zeros(1, 1, nhid)
        if use_cuda:
            hidden0 = hidden0.cuda()
        else:
            hidden0 = hidden0
        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)
        self.brnn = BRNN(rnn_type, ninp, nhid, nlayers)
        self.brnn.weight_hh_l0 = self.rnn.weight_hh_l0
        self.brnn.weight_ih_l0 = self.rnn.weight_ih_l0
        self.brnn.bias_hh_l0 = self.rnn.bias_hh_l0
        self.brnn.bias_ih_l0 = self.rnn.bias_ih_l0
        self.changed_to_bin = False

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        if self.changed_to_bin:
            emb = smoothBinerazer(emb, is_embedding=True)
            rnn_outputs, hidden = self.brnn(emb, hidden)
            output = smoothBinerazer(rnn_outputs, is_embedding=True)
            output = self.drop(output)
        else:
            rnn_outputs, hidden = self.rnn(emb, hidden)
            output = self.drop(rnn_outputs)
        if self.changed_to_bin:
            decoded = self.bdecoder(output.view(output.size(0)*output.size(1), output.size(2)))
        else:
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, rnn_outputs

    def init_hidden(self, bsz):
        return self.hidden0.repeat(1, bsz, 1)

