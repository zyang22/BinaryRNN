import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from telegram.ext import Updater
updater = Updater(token='')#your bot token
sepehr = 0#your telegram id
bot_users = [sepehr]

# TODO: std loss on hidden state, hidden_residual

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden, _ = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_weight_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    #hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        #hidden = repackage_hidden(hidden)
        hidden = model.init_hidden(args.batch_size)
        model.zero_grad()
        output, hidden, rnn_outs = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        main_loss = loss.clone()
        #0: initial_hidden_state
        #1: embedding(encoder, decoder weights)
        #2,3: rnn_weights
        #4,5: rnn_biases
        #6: decoder_bias
        for counter, p in enumerate(model.parameters()):
            if counter == 0:
                tmp = p.abs()
                tmp1 = torch.std(tmp, unbiased=False)
                tmp2 = tmp.mean()
                tmp = tmp1 / (tmp2 + 1.0)
                if tmp[0].data[0] > 0.001: # is it because of the zero?
                    loss += tmp*60./4./5./2.
            elif counter < 4:
                tmp = p.abs()
                tmp1 = tmp.std(dim=1, unbiased=False)
                tmp2 = tmp.mean(dim=1)
                tmp3 = (tmp1/(tmp2+1.0))
                tmp = tmp3.mean()
                if tmp[0].data[0] > 0.001:
                    loss += tmp*200./4./4./2.
        tmp = rnn_outs.abs()
        tmp1 = tmp.std(dim=2, unbiased=False)
        tmp2 = tmp.mean(dim=2)
        tmp3 = (tmp1 / (tmp2 + 1.0))
        tmp = tmp3.mean()
        if tmp[0].data[0] > 0.001:
            loss += tmp*50./6./4./2.
        if loss[0].data[0] != loss[0].data[0]:
            continue
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)
            p.data.clamp_(min=-1000., max=1000.)

        total_loss += main_loss.data
        total_weight_loss += loss.data - main_loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss_ppl {:5.2f} | ppl {:8.2f} | weight_loss {:5.2f} | total_loss {:5.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss),
                              total_weight_loss[0] / args.log_interval,
                              cur_loss + total_weight_loss[0] / args.log_interval))
            total_loss = 0
            total_weight_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        if epoch == 16:
            model.changed_to_bin = True
            args.log_interval = 25
            best_val_loss = None
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        try:
            for bot_user in bot_users:
                updater.bot.send_message(chat_id=bot_user, text='| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        except:
            pass
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

'''
full net:
| epoch  20 |  1327/ 1327 batches | lr 0.02 | ms/batch 17.40 | loss_ppl  3.80 | ppl    44.77 | weight_loss  4.72
| end of epoch  20 | time: 24.60s | valid loss  4.79 | valid ppl   120.46
| End of training | test loss  4.74 | test ppl   114.82

tied weights loss @ x40:
| epoch  10 |  1200/ 1327 batches | lr 5.00 | ms/batch 24.00 | loss_ppl  4.92 | ppl   137.47 | weight_loss  0.14 | total_loss  5.06
| end of epoch  10 | time: 32.59s | valid loss  5.20 | valid ppl   181.53
| End of training | test loss  5.17 | test ppl   175.86
-----------------------------------------------------------------------------------------
| epoch  20 |  1200/ 1327 batches | lr 0.02 | ms/batch 23.96 | loss_ppl  4.85 | ppl   128.14 | weight_loss  0.05 | total_loss  4.91
| end of epoch  20 | time: 32.68s | valid loss  5.16 | valid ppl   173.93
| End of training | test loss  5.13 | test ppl   168.53
'''


'''| end of epoch  13 | time: 1099.74s | valid loss  6.38 | valid ppl   590.18 (60/4, 200/4, 0), 574@test'''
