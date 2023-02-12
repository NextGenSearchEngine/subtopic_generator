from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle # to dump and load pretrained glove vectors 
import copy   # to make deepcopy of python lists and dictionaries
import operator
import numpy as np
from pandas import DataFrame # to visualize the glove word embeddings in form of DataFrame
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx = 0
word2idx = {}
vectors = []
words = []
with open(f'/mnt/nvme5/wikipedia/glove.6B.300d.txt', 'rb') as f:
    for l in tqdm(f, total=400000):
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.asarray(line[1:],'float32')
        vectors.append(vect)

glove = {w: vectors[word2idx[w]] for w in words}

sos_index = word2idx['sos']
eos_index = word2idx['eos']
sos_swap_word = words[0]
eos_swap_word = words[1]
words[0], words[sos_index] = words[sos_index], words[0]
words[1], words[eos_index] = words[eos_index], words[1]
word2idx[sos_swap_word], word2idx['sos'] = word2idx['sos'], word2idx[sos_swap_word]
word2idx[eos_swap_word], word2idx['eos'] = word2idx['eos'], word2idx[eos_swap_word]

word2idx = { k : v for k , v in sorted(word2idx.items(), key=operator.itemgetter(1))}

class InputLang:
    def __init__(self, name):
        self.name = name
        self.word2index = { k : v for k , v in sorted(word2idx.items(), key=operator.itemgetter(1))}
        self.word2count = { word : 1 for word in words }
        self.index2word = { i : word for word, i in word2idx.items() }
        self.n_words = 400001

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

SOS_token = 0
EOS_token = 1

class OutputLang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "sos", 1: "eos"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs():
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('/mnt/nvme5/wikipedia/wikipedia_dataset_clean.txt', encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
   
    pairs = [list((p)) for p in pairs]

    input_lang = InputLang('categories')
    output_lang = OutputLang('subtopics')
    
    return input_lang, output_lang, pairs


MAX_LENGTH = 60


def filterPair(p):
    return len(p[1].split(' ')) < MAX_LENGTH and \
        len(p[0].split(' ')) < MAX_LENGTH 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData():
    input_lang, output_lang, pairs = readLangs()
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

 
input_lang, output_lang, pairs = prepareData()
print(random.choice(pairs))

matrix_len = input_lang.n_words

weights_matrix = np.zeros((matrix_len, 300))
words_found = 0

for i, word in enumerate(input_lang.word2index):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))


class EncoderRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(EncoderRNN, self).__init__()
        self.hidden_size = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    res = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(0)
    return res


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, n_iters + 1)):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            torch.save(encoder.state_dict(), 'encoder_init_300.dict')
            torch.save(decoder.state_dict(), 'decoder_init_300.dict')


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    hidden_size = 300
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # encoder.load_state_dict(torch.load('encoder_init.dict'))
    # attn_decoder.load_state_dict(torch.load('decoder_init.dict'))

    trainIters(encoder, attn_decoder, 50000, print_every=5000)
    evaluateRandomly(encoder, attn_decoder)
    decoded_words, decoder_attentions = evaluate(encoder, attn_decoder, 'president senator america')
    print(decoded_words)
    decoded_words, decoder_attentions = evaluate(encoder, attn_decoder, 'india country delhi')
    print(decoded_words)
    decoded_words, decoder_attentions = evaluate(encoder, attn_decoder, 'football season match game')
    print(decoded_words)
    decoded_words, decoder_attentions = evaluate(encoder, attn_decoder, 'actress singer pop star guitarist')
    print(decoded_words)
    test = 'films american sports comedy films'
    decoded_words, decoder_attentions = evaluate(encoder, attn_decoder, test)
    print(test, decoded_words)



    




# # sos_index = word2idx['sos']

# # eos_index = word2idx['eos']
# # sos_swap_word = words[0]
# # eos_swap_word = words[1]

# # words[0], words[sos_index] = words[sos_index], words[0]
# # words[1], words[eos_index] = words[eos_index], words[1]
# # word2idx[sos_swap_word], word2idx['sos'] = word2idx['sos'], word2idx[sos_swap_word]
# # word2idx[eos_swap_word], word2idx['eos'] = word2idx['eos'], word2idx[eos_swap_word]

# #  word2idx = { k : v for k , v in sorted(word2idx.items(), key=operator.itemgetter(1))}

# class InputLang:
#     def __init__(self, name):
#         self.name = name
#         self.word2index = { k : v for k , v in sorted(word2idx.items(), key=operator.itemgetter(1))}
#         self.word2count = { word : 1 for word in words }
#         self.index2word = { i : word for word, i in word2idx.items() }
#         self.n_words = 400001

#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)

#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1


# SOS_token = 0
# EOS_token = 1
# class OutputLang:
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {0: "sos", 1: "eos"}
#         self.n_words = 2  # Count SOS and EOS

#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)

#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1


# def readLangs(): 
#     print("Reading lines...") # Read the file and split into lines 
#     lines = open('/mnt/nvme5/wikipedia/wikipedia_dataset_clean.txt', encoding='utf-8').read().strip().split('\n') # Split every line into pairs and normalize 
#     pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines] 
#     pairs = [list((p)) for p in pairs] 
#     input_lang = InputLang('que') 
#     output_lang = OutputLang('ans') 
#     return input_lang, output_lang, pairs
# MAX_LENGTH = 60 
# def filterPair(p): 
#     return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH 
    
# def filterPairs(pairs): 
#     return [pair for pair in pairs if filterPair(pair)]


# def normalizeString(s):
#     s = s.lower().strip()
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s


# def prepareData(): 
#     input_lang, output_lang, pairs = readLangs() 
#     print("Read %s sentence pairs" % len(pairs)) 
#     pairs = filterPairs(pairs) 
#     print("Trimmed to %s sentence pairs" % len(pairs)) 
#     print("Counting words...") 
#     for pair in pairs: 
#         input_lang.addSentence(pair[0]) 
#         output_lang.addSentence(pair[1]) 
#     print("Counted words:") 
#     print(input_lang.name, input_lang.n_words) 
#     print(output_lang.name, output_lang.n_words) 
#     return input_lang, output_lang, pairs 


# class EncoderRNN(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = embedding_dim

#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
#         self.gru = nn.GRU(hidden_size, hidden_size)

#     def forward(self, input, hidden):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         output, hidden = self.gru(output, hidden)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)

#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))

#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)

#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)

#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)


# def indexesFromSentence(lang, sentence):
#     return [lang.word2index[word] for word in sentence.split(' ')]


# def tensorFromSentence(lang, sentence):
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# def tensorsFromPair(pair):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     return (input_tensor, target_tensor)


# teacher_forcing_ratio = 0.5


# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
#     encoder_hidden = encoder.initHidden()

#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()

#     input_length = input_tensor.size(0)
#     target_length = target_tensor.size(0)

#     encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

#     loss = 0

#     for ei in range(input_length):
#         encoder_output, encoder_hidden = encoder(
#             input_tensor[ei], encoder_hidden)
#         encoder_outputs[ei] = encoder_output[0, 0]

#     decoder_input = torch.tensor([[SOS_token]], device=device)

#     decoder_hidden = encoder_hidden

#     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

#     if use_teacher_forcing:
#         # Teacher forcing: Feed the target as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             loss += criterion(decoder_output, target_tensor[di])
#             decoder_input = target_tensor[di]  # Teacher forcing

#     else:
#         # Without teacher forcing: use its own predictions as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             topv, topi = decoder_output.topk(1)
#             decoder_input = topi.squeeze().detach()  # detach from history as input

#             loss += criterion(decoder_output, target_tensor[di])
#             if decoder_input.item() == EOS_token:
#                 break

#     loss.backward()

#     encoder_optimizer.step()
#     decoder_optimizer.step()

#     return loss.item() / target_length



# import time
# import math


# def asMinutes(s):
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)


# def timeSince(since, percent):
#     now = time.time()
#     s = now - since
#     es = s / (percent)
#     rs = es - s
#     return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
#     start = time.time()
#     print_loss_total = 0  # Reset every print_every
    

#     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
#     training_pairs = [tensorsFromPair(random.choice(pairs))
#                       for i in range(n_iters)]
#     criterion = nn.NLLLoss()

#     for iter in range(1, n_iters + 1):
#         training_pair = training_pairs[iter - 1]
#         input_tensor = training_pair[0]
#         target_tensor = training_pair[1]

#         loss = train(input_tensor, target_tensor, encoder,
#                      decoder, encoder_optimizer, decoder_optimizer, criterion)
#         print_loss_total += loss
        

#         if iter % print_every == 0:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
#                                          iter, iter / n_iters * 100, print_loss_avg))

#         torch.save(encoder.state_dict(), 'encoder_init.dict')
#         torch.save(decoder.state_dict(), 'decoder_init.dict')




# def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
#     with torch.no_grad():
#         input_tensor = tensorFromSentence(input_lang, sentence)
#         input_length = input_tensor.size()[0]
#         encoder_hidden = encoder.initHidden()

#         encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

#         for ei in range(input_length):
#             encoder_output, encoder_hidden = encoder(input_tensor[ei],
#                                                      encoder_hidden)
#             encoder_outputs[ei] += encoder_output[0, 0]

#         decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

#         decoder_hidden = encoder_hidden

#         decoded_words = []
#         decoder_attentions = torch.zeros(max_length, max_length)

#         for di in range(max_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             decoder_attentions[di] = decoder_attention.data
#             topv, topi = decoder_output.data.topk(1)
#             if topi.item() == EOS_token:
#                 decoded_words.append('<EOS>')
#                 break
#             else:
#                 decoded_words.append(output_lang.index2word[topi.item()])

#             decoder_input = topi.squeeze().detach()

#         return decoded_words, decoder_attentions[:di + 1]





# def evaluateRandomly(encoder, decoder, n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('>', pair[0])
#         print('=', pair[1])
#         output_words, attentions = evaluate(encoder, decoder, pair[0])
#         print(output_words)
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')





# if __name__ == '__main__':
#     input_lang, output_lang, pairs = prepareData() 
#     print(random.choice(pairs))

#     matrix_len = input_lang.n_words 
#     weights_matrix = np.zeros((matrix_len, 50)) 
#     words_found = 0 
#     for i, word in enumerate(input_lang.word2index): 
#         try: 
#             weights_matrix[i] = glove[word] 
#             words_found += 1 
#         except KeyError: 
#             weights_matrix[i] = glove['unk'] 


#     hidden_size = 50
#     encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#     attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

#     # encoder.load_state_dict(torch.load('encoder_init.dict'))
#     # attn_decoder.load_state_dict(torch.load('decoder_init.dict'))

#     trainIters(encoder, attn_decoder, 500, print_every=10)

#     evaluateRandomly(encoder, attn_decoder)

    



