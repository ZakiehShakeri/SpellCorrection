import gc
from random import randint, random, choice, choices
import numpy as np
import math
import os
import sys
from tempfile import TemporaryDirectory
from typing import Tuple
import io
import copy
import time
import csv
import argparse

import torch
from torch.nn.modules.sparse import EmbeddingBag
from fasttext import load_model
from torch.autograd import Variable
from torch import dropout, nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchdata.dataloader2 as dl
import torchdata.datapipes as dp
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


epsilon = 1e-5


def load_moe(moe_path):
    moe = {}
    with open(moe_path, 'rt', encoding='utf-8') as f:
        lines = csv.reader(f, delimiter='\t')
        for m, e in lines:
            if e not in moe:
                moe[e] = []
            moe[e].append(m)
    return moe


def misspelled_sentence(sentence):
    return [choice(misspelled_sentence.moe[w]) if w in misspelled_sentence.moe and random() < 0.2 else w for w in sentence]

# ############## class definitions ##################################


class FastTextEmbeddingBag(EmbeddingBag):
    def __init__(self, model_path):
        self.model = load_model(model_path)
        input_matrix = self.model.get_input_matrix()
        input_matrix_shape = input_matrix.shape
        super().__init__(input_matrix_shape[0], input_matrix_shape[1])
        self.weight.data.copy_(torch.FloatTensor(input_matrix))

    def forward(self, sentences):
        results = []
        max_len = max(len(s) for s in sentences)
        for words in sentences:
            word_subinds = np.empty([0], dtype=np.int64)
            word_offsets = [0]
            for word in words:
                _, subinds = self.model.get_subwords(word)
                word_subinds = np.concatenate((word_subinds, subinds))
                word_offsets.append(word_offsets[-1] + len(subinds))
            word_offsets = word_offsets[:-1]
            ind = Variable(torch.LongTensor(word_subinds).to(device))
            offsets = Variable(torch.LongTensor(word_offsets).to(device))
            results.append(torch.cat((super().forward(ind, offsets), torch.zeros(
                max_len - len(words), self.embedding_dim).to(device)), dim=0).unsqueeze(0))
        return torch.cat(tuple(results), dim=0)

# sanity check!


def scheck(n: str, v: Tensor, b: Tensor = None) -> None:
    # return
    if v.isnan().any():
        print(f"{n}:", v)
        if b is not None:
            print(f"before {n}:", b)
        sys.exit(1)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        dropout: float = 0.1,
        causual: bool = False
    ):
        nn.Module.__init__(self)

        assert embedding_size % num_heads == 0

        self.head_size = embedding_size // num_heads
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.causual = causual

        self.k_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.v_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.q_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.o_linear = nn.Linear(embedding_size, embedding_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        return f'mha(embedding_size={self.embedding_size},num_heads={self.num_heads}, causual={self.causual})'

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):

        v_len = v.size(-2)
        q_len = q.size(-2)

        scheck("k", k)
        scheck("q", q)
        scheck("v", v)

        k = self.k_linear(k)
        v = self.v_linear(v)
        q = self.q_linear(q)

        scheck("k1", k)
        scheck("q1", q)
        scheck("v1", v)

        k = k.view(-1, v_len, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(-1, v_len, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(-1, q_len, self.num_heads, self.head_size).transpose(1, 2)

        q = q / math.sqrt(self.head_size)

        scores = torch.matmul(q, k.transpose(2, 3))
        scheck("scores", scores, k)

        if self.causual:
            causual_mask = torch.tril(torch.ones(q_len, v_len))
            causual_mask = causual_mask.view(1, q_len, v_len).to(scores.device)
            causual_mask = causual_mask != 0
            if mask is None:
                mask = causual_mask
            else:
                mask = mask & causual_mask

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -math.inf)

        attention = self.softmax(scores)
        scheck("attention", attention, scores)
        attention = self.dropout(attention)

        result = torch.matmul(attention, v) \
            .transpose(1, 2).contiguous() \
            .view(-1, q_len, self.embedding_size)
        scheck("result", result, v)
        result = self.o_linear(result)
        scheck("result1", result)

        return result


def create_positional_encoding_vector(embedding_size: int, max_length: int):
    pe = torch.zeros(max_length, embedding_size)
    p = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
    s = torch.exp(
        -torch.arange(0, embedding_size, 2, dtype=torch.float) *
        (math.log(1e4) / embedding_size)
    )
    pe[:, 0::2] = torch.sin(p * s)
    pe[:, 1::2] = torch.cos(p * s)
    return pe


def create_positionwise_feedforward(
    embedding_size: int, ffn_size: int, dropout: float
):
    return nn.Sequential(
        nn.Linear(embedding_size, ffn_size), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(ffn_size, embedding_size)
    )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, num_heads: int, embedding_size: int, ffn_size: int, dropout: float
    ):
        nn.Module.__init__(self)
        self.mha = MultiheadAttention(
            num_heads, embedding_size, dropout=dropout, causual=False
        )
        self.ln0 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.ffn = create_positionwise_feedforward(
            embedding_size, ffn_size, dropout
        )
        self.ln1 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        h = self.ln0(x)
        scheck("ln0", h, x)
        h = self.mha(h, h, h, x_mask.unsqueeze(1))
        scheck("mha", h)
        x = self.dropout(h) + x
        h = self.ln1(x)
        scheck("ln1", h, x)
        h = self.ffn(h)
        scheck("ffn", h)
        h = self.dropout(h) + x
        return h


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, fasttext_model_path: str, nhead: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder = FastTextEmbeddingBag(fasttext_model_path)
        self.d_model = d_model = self.encoder.embedding_dim
        d_hid = 2 * d_model
        self.encoder.weight.requires_grad_(False)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = nn.ModuleList(TransformerEncoderLayer(
            nhead, d_model, d_hid, dropout) for _ in range(nlayers))
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        for p in self.transformer_encoder.parameters():
            if p.data.dim() < 2:
                p.data.uniform_(-0.001, 0.001)
            else:
                nn.init.xavier_normal_(p.data)
        self.decoder.bias.data.zero_()
        nn.init.xavier_normal_(self.decoder.weight.data)

    def forward(self, src, src_mask: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        # print(src.shape)
        src = self.pos_encoder(src)
        output = src
        for layer in self.transformer_encoder:
            output = layer(output, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer(
            'pe', create_positional_encoding_vector(d_model, max_len).unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ############################# main ###################################


parser = argparse.ArgumentParser(description='')
parser.add_argument('--moe-path', type=str, default='',
                    help='path to MOE model')
parser.add_argument('--data', type=str, default='',
                    help='path to original file for training, validating and testing the LM')
parser.add_argument('--fasttext-path', type=str,
                    default='', help='path to fasttext model')

args = parser.parse_args()
moe_model_path = args.moe_path
source = args.data
fasttext_model_path = args.fasttext_path

print("Loading data ...")

datapipe = dp.iter.FileOpener(
    [source], mode='rt', encoding='utf-8')
datapipe = datapipe.readlines(strip_newline=False, return_path=False).shuffle(
).set_shuffle(False).sharding_filter()
print("Loading data, DONE.")

tokenizer = get_tokenizer('basic_english')

print("Building vocabulary if not exist ...")
if os.path.exists("transformer-LM/large_lm/models/vocab.pt"):
    vocab = torch.load(
        "transformer-LM/large_lm/models/vocab.pt", map_location="cpu")
else:
    vocab = build_vocab_from_iterator(
        map(tokenizer, datapipe), min_freq=7, max_tokens=200000, specials=['<pad>', '<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    torch.save(vocab, "transformer-LM/large_lm/models/vocab.pt")
print("Building vocabulary, DONE.")

total_length = len(list(datapipe))

train_iter, valid_iter, test_iter = datapipe.random_split(
    total_length=total_length, weights={"train": 0.8, "valid": 0.1, "test": 0.1}, seed=0)


# TODO multiprocessing and distributing features should be added!
# mp_rs = dl.MultiProcessingReadingService(num_workers=4)
# dist_rs = dl.DistributedReadingService()
# rs = dl.SequentialReadingService(dist_rs, mp_rs)
# train_dl = dl.DataLoader2(train_iter, reading_service=rs)
# valid_dl = dl.DataLoader2(valid_iter, reading_service=rs)
# test_dl = dl.DataLoader2(test_iter, reading_service=rs)

train_dl = DataLoader(train_iter, batch_size=20, num_workers=4)
valid_dl = DataLoader(valid_iter, batch_size=10, num_workers=4)
test_dl = DataLoader(test_iter, batch_size=10, num_workers=4)

print("Loading MOE ...")
misspelled_sentence.moe = load_moe(moe_model_path)
print("Loading MOE, DONE.")


def prepare_target_data(data):
    max_len = max(len(s) for s in data)
    data = [torch.LongTensor(vocab(s)) for s in data]
    result = torch.cat([torch.cat((item, torch.zeros(
        max_len - len(item), dtype=torch.long)), dim=0).unsqueeze(0) for item in data], dim=0).to(device)
    return result


def prepare_data(raw_text_iter) -> Tensor:
    result = [tokenizer(item) for item in raw_text_iter if item.strip()]
    result = [e for e in result if e]
    return result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 20
num_batch = math.ceil(total_length / batch_size)
eval_batch_size = 10

#  The model hyperparameters are defined below. The vocab size is
# equal to the length of the vocab object.

ntokens = len(vocab)
fasttext_model_path = args.fasttext_path
nlayers = 2
nhead = 2
dropout = 0.2
model = TransformerModel(ntokens, fasttext_model_path,
                         nhead, nlayers, dropout).to(device)

#################################################################
# Run the model

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
lr = 1
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: nn.Module, epoch, best_val_loss, best_model_params_path) -> None:
    model.train()
    total_loss = 0.
    log_interval = 50
    total_batch = 0
    batch_count = 0
    start_time = time.time()

    for batch in iter(train_dl):
        batch_count += 1
        data = prepare_data(batch)
        misspelled_data = [misspelled_sentence(s) for s in data]
        target = prepare_target_data(data)
        output = model(misspelled_data, target != 0)
        loss = criterion(output.view(-1, ntokens), target.view(-1))
        if loss.isnan().any():
            print("loss is nan")
            print("data")
            print("misspelled data")
            print("target")
            print("output")
            sys.exit(1)
        # print("loss.item()", loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        total_batch += 1

        # Reporting after log_interval=50 iterations
        if batch_count % log_interval == 0 and batch_count > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / total_batch
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch_count:5d}/{num_batch:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            total_batch = 0
            start_time = time.time()

        # saving after 10,000 iteration
        if batch_count % (10000) == 0 and batch_count > 0:
            val_loss = evaluate(model, valid_dl)
            print("evaluation completed!!!!!!!!!!!!!!!!!")
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of iter {batch_count:3d} | time: {elapsed:5.2f}s | '
                  f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)
        model.train()
        del data
        del misspelled_data
        del target
        del output
        del loss
        gc.collect()
        # print(torch.cuda.memory_summary(device=device))


def evaluate(model: nn.Module, eval_dl) -> float:
    model.eval()
    total_loss = 0.
    total_batches = 0
    with open("data.txt", 'wt', encoding='utf-8') as d, \
            open("mis_data.txt", 'wt', encoding='utf-8') as m, \
            open("predicted.txt", 'wt', encoding='utf-8') as p:
        with torch.no_grad():
            for batch in iter(eval_dl):
                data = prepare_data(batch)
                total_batches += 1
                misspelled_data = [misspelled_sentence(s) for s in data]

                differences = [[index for index, (w0, w1) in enumerate(
                    zip(s0, s1)) if w0 != w1] for s0, s1 in zip(data, misspelled_data)]
                target = prepare_target_data(data)
                output = model(misspelled_data, target != 0)
                # predicted_indices = torch.argmax(output, dim=-1)
                (_, predicted_indices) = torch.topk(output, 10, dim=-1)
                total_words = []

                # writing some data to files for quality check: original word, misspelled_word and the modification proposed by model
                for sentence in predicted_indices:
                    total_words.append([vocab.lookup_tokens(sentence[i].tolist())
                                        for i in range(len(sentence))])

                for i in range(len(data)):
                    for j in differences[i]:
                        d.write(str(data[i][j]) + '\n')
                        m.write(str(misspelled_data[i][j]) + '\n')
                        p.write(str(total_words[i][j]) + '\n')
                loss = criterion(output.view(-1, ntokens), target.view(-1))
                if loss.isnan().any():
                    print("loss is nan")
                    print("data")
                    print("misspelled data")
                    print("target")
                    print("output")
                    sys.exit(1)
                total_loss += loss.item()
        return total_loss / total_batches

######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.


best_val_loss = float('inf')
epochs = 20
best_model_params_path = "transformer-LM/large_lm/models/best_model.pt"

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model, epoch, best_val_loss, best_model_params_path)
    val_loss = evaluate(model, valid_dl)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_params_path)

    scheduler.step()


######################################################################
# Evaluate the best model on the test dataset
# -------------------------------------------

model.load_state_dict(torch.load(best_model_params_path))
test_loss = evaluate(model, test_dl)
test_ppl = math.exp(test_loss)
print('-' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('-' * 89)
