import json
import random
import time
import re
from collections import Counter
# from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from nltk.util import ngrams
# from bleu_freq import corpus_bleu, SmoothingFunction
from CodeBLEU.code_bleu import code_bleu
from bleu_ignoring import corpus_bleu, SmoothingFunction
from pygments import lex
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CLexer, CppLexer
from matplotlib import pyplot as plt
from ast import literal_eval as make_tuple
from pygments.token import Comment
from statistical_test import stat_test

MAXN = 4
mc = 500

sm_func = SmoothingFunction(epsilon=0.0001).method1

total = 0
with open('concode/train.json') as f:
    data = list(map(lambda x: json.loads(x)['code'] ,f.read().split('\n')[:-1]))

ref = []

start_time = time.process_time()
all_ngrams = []

total_tokens = 0
for j in data:
    tokenized = j.split(' ')
    total_tokens += len(tokenized)
    # ref.append([tokenized])
    for j in range(1, MAXN+1):
        n_grams = list(ngrams(tokenized, j))
        all_ngrams.extend(n_grams)

freq = Counter(all_ngrams)
print(time.process_time() - start_time, 'seconds')
# print(len(all_ngrams), len(freq))
print('{} tokens'.format(total_tokens))

with open('concode/predictions.txt') as f:
    tmp = f.read().split('\n')[:-1]

hyp = []

for j in tmp:
    hyp.append(j.split(' '))

# with open('nexgen/tgt-test.txt') as f:
with open('concode/answers.json') as f:
    tmp = list(map(lambda x: json.loads(x)['code'], f.read().split('\n')[:-1]))

for j in tmp:
    ref.append([j.split(' ')])

hyp2 = []
target = []
comm_ngrams = dict(freq.most_common(mc))
most_common_dict = comm_ngrams
c = 0
fltr = []
for j in range(len(ref)):
    res = []

    cn = comm_ngrams.items().__iter__()
    i = 1
    while len(res) < len(ref[j][0]):
        try:
            if random.random() < 0.825:#0.82
                k, v = cn.__next__()
                res = list(k) + res
            else:
                res.append(ref[j][0][i])
                i = (i+1)%len(ref[j][0])
        except:
            cn = comm_ngrams.items().__iter__()

    hyp2.append(res)
    c += 1

print(len(ref))
print('Real predictions:')
em = 0
for i, j in zip(ref, hyp):
    if i[0] == j:
        em += 1
print(f'Exact match: {em}')
start_time = time.process_time()
crystalbleu = corpus_bleu(
    ref, hyp, smoothing_function=sm_func, ignoring=most_common_dict)
print(time.process_time() - start_time, 'seconds for CrystalBLEU')
print('CrystalBLEU:', crystalbleu)

start_time = time.process_time()
bleu_vanilla = corpus_bleu(
    ref, hyp, smoothing_function=sm_func)
print(time.process_time() - start_time, 'seconds for BLEU')
print('BLEU:', bleu_vanilla)

start_time = time.process_time()
codebleu = code_bleu(
    ref, hyp)
print(time.process_time() - start_time, 'seconds for CodeBLEU')
print('CodeBLEU:', codebleu)

print('--------------------------------')
print('Fake predictions:')
em = 0
for i, j in zip(ref, hyp2):
    if i[0] == j:
        em += 1
print(f'Exact match: {em}')
start_time = time.process_time()
crystalbleu = corpus_bleu(
    ref, hyp2, smoothing_function=sm_func, ignoring=most_common_dict)
print(time.process_time() - start_time, 'seconds for CrystalBLEU')
print('CrystalBLEU:', crystalbleu)

start_time = time.process_time()
bleu_vanilla = corpus_bleu(
    ref, hyp2, smoothing_function=sm_func)
print(time.process_time() - start_time, 'seconds for BLEU')
print('BLEU:', bleu_vanilla)

start_time = time.process_time()
codebleu = code_bleu(
    ref, hyp2)
print(time.process_time() - start_time, 'seconds for CodeBLEU')
print('CodeBLEU:', codebleu)

stat_test(ref, hyp, hyp2, most_common_dict)