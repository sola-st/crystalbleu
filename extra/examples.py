import sys
import time
import json
import random
import re
import math
from collections import Counter
import numpy as np
from nltk.util import ngrams
from CodeBLEU.code_bleu import code_bleu
from bleu_ignoring import corpus_bleu, sentence_bleu, SmoothingFunction
from pygments import lex
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.token import Comment
from matplotlib import pyplot as plt
from ast import literal_eval as make_tuple

LANG = 2
MAXN = 4
N = 1
MC = 150
sample_size = 1000
if LANG == 0:
    lexer = CLexer()
elif LANG == 1:
    lexer = CppLexer()
elif LANG == 2:
    lexer = JavaLexer()

sm_func = SmoothingFunction(epsilon=0.0001).method1

total = 0
with open('lang' + str(LANG) + '.json') as f:
    data = json.load(f)

start_time = time.process_time()
all_ngrams = []

total_tokens = 0
total_prog = 0
for k, v in data.items():
    prob_name = k.split('_')
    if int(prob_name[0]) > 1 or len(prob_name[1]) > 4:
        continue
    total_prog += len(v)
    for i in v:
        if random.random() < 0.3:
            tokenized = [j[1] for j in lexer.get_tokens(i) if not (re.fullmatch('\s+', j[1]) or (j[0] in Comment))]
            total_tokens += len(tokenized)
            for j in range(1, MAXN+1):
                n_grams = list(ngrams(tokenized, j))
                all_ngrams.extend(n_grams)

freq = Counter(all_ngrams)
print(time.process_time() - start_time, 'seconds')
mc = 150
most_common_dict = dict(freq.most_common(mc))

pairs = []
for k1, v1 in data.items():
    prob_name = k1.split('_')
    if int(prob_name[0]) > 1 or len(prob_name[1]) > 4:
        continue
    for k2, v2 in data.items():
        prob_name = k2.split('_')
        if int(prob_name[0]) > 1 or len(prob_name[1]) > 4:
            continue
        if k1 == k2 or random.random() < 0.6:
            continue
        for ii in v1:
            refs = []
            c = random.sample(v2, 2)
            ref = [j[1] for j in lexer.get_tokens(c[0]) if not (re.fullmatch('\s+', j[1]) or (j[0] in Comment))]
            hyp1 = [j[1] for j in lexer.get_tokens(c[1]) if not (re.fullmatch('\s+', j[1]) or (j[0] in Comment))]
            hyp2 = [j[1] for j in lexer.get_tokens(ii) if not (re.fullmatch('\s+', j[1]) or (j[0] in Comment))]
            bleu_intra = sentence_bleu([ref], hyp1, smoothing_function=sm_func)
            bleu_inter = sentence_bleu([ref], hyp2, smoothing_function=sm_func)
            bleu_dist = bleu_intra / (bleu_inter + 1e-10)
            if (bleu_dist > 2.2) or (len(ii) + len(c[0]) + len(c[1]) > 1000):
                continue

            crystalbleu_intra = sentence_bleu([ref], hyp1, smoothing_function=sm_func, ignoring=most_common_dict)
            crystalbleu_inter = sentence_bleu([ref], hyp2, smoothing_function=sm_func, ignoring=most_common_dict)
            crystalbleu_dist = crystalbleu_intra / (crystalbleu_inter + 1e-10)
            if crystalbleu_intra < 0.2:
                continue
            if crystalbleu_dist > bleu_dist + 3:
                print('ref:')
                print(c[0])
                print('hyp1:')
                print(c[1])
                print('hyp2:')
                print(ii)
                print(f'BLEU hyp1: {bleu_intra}')
                print(f'BLEU hyp2: {bleu_inter}')
                print(f'CrystalBLEU hyp1: {crystalbleu_intra}')
                print(f'CrystalBLEU hyp2: {crystalbleu_inter}')
                print('--------------------------')
