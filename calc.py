import sys
import time
import json
import random
import re
import math
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

LANG = 2
MAXN = 4
N = 1
MC = 500
sample_size = 1000
if LANG == 0:
    lexer = CLexer()
elif LANG == 1:
    lexer = CppLexer()
elif LANG == 2:
    lexer = JavaLexer()

sm_func = SmoothingFunction(epsilon=0.0001).method1

total = 0
with open('nexgen/tgt-test.txt') as f:
# with open('codexglue/test.java-cs.txt.java') as f:
    data = f.read().split('\n')

ref = []

start_time = time.process_time()
all_ngrams = []

total_tokens = 0
for i in data:
    temp = list(map(lambda x: x[1], lexer.get_tokens(i)))
    tokenized = []
    for tok in temp:
        if (not re.fullmatch('\s+', tok)) and (not re.fullmatch('\/\/.*\n', tok)) and (not re.fullmatch('\/\*.*\*\/', tok, re.DOTALL)):
            tokenized.append(tok)
    total_tokens += len(tokenized)
    ref.append([tokenized])
    for j in range(1, MAXN+1):
        n_grams = list(ngrams(tokenized, j))
        all_ngrams.extend(n_grams)

freq = Counter(all_ngrams)
print(time.process_time() - start_time, 'seconds')
# print(len(all_ngrams), len(freq))
print('{} tokens'.format(total_tokens))

# with open('nexgen/baseline-100k.out') as f:
with open('codexglue/cs-java-model1.output') as f:
    tmp = f.read().split('\n')

hyp = []

for i in tmp:
    hyp.append([t for t in list(map(lambda x: x[1], lexer.get_tokens(i))) if not (re.fullmatch('\s+', t) or re.fullmatch('\/\/.*\n', t) or re.match('\/\*.*\*\/', t, re.DOTALL))])

# with open('nexgen/multi_slicing-100k.out') as f:
with open('codexglue/cs-java-model2.output') as f:
    tmp = f.read().split('\n')

hyp2 = []

for i in tmp:
    hyp2.append([t for t in list(map(lambda x: x[1], lexer.get_tokens(i))) if not (re.fullmatch('\s+', t) or re.fullmatch('\/\/.*\n', t) or re.match('\/\*.*\*\/', t, re.DOTALL))])


mc = 500
for i in range(N):
    most_common_dict = dict(freq.most_common(mc))
    start_time = time.process_time()
    crystalbleu = corpus_bleu(
        ref, hyp, smoothing_function=sm_func, ignoring=most_common_dict)
    print(time.process_time() - start_time, 'seconds for CrystalBLEU')
    print('CrystalBLEU:', crystalbleu)
    mc += 100
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

# plt.plot(X, np.array(Y_intra) - np.array(Y_inter), label='CrystalBLEU')
# plt.plot(X, np.array(Y_v_intra) - np.array(Y_v_inter), label='BLEU')
# # plt.xscale('log')
# plt.xlabel('K')
# plt.ylabel('Distinguishability')
# plt.grid()
# plt.legend()
# plt.show()
