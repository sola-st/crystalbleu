import json
import random
import re
import math
from collections import Counter
# from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from nltk.util import ngrams
from bleu_freq import corpus_bleu, SmoothingFunction
from pygments import lex
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CLexer, CppLexer
from matplotlib import pyplot as plt

LANG = 2
MAXN = 4
MC = 500
sample_size = 3000
if LANG == 0:
    lexer = CLexer()
elif LANG == 1:
    lexer = CppLexer()
elif LANG == 2:
    lexer = JavaLexer()

total = 0
with open('lang' + str(LANG) + '.json') as f:
    data = json.load(f)

# sm_func = SmoothingFunction(epsilon=0.0001).method1

score = {}
inc = set()
exc = set()

# Intra-class
pairs = []
for k, v in data.items():
    for i in range(len(v)):
        for j in range(len(v)):
            if i != j:
                pairs.append((v[i], v[j]))
print(len(pairs))
sample_pairs = random.sample(pairs, sample_size)
print(len(sample_pairs))

candidates = list(
    map(lambda x: [i for i in list(map(lambda y: y[1], lexer.get_tokens(x[0]))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i))], sample_pairs))
references = list(
    map(lambda x: [[i for i in list(map(lambda y: y[1], lexer.get_tokens(x[1]))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i))]], sample_pairs))

for i in range(len(candidates)):
    for j in range(1, MAXN+1):
        ref_ngrams = Counter(list(ngrams(references[i][0], j)))
        can_ngrams = Counter(list(ngrams(candidates[i], j)))
        for n_gram, c in can_ngrams.items():
            if (n_gram in ref_ngrams) and (min(ref_ngrams[n_gram], c)/c > 0.4):
                inc.add(n_gram)
                if n_gram in score:
                    score[n_gram] += 1
                else:
                    score[n_gram] = 1

# Inter-class
pairs = []
print(len(pairs))
for k1, v1 in data.items():
    prob_name = k1.split('_')
    if int(prob_name[0]) > 1 or len(prob_name[1]) > 4:
        continue
    for k2, v2 in data.items():
        if k1 == k2 or random.random() < 0.6:
            continue
        for i in v1:
            for j in v2:
                if random.random() < 0.2:
                    pairs.append((i, j))
print(len(pairs))
sample_pairs = random.sample(pairs, sample_size)
print(len(sample_pairs))

candidates = list(
    map(lambda x: [i for i in list(map(lambda y: y[1], lexer.get_tokens(x[0]))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i))], sample_pairs))
references = list(
    map(lambda x: [[i for i in list(map(lambda y: y[1], lexer.get_tokens(x[1]))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i))]], sample_pairs))

for i in range(len(candidates)):
    for j in range(1, MAXN+1):
        ref_ngrams = Counter(list(ngrams(references[i][0], j)))
        can_ngrams = Counter(list(ngrams(candidates[i], j)))
        for n_gram, c in can_ngrams.items():
            if (n_gram in ref_ngrams) and (min(ref_ngrams[n_gram], c)/c > 0.2):
                exc.add(n_gram)
                if n_gram in score:
                    score[n_gram] -= 1
                else:
                    score[n_gram] = -1

res = exc.difference(inc)
with open('cheat_set_precision.json', 'w') as f:
    json.dump(list(res), f, indent=2)

with open('cheat.json', 'w') as f:
    json.dump({str(k): v for k, v in score.items()}, f, indent=4)
