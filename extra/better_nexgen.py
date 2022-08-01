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

LANG = 2
MAXN = 4
mc = 150
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
    data = f.read().split('\n')

ref = []

start_time = time.process_time()
all_ngrams = []

total_tokens = 0
for j in data:
    tokenized = [i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
    total_tokens += len(tokenized)
    ref.append([tokenized])
    for j in range(1, MAXN+1):
        n_grams = list(ngrams(tokenized, j))
        all_ngrams.extend(n_grams)

freq = Counter(all_ngrams)
print(time.process_time() - start_time, 'seconds')
# print(len(all_ngrams), len(freq))
print('{} tokens'.format(total_tokens))

with open('nexgen/baseline-100k.out') as f:
    tmp = f.read().split('\n')

hyp = []

for j in tmp:
    prediction = [i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
    worse_prediction = prediction[:-1] + \
        ['for', '(' ,'int', 'uselessVar', '=', '0', ';', '1', '<', '0', ';', 'uselessVar', '++' , ')', 'System', '.', 'out', '.', 'println', '(', ')', ';'] + \
        prediction[-1:]
    hyp.append(worse_prediction)

with open('nexgen/tgt-test.txt') as f:
# with open('codexglue/cs-java-model2.output') as f:
    tmp = f.read().split('\n')

hyp2 = []
target = []
comm_ngrams = dict(freq.most_common(mc))
c = 0
fltr = []
for j in tmp:
    c += 1
    res = ''
    count = 0
    tokens = [i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
    used_ngrams = set()
    for i in range(MAXN, 1, -1):
        if count >= len(tokens):
            break
        nnn = list(ngrams(tokens, i))
        for k in nnn:
            if count >= len(tokens):
                break
            # if (k in comm_ngrams) and ((k not in used_ngrams) or (random.random() < 0.25)):
            if (k in comm_ngrams) and (k not in used_ngrams):
                new_part = ' '.join(k)
                for ii in range(i):
                    used_ngrams.update(ngrams(new_part.split(' '), ii))
                res += ' ' + new_part
                count += len(k)
    remaining = max(0, len(tokens) - count)
    # if remaining / len(tokens) <= 0.05:
    #     fltr.append(c-1)
    # res += ' ' + ' '.join(tokens[(len(tokens)-remaining)//2:-(len(tokens)-remaining)//2])
    # res += ' ' + ' '.join(tokens[(len(tokens)-remaining)//4:-3*(len(tokens)-remaining)//4])
    # res += ' ' + ' '.join(tokens[-1:-(len(tokens)-remaining):-1])
    for k, v in comm_ngrams.items():
        if (remaining > -5):
            res += ' ' + ' '.join(k)
            remaining -= len(k)
    # print(res)
    # for k, v in comm_ngrams.items():
    #     if k in ngrams_here:
    #         res += ' '.join(k)
    #         count += len(k)
    #     if count >= len(tokens):
    #         break
    hyp2.append([i[1] for i in lexer.get_tokens(res) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))])

# print(len(fltr))
# ref = [ref[i] for i in fltr]
# hyp = [hyp[i] for i in fltr]
# hyp2 = [hyp2[i] for i in fltr]
# ref = ref[:10000]
# hyp = hyp[:10000]
# hyp2 = hyp2[:10000]

most_common_dict = dict(freq.most_common(mc))
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

