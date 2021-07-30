import json
import random
import re
import math
from collections import Counter
# from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from nltk.util import ngrams
# from bleu_freq import corpus_bleu, SmoothingFunction
from bleu_ignoring import corpus_bleu, SmoothingFunction
from pygments import lex
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CLexer, CppLexer
from matplotlib import pyplot as plt
from ast import literal_eval as make_tuple

LANG = 2
MAXN = 4
MC = 500
sample_size = 200
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

# with open('cheat.json') as f:
#     cheat = json.load(f)

# with open('cheat_set_precision.json') as f:
#     cheat_set = json.load(f)

# # scores = {make_tuple(k): v for k, v in cheat.items() if v < 10}
# # sorted_tfidf = sorted(scores.items(), key=lambda item: item[1])
# sorted_tfidf = [tuple(k) for k in cheat_set]
# print(len(sorted_tfidf))

all_ngrams = []
# doc_count = {}

# count = 0
for k, v in data.items():
    prob_name = k.split('_')
    if int(prob_name[0]) > 1 or len(prob_name[1]) > 4:
        continue
    for i in v:
        if random.random() < 0.3:
            # this_doc = set()
            # count += 1
            temp = list(map(lambda x: x[1], lexer.get_tokens(i)))
            tokenized = []
            for tok in temp:
                if (not re.fullmatch('\s+', tok)) and (not re.fullmatch('\/\/.*\n', tok)) and (not re.fullmatch('\/\*.*\*\/', tok)):
                    tokenized.append(tok)
            for j in range(1, MAXN+1):
                n_grams = list(ngrams(tokenized, j))
                # for l in n_grams:
                #     # if ''.join(l) == '}else{':
                #     #     print('here')
                #     this_doc.add(l)
                all_ngrams.extend(n_grams)
                # print(n_grams)
            # print(this_doc)
            # for j in this_doc:
            #     if j in doc_count:
            #         doc_count[j] += 1
            #     else:
            #         doc_count[j] = 1

# # print(count, len(all_ngrams))
# L = len(all_ngrams)
# print(L)
freq = Counter(all_ngrams)
# tmp_most_common = freq.most_common(MC)
# most_common_dict = dict(tmp_most_common)
# # print('min_count: ', tmp_most_common[-1])
print(freq.most_common(100))
# # exit()

# tfidf = {}
# for k, v in doc_count.items():
#     if ''.join(k) == '}else{':
#         print('here')
#     # tfidf[k] = math.log(1 + freq[k]) * math.log(count / v)
#     tfidf[k] = freq[k]/math.log(L) * math.log(count / v)
#     # tfidf[k] = math.log(count / v)

# # print(doc_count)
# # sorted_doc_counts = sorted(doc_count.items(), key=lambda item: item[1])
# sorted_tfidf = sorted(tfidf.items(), key=lambda item: item[1])
# print(len(sorted_tfidf))
# print(sorted_tfidf[:50])
# print(sorted_tfidf[-50:])
# with open('TFIDF.txt', 'w') as f:
#     for a, b in sorted_tfidf:
#         f.write(''.join(a) + '\t\t' + str(b) + '\n')
# Y = list(map(lambda y: y[1], sorted_tfidf))
# plt.plot(Y)
# plt.show()


# Intra-class
pairs = []
for k, v in data.items():
    for i in range(len(v)):
        refs = []
        for j in range(len(v)):
            if i != j:
                refs.append(v[j])
        pairs.append((v[i], refs))
print(len(pairs))
sample = random.sample(pairs, sample_size)
print(len(sample))

candidates = list(
    map(lambda x: [i for i in list(map(lambda y: y[1], lexer.get_tokens(x[0]))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i))], sample))
references = list(
    map(lambda x: [[i for i in list(map(lambda y: y[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i))] for j in x[1]], sample))
# print(candidates[0])
# print('===============================')
# print(references[0])
# print(candidates)
# print(references)

Y_intra = []
Y_v_intra = []
mc = 1
for i in range(6):
    most_common_dict = dict(freq.most_common(mc))
    # most_common_dict = dict(sorted_tfidf[:mc])
    # most_common_dict = set(sorted_tfidf[:mc])
    # most_common_dict = sorted_tfidf[:mc]
    intra_bleu_w_freq = corpus_bleu(
        references, candidates, smoothing_function=sm_func, ignoring=most_common_dict)
    intra_bleu_vanilla = corpus_bleu(
        references, candidates, smoothing_function=sm_func)
    print('Intra-class corpus BLEU with frequency adjustment:', intra_bleu_w_freq)
    print('Intra-class vanilla corpus BLEU:', intra_bleu_vanilla)
    Y_intra.append(intra_bleu_w_freq)
    Y_v_intra.append(intra_bleu_vanilla)
    mc *= 10

# Inter-class
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
        for i in v1:
            refs = []
            for j in v2:
                refs.append(j)
            pairs.append((i, refs))
print(len(pairs))
sample = random.sample(pairs, sample_size)
print(len(sample))

candidates = list(
    map(lambda x: [i for i in list(map(lambda y: y[1], lexer.get_tokens(x[0]))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i))], sample))
references = list(
    map(lambda x: [[i for i in list(map(lambda y: y[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i))] for j in x[1]], sample))

# print(candidates)
# print(references)

Y_inter = []
Y_v_inter = []
X = []
mc = 1
for i in range(6):
    X.append(mc)
    most_common_dict = dict(freq.most_common(mc))
    # most_common_dict = dict(sorted_tfidf[:mc])
    # most_common_dict = set(sorted_tfidf[:mc])
    # most_common_dict = sorted_tfidf[:mc]
    inter_bleu_w_freq = corpus_bleu(
        references, candidates, smoothing_function=sm_func, ignoring=most_common_dict)
    inter_bleu_vanilla = corpus_bleu(
        references, candidates, smoothing_function=sm_func)
    print('Inter-class corpus BLEU with frequency adjustment:', inter_bleu_w_freq)
    print('Inter-class vanilla corpus BLEU:', inter_bleu_vanilla)
    Y_inter.append(inter_bleu_w_freq)
    Y_v_inter.append(inter_bleu_vanilla)
    mc *= 10


print(np.array(Y_intra) / np.array(Y_inter))
print(np.array(Y_intra) - np.array(Y_inter))
# print('Diff for intra-inter with frequency adjustment:',
#       intra_bleu_w_freq - inter_bleu_w_freq)
# print('Diff for intra-inter vanilla:', intra_bleu_vanilla - inter_bleu_vanilla)
plt.xscale('log')
# plt.plot(X, np.array(Y_intra) - np.array(Y_v_intra), label='Intra-class')
# plt.plot(X, np.array(Y_inter) - np.array(Y_v_inter), label='Inter-class')
# plt.plot(X, np.array(Y_inter) - np.array(Y_intra), label='new BLEU')
# plt.plot(X, np.array(Y_v_inter) - np.array(Y_v_intra), label='vanilla BLEU')
plt.plot(X, np.array(Y_intra), label='Intra-class')
plt.plot(X, np.array(Y_inter), label='Inter-class')
plt.plot(X, np.array(Y_v_intra), label='Vanilla intra-class')
plt.plot(X, np.array(Y_v_inter), label='Vanilla inter-class')
plt.xlabel('# of common n-grams')
plt.ylabel('BLEU')
plt.grid()
plt.legend()
plt.show()
