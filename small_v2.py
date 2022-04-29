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
from bleu_ignoring import corpus_bleu, SmoothingFunction
from pygments import lex
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CLexer, CppLexer
from matplotlib import pyplot as plt
from ast import literal_eval as make_tuple

LANG = 2
MAXN = 4
N = 5
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

start_time = time.process_time()
all_ngrams = [[], [], [], []]
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
                if (not re.fullmatch('\s+', tok)) and (not re.fullmatch('\/\/.*\n', tok)) and (not re.fullmatch('\/\*.*\*\/', tok, re.DOTALL)):
                    tokenized.append(tok)
            for j in range(1, MAXN+1):
                n_grams = list(ngrams(tokenized, j))
                # for l in n_grams:
                #     # if ''.join(l) == '}else{':
                #     #     print('here')
                #     this_doc.add(l)
                all_ngrams[j-1].extend(n_grams)
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
freq = []
for j in range(MAXN):
    freq.append(Counter(all_ngrams[j]))
# tmp_most_common = freq.most_common(MC)
# most_common_dict = dict(tmp_most_common)
# # print('min_count: ', tmp_most_common[-1])
# print(freq.most_common(100))
# freq.most_common(1000)
print(time.process_time() - start_time, 'seconds')
print(len(all_ngrams), len(freq))
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
# pairs = []
# for k, v in data.items():
#     cc = 0
#     for i in range(len(v)):
#         refs = []
#         for j in range(len(v)):
#             if i != j:
#                 refs.append(v[j])
#         if len(refs) >= 20:
#             pairs.append((v[i], random.sample(refs, 20)))
#             cc += 1
#         # pairs.append((v[i], refs))
#         if cc >= 20:
#             break
# print(len(pairs))
# sample = random.sample(pairs, sample_size)
# print(len(sample))

# candidates = list(
#     map(lambda x: [i for i in list(map(lambda y: y[1], lexer.get_tokens(x[0]))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i, re.DOTALL))], sample))
# references = list(
#     map(lambda x: [[i for i in list(map(lambda y: y[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i, re.DOTALL))] for j in x[1]], sample))
# print(candidates[0])
# print('===============================')
# print(references[0])
# print(candidates)
# print(references)

# with open('intra_hyp_java.json', 'w') as f:
#     json.dump(candidates, f)
# with open('intra_ref_java.json', 'w') as f:
#     json.dump(references, f)

with open('intra_hyp_java.json') as f:
    candidates = json.load(f)
with open('intra_ref_java.json') as f:
    references = json.load(f)

Y_intra = []
Y_v_intra = []
mc = 5
for i in range(N):
    most_common_dict = dict(freq[0].most_common(mc)) | dict(freq[1].most_common(mc)) | dict(freq[2].most_common(mc)) | dict(freq[3].most_common(mc))
    # most_common_dict = dict(sorted_tfidf[:mc])
    # most_common_dict = set(sorted_tfidf[:mc])
    # most_common_dict = sorted_tfidf[:mc]
    start_time = time.process_time()
    intra_bleu_w_freq = corpus_bleu(
        references, candidates, smoothing_function=sm_func, ignoring=most_common_dict)
    print(time.process_time() - start_time, 'seconds for CrystalBLEU')
    print('Intra-class corpus BLEU with frequency adjustment:', intra_bleu_w_freq)
    Y_intra.append(intra_bleu_w_freq)
    mc *= 4
start_time = time.process_time()
intra_bleu_vanilla = corpus_bleu(
    references, candidates, smoothing_function=sm_func)
print(time.process_time() - start_time, 'seconds for BLEU')
print('Intra-class vanilla corpus BLEU:', intra_bleu_vanilla)
for i in range(N):
    Y_v_intra.append(intra_bleu_vanilla)


# Inter-class
# pairs = []
# for k1, v1 in data.items():
#     cc = 0
#     prob_name = k1.split('_')
#     if int(prob_name[0]) > 1 or len(prob_name[1]) > 4:
#         continue
#     for k2, v2 in data.items():
#         prob_name = k2.split('_')
#         if int(prob_name[0]) > 1 or len(prob_name[1]) > 4:
#             continue
#         if k1 == k2 or random.random() < 0.6:
#             continue
#         for i in v1:
#             refs = []
#             for j in v2:
#                 refs.append(j)
#             if len(refs) >= 20:
#                 pairs.append((i, random.sample(refs, 20)))
#                 cc += 1
#             # pairs.append((i, refs))
#         if cc >= 20:
#             break
# print(len(pairs))
# sample = random.sample(pairs, sample_size)
# print(len(sample))

# candidates = list(
#     map(lambda x: [i for i in list(map(lambda y: y[1], lexer.get_tokens(x[0]))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i, re.DOTALL))], sample))
# references = list(
#     map(lambda x: [[i for i in list(map(lambda y: y[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i, re.DOTALL))] for j in x[1]], sample))

# print(candidates)
# print(references)
# with open('inter_hyp_java.json', 'w') as f:
#     json.dump(candidates, f)
# with open('inter_ref_java.json', 'w') as f:
#     json.dump(references, f)

with open('inter_hyp_java.json') as f:
    candidates = json.load(f)
with open('inter_ref_java.json') as f:
    references = json.load(f)

Y_inter = []
Y_v_inter = []
X = []
mc = 5
for i in range(N):
    X.append(mc)
    most_common_dict = dict(freq[0].most_common(mc)) | dict(freq[1].most_common(mc)) | dict(freq[2].most_common(mc)) | dict(freq[3].most_common(mc))
    # most_common_dict = dict(sorted_tfidf[:mc])
    # most_common_dict = set(sorted_tfidf[:mc])
    # most_common_dict = sorted_tfidf[:mc]
    start_time = time.process_time()
    inter_bleu_w_freq = corpus_bleu(
        references, candidates, smoothing_function=sm_func, ignoring=most_common_dict)
    print(time.process_time() - start_time, 'seconds for CrystalBLEU')
    print('Inter-class corpus BLEU with frequency adjustment:', inter_bleu_w_freq)
    Y_inter.append(inter_bleu_w_freq)
    mc *= 4
start_time = time.process_time()
inter_bleu_vanilla = corpus_bleu(
    references, candidates, smoothing_function=sm_func)
print(time.process_time() - start_time, 'seconds for BLEU')
print('Inter-class vanilla corpus BLEU:', inter_bleu_vanilla)
for i in range(N):
    Y_v_inter.append(inter_bleu_vanilla)

# print((Y_v_intra[0] - Y_v_inter[0])/Y_v_intra[0], (np.array(Y_intra) - np.array(Y_inter))/np.array(Y_intra))
# print(Y_v_intra[0] - Y_v_inter[0], np.array(Y_intra) - np.array(Y_inter))
print(Y_v_intra[0] / Y_v_inter[0], np.array(Y_intra) / np.array(Y_inter))
# print('Diff for intra-inter with frequency adjustment:',
#       intra_bleu_w_freq - inter_bleu_w_freq)
# print('Diff for intra-inter vanilla:', intra_bleu_vanilla - inter_bleu_vanilla)
# plt.xscale('log')
# plt.plot(X, np.array(Y_intra) - np.array(Y_v_intra), label='Intra-class')
# plt.plot(X, np.array(Y_inter) - np.array(Y_v_inter), label='Inter-class')
# plt.plot(X, np.array(Y_inter) - np.array(Y_intra), label='new BLEU')
# plt.plot(X, np.array(Y_v_inter) - np.array(Y_v_intra), label='vanilla BLEU')

# plt.plot(X, np.array(Y_intra), label='Intra-class')
# plt.plot(X, np.array(Y_inter), label='Inter-class')
# plt.plot(X, np.array(Y_v_intra), label='Vanilla intra-class')
# plt.plot(X, np.array(Y_v_inter), label='Vanilla inter-class')

plt.plot(X, np.array(Y_intra) - np.array(Y_inter), label='CrystalBLEU')
plt.plot(X, np.array(Y_v_intra) - np.array(Y_v_inter), label='BLEU')
# plt.plot(X, np.array(Y_intra) / np.array(Y_inter), label='CrystalBLEU')
# plt.plot(X, np.array(Y_v_intra) / np.array(Y_v_inter), label='BLEU')
plt.xscale('log')
plt.xlabel('K')
plt.ylabel('Distinguishability')
plt.grid()
plt.legend()
plt.show()
