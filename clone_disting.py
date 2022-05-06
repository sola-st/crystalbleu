from collections import Counter
import json
import random
import re
from pygments.lexers.jvm import JavaLexer
from pygments.token import Comment
from bleu_ignoring import corpus_bleu, SmoothingFunction
from nltk.util import ngrams
import numpy as np

sm_func = SmoothingFunction(epsilon=0.0001).method1

lexer = JavaLexer()
code = {}
with open('clone_detection/data.jsonl') as f:
    tmp = f.read().split('\n')
# tmp = tmp[:3000]
all_ngrams = []

for j in tmp:
    try:
        this = json.loads(j)
        tok = [i[1] for i in lexer.get_tokens(this['func']) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
        code[this['idx']] = tok
        if random.random() < 0.3:
            for i in range(1, 5):
                all_ngrams.extend(ngrams(tok, i))
    except:
        break

all_ngrams = Counter(all_ngrams)
# most_common_dict = dict(all_ngrams.most_common(150))

# print(len(code.items()))

with open('clone_detection/test.txt') as f:
    tmp = f.read().split('\n')

true_label = []
bleu_label = []
crystal_label = []

intra_h = []
intra_r = []
inter_h = []
inter_r = []

bs = [[], []]
cs = [[], []]

smp = random.sample(tmp, 30000)
for j in smp:
    x = re.split('\s+', j)
    if len(x) == 3:
        c1, c2, label = x
        code1 = code[c1]
        code2 = code[c2]
        if int(label) == 0:
            inter_h.append(code2)
            inter_r.append([code1])
        else:
            intra_h.append(code2)
            intra_r.append([code1])


mc = 1
crystal_inter = []
crystal_intra = []

bleu_inter = [corpus_bleu(inter_r, inter_h, smoothing_function=sm_func)]
bleu_intra = [corpus_bleu(intra_r, intra_h, smoothing_function=sm_func)]
for i in range(11):
    most_common_dict = dict(all_ngrams.most_common(mc))

    bleu_inter.append(bleu_inter[0])
    bleu_intra.append(bleu_intra[0])
    crystal_inter.append(corpus_bleu(inter_r, inter_h, smoothing_function=sm_func, ignoring=most_common_dict))
    crystal_intra.append(corpus_bleu(intra_r, intra_h, smoothing_function=sm_func, ignoring=most_common_dict))

    mc *= 3

with open('clone_distinguishability.npy', 'wb') as f:
    np.save(f, np.array(crystal_intra))
    np.save(f, np.array(crystal_inter))
    np.save(f, np.array(bleu_intra[:-1]))
    np.save(f, np.array(bleu_inter[:-1]))