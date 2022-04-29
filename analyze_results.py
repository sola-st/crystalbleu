import json
from collections import Counter
import re

from nltk.util import ngrams
from bleu_ignoring import corpus_bleu, sentence_bleu, SmoothingFunction
from pygments.lexers.jvm import JavaLexer
from CodeBLEU.code_bleu import code_bleu

from scipy import stats

from matplotlib import pyplot as plt

MAXN = 4
sm_func = SmoothingFunction(epsilon=0.0001).method1
lexer = JavaLexer()
with open('codexglue/test.java-cs.txt.java') as f:
    corpus = f.readlines()
all_ngrams = []
for i in corpus:
    for j in range(1, MAXN+1):
        all_ngrams.extend(list(ngrams([t for t in list(map(lambda x: x[1], lexer.get_tokens(i))) if not (re.fullmatch('\s+', t) or re.fullmatch('\/\/.*\n', t) or re.match('\/\*.*\*\/', t, re.DOTALL))], j)))

freq = Counter(all_ngrams)
comm2 = dict(freq.most_common(500))
# print(comm2)

with open('nexgen/tgt-test.txt') as f:
    corpus = f.readlines()
all_ngrams = []
for i in corpus:
    for j in range(1, MAXN+1):
        all_ngrams.extend(list(ngrams([t for t in list(map(lambda x: x[1], lexer.get_tokens(i))) if not (re.fullmatch('\s+', t) or re.fullmatch('\/\/.*\n', t) or re.match('\/\*.*\*\/', t, re.DOTALL))], j)))

freq = Counter(all_ngrams)
comm1 = dict(freq.most_common(500))
# print(comm1)

with open('scores_5Aug.json') as f:
    scores = json.load(f)
with open('scores_16Aug.json') as f:
    scores = json.load(f) | scores

with open('data.json') as f:
    data = json.load(f)
with open('data2.json') as f:
    data.extend(json.load(f))

crystalbleu = []
bleu = []
codebleu = []
score = []
refs = []
hyps = []
nexgen = False
codexglue = False
for i in data:
    currID = i['id']
    if currID < '1999':
        
        nexgen = True
        common = comm2
    else:
        continue
        codexglue = True
        common = comm1
    code1 = i['code1']
    code2 = i['code2']
    tokens1 = [t for t in list(map(lambda x: x[1], lexer.get_tokens(code1))) if not (re.fullmatch('\s+', t) or re.fullmatch('\/\/.*\n', t) or re.match('\/\*.*\*\/', t, re.DOTALL))]
    tokens2 = [t for t in list(map(lambda x: x[1], lexer.get_tokens(code2))) if not (re.fullmatch('\s+', t) or re.fullmatch('\/\/.*\n', t) or re.match('\/\*.*\*\/', t, re.DOTALL))]
    crystalbleu.append(sentence_bleu([tokens2], tokens1, smoothing_function=sm_func, ignoring=common))
    bleu.append(sentence_bleu([tokens2], tokens1, smoothing_function=sm_func))
    codebleu.append(code_bleu([[tokens2]], [tokens1]))
    score.append(int(scores[currID]['Aryaz']))
    # score.append(int(scores[currID]['Aryaz'])/5.0)
    refs.append([tokens2])
    hyps.append(tokens1)

# if nexgen:
#     with open('corr_nexgen_hyp.json', 'w') as f:
#         json.dump(hyps, f)
#     with open('corr_nexgen_ref.json', 'w') as f:
#         json.dump(refs, f)

# if codexglue:
#     with open('corr_codexglue_hyp.json', 'w') as f:
#         json.dump(hyps, f)
#     with open('corr_codexglue_ref.json', 'w') as f:
#         json.dump(refs, f)

print('Nexgen' if nexgen else 'CodeXGLUE')
# print(stats.spearmanr(crystalbleu, score))
# print(stats.spearmanr(bleu, score))
# print(stats.spearmanr(codebleu, score))
print(stats.pearsonr(crystalbleu, score))
print(stats.pearsonr(bleu, score))
print(stats.pearsonr(codebleu, score))

boxcrystal = [[crystalbleu[i] for i in range(len(score)) if score[i] == j] for j in range(6)]
boxbleu = [[bleu[i] for i in range(len(score)) if score[i] == j] for j in range(6)]

box = [boxcrystal[int(i/2)] if i%2 == 0 else boxbleu[int(i/2)] for i in range(12)]

plt.scatter(crystalbleu, score, c='b', alpha=0.7)
plt.scatter(bleu, score, c='r', alpha=0.7)
plt.scatter(codebleu, score, c='g', alpha=0.7)
# plt.boxplot(box)
# plt.xticks(list(range(1, 13)), ['Crystal'+str(int(i/2)) if i%2 == 0 else 'BLEU'+str(int(i/2)) for i in range(12)])
# plt.grid(axis='y')
plt.show()