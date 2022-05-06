import random
import re
import json
from collections import Counter
from mosestokenizer import *
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.c_like import CLexer
from pygments.token import Comment
from nltk.util import ngrams
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

TH = 2080

def is_code_py(s):
    tokens = [i[1] for i in lexer.get_tokens(s) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
    if 10 < len(tokens) < 700:
        for i in s.split('\n'):
            if (not re.fullmatch('^\s*#.*$', i)) and (not re.fullmatch('^\s*$', i)):
                return True
    return False

def is_code_java(s):
    tokens = [i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
    if 10 < len(tokens) < 700:
        for i in s.split('\n'):
            if (not re.fullmatch('^\s*\/\/.*$', i)) and (not re.fullmatch('^\s*$', i)):
                return True
    return False

print('English Language')
corpus = []
l = []
for x in 'abcdefghjklmnpr':
    for i in range(1, 7):
        with open('brown/c' + x + '0' + str(i)) as f:
            content = f.read()
            lines = content.split('\n\n')
            cleanLines = [' '.join(list(map(lambda x: x.split('/')[0], l.strip().split(' ')))) for l in lines if len(l) > 0]
            # corpus.extend(cleanLines)
            corpus.append('\n'.join(cleanLines[:3]).replace('? ?', '?').replace('! !', '!').replace('; ;', ';'))
            # print(re.split('[ \n]', corpus[-1]))
            l.append(len(re.split('[ \n]', corpus[-1])))

print(np.mean(l[:TH]), np.std(l[:TH]))

en_corpus_results = []
nc = []
for i in range(len(corpus)):
    c = {}
    for n in range(1, 5):
        c |= Counter(ngrams(re.split('[ \n]', corpus[i]), n))
    nc.append(c)
    for j in range(i):
        res = 0
        total = 0
        for k, v in c.items():
            total += v
            if k in nc[j]:
                res += min(v, nc[j][k])
        en_corpus_results.append(res/total)
        total = 0
        for k, v in nc[j].items():
            total += v
        en_corpus_results.append(res/total)

tokenize = MosesTokenizer('fr')
print('French Language')
corpus = []
l = []
with open('europarl-v7.fr-en.fr') as f:
    content = f.read()
    lines = content.split('\n')

for i in range(0, 1515, 14):
    corpus.append(' '.join(lines[i:i+14-random.randint(1, 13)]))
    # print(corpus[-1])
    l.append(len(tokenize(corpus[-1])))
print(np.mean(l[:100]), np.std(l[:100]))
corpus = corpus[:100]

fr_corpus_results = []
nc = []
for i in range(len(corpus)):
    c = {}
    tokenized = tokenize(corpus[i])
    for n in range(1, 5):
        c |= Counter(ngrams(tokenized, n))
    nc.append(c)
    for j in range(i):
        res = 0
        total = 0
        for k, v in c.items():
            total += v
            if k in nc[j]:
                res += min(v, nc[j][k])
        fr_corpus_results.append(res/total)
        total = 0
        for k, v in nc[j].items():
            total += v
        fr_corpus_results.append(res/total)
    

lexer = PythonLexer()
with open('python_data.txt') as f:
    content = list(filter(is_code_py, f.read().split('--------------------------=====================---------------------------------')[:5000:25]))

print('Python')
l = []
nc = []
lt = []
py_corpus_results = []
for x in range(len(content)):
    c = {}
    j = content[x]
    tokens = [i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
    l.append(len(tokens))
    # tokens = [i for i in list(map(lambda x: x[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('#.*', i) or re.match('\"\"\".*\"\"\"', i, re.DOTALL))]
    for i in range(1, 5):
        c |= Counter(ngrams(tokens, i))
    nc.append(c)
    lt.append(len(tokens))
    for y in range(len(nc)):
        # if not((j != content[y]) and (0.5 < lt[x]/lt[y] < 2)):
        #     continue
        res = 0
        total = 0
        for k, v in c.items():
            total += v
            if k in nc[y]:
                res += min(v, nc[y][k])
        if True:# res < total:
            # if res/total < 0.02:
            #     print(content[x])
            #     print(content[y])
            #     exit(0)
            py_corpus_results.append(res/total)
            # total = 0
            # for k, v in nc[y].items():
            #     total += v
            # py_corpus_results.append(res/total)
print(np.mean(l[:TH]), np.std(l[:TH]))

lexer = JavaLexer()
with open('java_data.txt') as f:
    content = list(filter(is_code_java, f.read().split('--------------------------=====================---------------------------------')[:5000:25]))


print('Java')
l = []
nc = []
lt = []
java_corpus_results = []
for x in range(len(content)):
    c = {}
    j = content[x]
    tokens = [i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
    if (len(tokens) < 40) or (len(tokens) > 800):
        continue
    l.append(len(tokens))
    # tokens = [i for i in list(map(lambda x: x[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i, re.DOTALL))]
    for i in range(1, 5):
        c |= Counter(ngrams(tokens, i))
    nc.append(c)
    lt.append(len(tokens))
    for y in range(len(nc)):
        # if not((j != content[y]) and (0.5 < lt[x]/lt[y] < 2)):
        #     continue
        res = 0
        total = 0
        for k, v in c.items():
            total += v
            if k in nc[y]:
                res += min(v, nc[y][k])
        if True:#res < total:
            # if res/total < 0.02:
            #     print(content[x])
            #     print(content[y])
            #     exit(0)
            java_corpus_results.append(res/total)
            # total = 0
            # for k, v in nc[y].items():
            #     total += v
            # java_corpus_results.append(res/total)

print(np.mean(l[:TH]), np.std(l[:TH]))


lexer = CLexer()
with open('codexglue/large_cpp.jsonl') as f:
    tmp = list(f.read().split('\n'))
print(len(tmp))
content = []
used = set()
for t in tmp:
    try:
        x = json.loads(t)
        if x['label'] not in used:
            content.append(x['code'])
            used.add(x['label'])
    except:
        break

print('C/C++')
l = []
nc = []
lt = []
c_corpus_results = []
for x in range(len(content)):
    c = {}
    j = content[x]
    tokens = [i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
    l.append(len(tokens))
    # tokens = [i for i in list(map(lambda x: x[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('#.*', i) or re.match('\"\"\".*\"\"\"', i, re.DOTALL))]
    for i in range(1, 5):
        c |= Counter(ngrams(tokens, i))
    nc.append(c)
    lt.append(len(tokens))
    for y in range(len(nc)):
        # if not((j != content[y]) and (0.5 < lt[x]/lt[y] < 2)):
        #     continue
        res = 0
        total = 0
        for k, v in c.items():
            total += v
            if k in nc[y]:
                res += min(v, nc[y][k])
        if True:# res < total:
            # if res/total < 0.02:
            #     print(content[x])
            #     print(content[y])
            #     exit(0)
            c_corpus_results.append(res/total)
            # total = 0
            # for k, v in nc[y].items():
            #     total += v
            # c_corpus_results.append(res/total)
print(np.mean(l[:TH]), np.std(l[:TH]))

print(len(en_corpus_results), len(fr_corpus_results), len(py_corpus_results), len(java_corpus_results), len(c_corpus_results))

en_corpus_results = en_corpus_results[:TH]
fr_corpus_results = fr_corpus_results[:TH]
py_corpus_results = py_corpus_results[:TH]
java_corpus_results = java_corpus_results[:TH]
c_corpus_results = c_corpus_results[:TH]
print(np.mean(en_corpus_results))
print(np.mean(fr_corpus_results))
print(np.mean(py_corpus_results))
print(np.mean(java_corpus_results))
print(np.mean(c_corpus_results))

BINS = 40
n_en, x_en = np.histogram(en_corpus_results, bins=BINS, range=(0.0, 1.0))
n_fr, x_fr = np.histogram(fr_corpus_results, bins=BINS, range=(0.0, 1.0))
n_py, x_py = np.histogram(py_corpus_results, bins=BINS, range=(0.0, 1.0))
n_java, x_java = np.histogram(java_corpus_results, bins=BINS, range=(0.0, 1.0))
n_c, x_c = np.histogram(c_corpus_results, bins=BINS, range=(0.0, 1.0))
xs_en = 0.5*(x_en[1:] + x_en[:-1])
xs_fr = 0.5*(x_fr[1:] + x_fr[:-1])
xs_py = 0.5*(x_py[1:] + x_py[:-1])
xs_java = 0.5*(x_java[1:] + x_java[:-1])
xs_c = 0.5*(x_c[1:] + x_c[:-1])

font = {'size': 14}
matplotlib.rc('font', **font)

plt.xlabel('Ratio of shared n-grams')
plt.ylabel('Number of pairs')
plt.plot(xs_en, n_en, color='b', linestyle='solid', label='English')
plt.plot(xs_en, n_en, color='r', linestyle='dashed', label='French')
plt.plot(xs_py, n_py, color='black', linestyle='dashed', label='Python')
plt.plot(xs_java, n_java, color='g', linestyle='dashdot', label='Java')
plt.plot(xs_c, n_c, color='r', linestyle='dotted', label='C/C++')
plt.grid()
plt.legend()
plt.savefig('shared_ngrams_pairs.pdf')
plt.show()