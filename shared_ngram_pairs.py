import re
from collections import Counter
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.python import PythonLexer
from pygments.token import Comment
from nltk.util import ngrams
import numpy as np
import matplotlib.pyplot as plt

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

print('Natural Language')
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

print(np.mean(l), np.std(l))

# corpus level
nl_corpus_results = []
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
        nl_corpus_results.append(res/total)
        total = 0
        for k, v in nc[j].items():
            total += v
        nl_corpus_results.append(res/total)
    

lexer = PythonLexer()
with open('python_data.txt') as f:
    content = list(filter(is_code_py, f.read().split('--------------------------=====================---------------------------------')[:5000:50]))

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
            total = 0
            for k, v in nc[y].items():
                total += v
            py_corpus_results.append(res/total)
print(np.mean(l), np.std(l))

lexer = JavaLexer()
with open('java_data.txt') as f:
    content = list(filter(is_code_java, f.read().split('--------------------------=====================---------------------------------')[:5000:50]))


print('Java')
l = []
nc = []
lt = []
java_corpus_results = []
for x in range(len(content)):
    c = {}
    j = content[x]
    tokens = [i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))]
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
            total = 0
            for k, v in nc[y].items():
                total += v
            java_corpus_results.append(res/total)

print(np.mean(l), np.std(l))


print(len(nl_corpus_results), len(py_corpus_results), len(java_corpus_results))

nl_corpus_results = nl_corpus_results[:4200]
py_corpus_results = py_corpus_results[:4200]
java_corpus_results = java_corpus_results[:4200]
print(np.mean(nl_corpus_results))
print(np.mean(py_corpus_results))
print(np.mean(java_corpus_results))

n_nl, x_nl = np.histogram(nl_corpus_results, bins=40, range=(0.0, 1.0))
n_py, x_py = np.histogram(py_corpus_results, bins=40, range=(0.0, 1.0))
n_java, x_java = np.histogram(java_corpus_results, bins=40, range=(0.0, 1.0))
xs_nl = 0.5*(x_nl[1:] + x_nl[:-1])
xs_py = 0.5*(x_py[1:] + x_py[:-1])
xs_java = 0.5*(x_java[1:] + x_java[:-1])
plt.xlabel('Ratio of shared n-grams')
plt.ylabel('Number of pairs')
plt.plot(xs_nl, n_nl, color='b', label='Natural Language')
plt.plot(xs_py, n_py, color='r', label='Python')
plt.plot(xs_java, n_java, color='g', label='Java')
plt.legend()
plt.show()