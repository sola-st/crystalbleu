from collections import Counter
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.python import PythonLexer
from pygments.token import Comment
from nltk.util import ngrams
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import re

font = {'size': 14}
matplotlib.rc('font', **font)

content = ''
for x in 'abcdefghjklmnpr':
    for i in range(1, 7):
        with open('brown/c' + x + '0' + str(i)) as f:
            content += f.read()
print(len(content))
lines = content.split('\n\n')
cleanLines = [' '.join(list(map(lambda x: x.split('/')[0], l.strip().split(' ')))) for l in lines]
corpus = ' '.join(cleanLines).replace('? ?', '?').replace('! !', '!').replace('; ;', ';')

nl_counts = []
for i in range(1, 5):
    nl_counts.append(Counter(ngrams(corpus.split(' '), i)))

pl_counts = []
lexer = JavaLexer()
with open('java_data.txt') as f:
# lexer = PythonLexer()
# with open('python_data.txt') as f:
    content = f.read(1793313).split('--------------------------=====================---------------------------------')
    # content = f.read(2000000)
tokens = []
for j in content:
    # tokens.extend([i for i in list(map(lambda x: x[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i, re.DOTALL))])
    # tokens.extend([i for i in list(map(lambda x: x[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('#.*\n', i) or re.match('\"\"\".*\"\"\"', i, re.DOTALL))])
    tokens.extend([i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))])
print(tokens[-50:])
for i in range(1, 5):
    pl_counts.append(Counter(ngrams(tokens, i)))

nl = {}
pl = {}
fig, axs = plt.subplots(1, 4, figsize=(16, 4.8), sharey=True, sharex=True)
axs[0].set_ylabel('Frequency (share of\n same-length n-grams)')
for i in range(4):
    # nl = nl | nl_counts[i]
    # pl = pl | pl_counts[i]
    nl = nl_counts[i]
    pl = pl_counts[i]
    # print(len(nl), len(pl))

    nl_labels, nl_values = zip(*Counter(nl).most_common(5000))
    pl_labels, pl_values = zip(*Counter(pl).most_common(5000))

    nl_indexes = np.arange(len(nl_labels)) + 1
    pl_indexes = np.arange(len(pl_labels)) + 1
# width = 1

    nl_values = np.array(nl_values, dtype='f')
    nl_values /= nl_values.sum()
    pl_values = np.array(pl_values, dtype='f')
    pl_values /= pl_values.sum()

    # for j in range(100, 200):
        # print(' '.join(nl_labels[j]), '&', '({:.2f})'.format(nl_values[j]*100))
        # print('\\texttt{' + ' '.join(pl_labels[j]) + '}', '&', '({:.2f})'.format(pl_values[j]*100))
        # print(' '.join(pl_labels[j]))
    print(f'{i+1}-grams:')
    print(list(zip(nl_labels[:10], nl_values[:10])))
    print(list(zip(pl_labels[:10], pl_values[:10])))

# plt.plot(nl_indexes, nl_values, label='English language', color='red')
# plt.plot(pl_indexes, pl_values, label='Java language', color='blue')
# plt.yscale('log')
# plt.xscale('log')
# plt.legend()
# plt.xlabel('Most occurring n-grams')
# plt.ylabel('Frequency (share of all n-grams)')
# plt.grid(axis='y')
# plt.show()

    axs[i].plot(nl_indexes, nl_values, label='English', color='red', linestyle='solid')
    axs[i].plot(pl_indexes, pl_values, label='Java', color='blue', linestyle='dashed')
    axs[i].set_yscale('log')
    axs[i].set_xscale('log')
    axs[i].grid()
    axs[i].set_xlabel('Most occurring {}-grams'.format(i+1))
    axs[i].set_xticks([1, 10, 100, 1000])
    axs[i].set_aspect(1)
axs[3].legend()
handles, labels = axs[3].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
plt.savefig('NLPLDist.pdf', bbox_inches='tight')
# plt.plot(nl_indexes, nl_values, label='English', color='red', linestyle='dashed')
# plt.plot(pl_indexes, pl_values, label='Java', color='blue', linestyle='solid')
# plt.xlabel('Most occurring n-grams')
# plt.ylabel('Frequency (share of all n-grams)')
# plt.grid()
# # plt.legend()
# # plt.yscale('log')
# plt.xscale('log')
# plt.savefig('chooseK.pdf')
plt.show()