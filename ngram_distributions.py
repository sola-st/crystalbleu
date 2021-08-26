from collections import Counter
from pygments.lexers.jvm import JavaLexer
from nltk.util import ngrams
import numpy as np
import matplotlib.pyplot as plt
import math
import re

content = ''
for x in 'abcdefghjklmnpr':
    for i in range(1, 7):
        with open('brown/c' + x + '0' + str(i)) as f:
            content += f.read()
print(len(content))
lines = content.split('\n\n')
cleanLines = [' '.join(list(map(lambda x: x.split('/')[0], l.strip().split(' ')))) for l in lines]
corpus = ' '.join(cleanLines)

nl_counts = []
for i in range(1, 5):
    nl_counts.append(Counter(ngrams(corpus.split(' '), i)))

pl_counts = []
lexer = JavaLexer()
with open('java_data.txt') as f:
    content = f.read(1793300)
print(len(content))
tokens = [i for i in list(map(lambda x: x[1], lexer.get_tokens(content))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i, re.DOTALL))]
for i in range(1, 5):
    pl_counts.append(Counter(ngrams(tokens, i)))

nl = {}
pl = {}
for i in range(4):
    # nl = nl | nl_counts[i]
    # pl = pl | pl_counts[i]
    nl = nl_counts[i]
    pl = pl_counts[i]
    print(len(nl), len(pl))

    nl_labels, nl_values = zip(*Counter(nl).most_common(2000))
    pl_labels, pl_values = zip(*Counter(pl).most_common(2000))

    nl_indexes = np.arange(len(nl_labels))
    pl_indexes = np.arange(len(pl_labels))
    width = 1

    nl_values = np.array(nl_values, dtype='f')
    nl_values /= nl_values.sum()
    pl_values = np.array(pl_values, dtype='f')
    pl_values /= pl_values.sum()

    for j in range(100, 200):
        # print(' '.join(nl_labels[j]), '&', '({:.2f})'.format(nl_values[j]*100))
        # print('\\texttt{' + ' '.join(pl_labels[j]) + '}', '&', '({:.2f})'.format(pl_values[j]*100))
        print(' '.join(pl_labels[j]))
    # print(list(zip(nl_labels[:10], nl_values[:10])))
    # print(list(zip(pl_labels[:10], pl_values[:10])))

    # plt.plot(nl_indexes, nl_values, label='English language', color='red')
    # plt.plot(pl_indexes, pl_values, label='Java language', color='blue')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.legend()
    # plt.xlabel('Most occurring n-grams')
    # plt.ylabel('Frequency (share of all n-grams)')
    # plt.grid(axis='y')
    # plt.show()
