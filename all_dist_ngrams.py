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


pl_counts = []
lexer = JavaLexer()
with open('java_dataset/java_data.txt') as f:
# lexer = PythonLexer()
# with open('python_data.txt') as f:
    content = f.read(1793313).split('--------------------------=====================---------------------------------')
    # content = f.read(2000000)
tokens = []
for j in content:
    # tokens.extend([i for i in list(map(lambda x: x[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('\/\/.*\n', i) or re.match('\/\*.*\*\/', i, re.DOTALL))])
    # tokens.extend([i for i in list(map(lambda x: x[1], lexer.get_tokens(j))) if not (re.fullmatch('\s+', i) or re.fullmatch('#.*\n', i) or re.match('\"\"\".*\"\"\"', i, re.DOTALL))])
    tokens.extend([i[1] for i in lexer.get_tokens(j) if not (re.fullmatch('\s+', i[1]) or (i[0] in Comment))])
# print(tokens[-50:])
for i in range(1, 5):
    pl_counts.append(Counter(ngrams(tokens, i)))

nl = {}
pl = {}
for i in range(4):
    pl = pl | pl_counts[i]
pl_labels, pl_values = zip(*Counter(pl).most_common(5000))
pl_indexes = np.arange(len(pl_labels)) + 1

pl_values = np.array(pl_values, dtype='f')
pl_values /= pl_values.sum()

figure = {'figsize': (6.4, 4)}
matplotlib.rc('figure', **figure)

plt.axvspan(100, 1000, color='green', alpha=0.4)
plt.plot(pl_indexes, pl_values, label='Java', color='blue', linestyle='solid')
plt.xlabel('Most occurring n-grams')
plt.ylabel('Frequency (share of all n-grams)')
plt.grid()
# plt.legend()
# plt.yscale('log')
plt.xscale('log')
plt.savefig('chooseK.pdf', bbox_inches='tight')
plt.show()