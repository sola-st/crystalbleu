import matplotlib
from matplotlib import pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)
# figure = {'figsize': (8, 5)}
# matplotlib.rc('figure', **figure)

plt.ylabel('Similarity score')
plt.xticks([1.5, 4.5, 7.5], ['BLEU', 'CodeBLEU', 'CrystalBLEU'])
plt.bar([1, 4, 7], [0.1667, 0.2310, 0.1142], label='Neural Model', color='b', alpha=0.6)
plt.bar([2, 5, 8], [0.1665, 0.2135, 0.0915], label='Dummy Model', color='r', alpha=0.6)
# plt.xticks([1.5, 4.5], ['BLEU', 'CrystalBLEU'])
# plt.bar([1, 4], [0.1667, 0.1142], label='Neural Model', color='b', alpha=0.6)
# plt.bar([2, 5], [0.1665, 0.0915], label='Dummy Model', color='r', alpha=0.6)
plt.grid(axis='y')
plt.legend()
plt.savefig('RQ2.pdf', bbox_inches='tight')
plt.show() 