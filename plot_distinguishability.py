import matplotlib
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt
import numpy as np

font = {'size': 14}
matplotlib.rc('font', **font)
figure = {'figsize': (10, 10)}
matplotlib.rc('figure', **figure)

with open('clone_distinguishability.npy', 'rb') as f:
# with open('java_distinguishability.npy', 'rb') as f:
    Y_intra = np.load(f)[:-1]
    Y_inter = np.load(f)[:-1]
    Y_v_intra = np.load(f)[:-1]
    Y_v_inter = np.load(f)[:-1]

with open('sc_clone_distinguishability.npy', 'rb') as f:
# with open('java_distinguishability.npy', 'rb') as f:
    scY_intra = np.load(f)
    scY_inter = np.load(f)
    scY_v_intra = np.load(f)
    scY_v_inter = np.load(f)

X = [3**i for i in range(11)]

# plt.ylim([0.8, 10])
epsilon = 1e-16

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

# ax1.set_ylim(0, 20)
# ax3.set_ylim(0, 20)
ax2.set_ylim(0, 0.25)
ax4.set_ylim(0, 0.5)

ax1.axvspan(100, 1000, color='green', alpha=0.4)
lns1 = ax1.plot(X, np.array(Y_intra) / (np.array(Y_inter) + epsilon), label='CrystalBLEU', color='red', linestyle='solid')
lns2 = ax1.plot(X, np.array(Y_v_intra) / (np.array(Y_v_inter) + epsilon), label='BLEU', color='blue', linestyle='dashed')
ax2.axvspan(100, 1000, color='green', alpha=0.4)
lns3 = ax2.plot(X, np.array(Y_intra), label='intra-class', color='green', linestyle='solid')
lns4 = ax2.plot(X, np.array(Y_inter), label='inter-class', color='black', linestyle='dashed')
ax3.axvspan(100, 1000, color='green', alpha=0.4)
lns1 = ax3.plot(X, np.array(scY_intra) / (np.array(scY_inter) + epsilon), label='CrystalBLEU', color='red', linestyle='solid')
lns2 = ax3.plot(X, np.array(scY_v_intra) / (np.array(scY_v_inter) + epsilon), label='BLEU', color='blue', linestyle='dashed')
ax4.axvspan(100, 1000, color='green', alpha=0.4)
lns3 = ax4.plot(X, np.array(scY_intra), label='intra-class', color='green', linestyle='solid')
lns4 = ax4.plot(X, np.array(scY_inter), label='inter-class', color='black', linestyle='dashed')
# lns = lns1 + lns2 + lns3 + lns4
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc='upper center')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')
ax3.set_yscale('log')
ax1.set_ylabel('Distinguishability')
ax2.set_ylabel('CrystalBLEU score')
ax4.set_xlabel('k')
ax3.set_ylabel('Distinguishability')
ax4.set_ylabel('CrystalBLEU score')

ax1.set_yticks([2, 3, 4, 6, 10])
ax1.set_yticklabels(['2', '3', '4', '6', '10'])
# ax1.set_yticks([2 * 5**i for i in range(5)])#, [2 * 5**i for i in range(5)])
# plt.ylabel('Intra-class and inter-class difference')
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
# plt.legend()
plt.savefig('effect_of_k.pdf', bbox_inches='tight')
plt.show()