import matplotlib
from matplotlib import pyplot as plt
import numpy as np

font = {'size': 14}
matplotlib.rc('font', **font)

with open('java_distinguishability.npy', 'rb') as f:
    Y_intra = np.load(f)
    Y_inter = np.load(f)
    Y_v_intra = np.load(f)
    Y_v_inter = np.load(f)

X = [3**i for i in range(12)]

# plt.ylim([0.8, 10])
epsilon = 1e-16
plt.plot(X, np.array(Y_intra) / (np.array(Y_inter) + epsilon), label='CrystalBLEU', color='red', linestyle='solid')
plt.plot(X, np.array(Y_v_intra) / (np.array(Y_v_inter) + epsilon), label='BLEU', color='blue', linestyle='dashed')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('Distinguishability')
plt.yticks([2 * 5**i for i in range(5)], [2 * 5**i for i in range(5)])
# plt.ylabel('Intra-class and inter-class difference')
plt.grid()
plt.legend()
plt.savefig('effect_of_k.pdf')
plt.show()