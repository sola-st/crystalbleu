import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as st

bs = []
cs = []

with open('clone_sim_scores.npy', 'rb') as f:
    bs.append(np.load(f))
    bs.append(np.load(f))
    cs.append(np.load(f))
    cs.append(np.load(f))

CI=0.95

berr = []
berr.append(st.t.interval(CI, len(bs[0])-1, loc=np.mean(bs[0]), scale=st.sem(bs[0])))
berr.append(st.t.interval(CI, len(bs[1])-1, loc=np.mean(bs[1]), scale=st.sem(bs[1])))
cerr = []
cerr.append(st.t.interval(CI, len(cs[0])-1, loc=np.mean(cs[0]), scale=st.sem(cs[0])))
cerr.append(st.t.interval(CI, len(cs[1])-1, loc=np.mean(cs[1]), scale=st.sem(cs[1])))

bx = [1, 2]
cx = [4, 5]
by = [np.median(bs[0]), np.median(bs[1])]
cy = [np.median(cs[0]), np.median(cs[1])]

font = {'size': 14}
matplotlib.rc('font', **font)

plt.xticks(bx + cx, ['Not clone', 'Clone'] + ['Not clone', 'Clone'])
plt.ylabel('Similarity score')
plt.bar(bx, by, yerr=berr, error_kw={'capsize': 5}, color='blue', label='BLEU', alpha=0.6)
plt.bar(cx, cy, yerr=cerr, error_kw={'capsize': 5}, color='red', label='CrystalBLEU', alpha=0.6)
plt.grid(axis='y')
plt.legend()
plt.savefig('clone_dist.pdf', bbox_inches='tight')
plt.show()
