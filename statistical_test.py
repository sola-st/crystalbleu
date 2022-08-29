import math
import random
from bleu_ignoring import corpus_bleu, SmoothingFunction

def stat_test(references, neural, dummy, ignoring):
    bleu_neural = corpus_bleu(references, neural, smoothing_function=SmoothingFunction().method1)
    bleu_dummy = corpus_bleu(references, dummy, smoothing_function=SmoothingFunction().method1)
    crystalbleu_neural = corpus_bleu(references, neural, ignoring=ignoring, smoothing_function=SmoothingFunction().method1)
    crystalbleu_dummy = corpus_bleu(references, dummy, ignoring=ignoring, smoothing_function=SmoothingFunction().method1)
    print('BLEU neural: {}'.format(bleu_neural))
    print('BLEU dummy: {}'.format(bleu_dummy))
    print('CrystalBLEU neural: {}'.format(crystalbleu_neural))
    print('CrystalBLEU dummy: {}'.format(crystalbleu_dummy))
    c_bleu = 0
    c_crystalbleu = 0
    R = 1000
    for i in range(R):
        n = neural.copy()
        d = dummy.copy()
        for j in range(len(neural)):
            if random.random() < 0.5:
                n[j], d[j] = d[j], n[j]
        bleu_neural_new = corpus_bleu(references, n, smoothing_function=SmoothingFunction().method1)
        n = neural.copy()
        d = dummy.copy()
        for j in range(len(neural)):
            if random.random() < 0.5:
                n[j], d[j] = d[j], n[j]
        bleu_dummy_new = corpus_bleu(references, d, smoothing_function=SmoothingFunction().method1)
        n = neural.copy()
        d = dummy.copy()
        for j in range(len(neural)):
            if random.random() < 0.5:
                n[j], d[j] = d[j], n[j]
        crystalbleu_neural_new = corpus_bleu(references, n, ignoring=ignoring, smoothing_function=SmoothingFunction().method1)
        n = neural.copy()
        d = dummy.copy()
        for j in range(len(neural)):
            if random.random() < 0.5:
                n[j], d[j] = d[j], n[j]
        crystalbleu_dummy_new = corpus_bleu(references, d, ignoring=ignoring, smoothing_function=SmoothingFunction().method1)
        if bleu_neural - bleu_dummy < bleu_neural_new - bleu_dummy_new:
            c_bleu += 1
        if crystalbleu_neural - crystalbleu_dummy < crystalbleu_neural_new - crystalbleu_dummy_new:
            c_crystalbleu += 1
    print('p-value for BLEU: {}'.format(c_bleu / R))
    print('p-value for CrystalBLEU: {}'.format(c_crystalbleu / R))