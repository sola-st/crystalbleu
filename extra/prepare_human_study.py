import json
import random
from bleu_ignoring import sentence_bleu

N = 1000
n = 100

# nexgen
with open('nexgen/baseline-100k.out') as f:
    model1 = f.readlines()
with open('nexgen/multi_slicing-100k.out') as f:
    model2 = f.readlines()
with open('nexgen/tgt-test.txt') as f:
    ref = f.readlines()

with open('data.json') as f:
    old = [(x['code1'], x['code2']) for x in json.load(f)]

pairs = []
s = random.sample(range(len(ref)), int(N/3))
for i in s:
    pairs.append((model1[i], ref[i]))
s = random.sample(range(len(ref)), int(N/3))
for i in s:
    pairs.append((model2[i], ref[i]))
s = random.sample(range(len(ref)), int(N/3))
for i in s:
    pairs.append((ref[i], ref[(i+1)%(len(ref))]))

sorted_pairs = sorted(pairs, key=lambda x: sentence_bleu([x[1]], x[0]))
selection = []
for i in range(n):
    j = 0
    while j < int(N/(3*n)):
        if sorted_pairs[i * int(N/(3*n)) + j + int(N/3)] in old:
            j += 1
        else:
            break
    selection.append(sorted_pairs[i * int(N/(3*n)) + j + int(N/3)])

obj = [{"id": str(1200+i), "code1": selection[i][0], "code2": selection[i][1]} for i in range(len(selection))]

# codexglue cs-java
with open('codexglue/cs-java-model1.output') as f:
    model1 = f.readlines()
with open('codexglue/cs-java-model2.output') as f:
    model2 = f.readlines()
with open('codexglue/test.java-cs.txt.java') as f:
    ref = f.readlines()

pairs = []
s = random.sample(range(len(ref)), int(N/3))
for i in s:
    pairs.append((model1[i], ref[i]))
s = random.sample(range(len(ref)), int(N/3))
for i in s:
    pairs.append((model2[i], ref[i]))
s = random.sample(range(len(ref)), int(N/3))
for i in s:
    pairs.append((ref[i], ref[(i+1)%(len(ref))]))

sorted_pairs = sorted(pairs, key=lambda x: sentence_bleu([x[1]], x[0]))
selection = []
for i in range(n):
    j = 0
    while j < int(N/(3*n)):
        if sorted_pairs[i * int(N/(3*n)) + j + int(N/3)] in old:
            j += 1
        else:
            break
    selection.append(sorted_pairs[i * int(N/(3*n)) + j + int(N/3)])

obj.extend([{"id": str(2200+i), "code1": selection[i][0], "code2": selection[i][1]} for i in range(len(selection))])

with open('data3.json', 'w') as f:
    json.dump(obj, f, indent=2)