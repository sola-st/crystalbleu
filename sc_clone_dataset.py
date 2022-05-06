import json
import random

LANG = 2
with open(f'lang{LANG}.json') as f:
    tmp = json.load(f)

codes = {}
for k, v in tmp.items():
    for i in range(len(v)):
        codes[k + '_' + str(i)] = v[i]

tests = []
cl = 0
nc = 0
for i in range(2000000):
    [a, b] = random.sample(codes.keys(), 2)
    a_id = '_'.join(a.split('_')[:-1])
    b_id = '_'.join(b.split('_')[:-1])
    if (a_id == b_id) and (cl*6 <= nc):
        tests.append((a, b, 1))
        cl += 1
    elif cl*6.5 >= nc:
        tests.append((a, b, 0))
        nc += 1

with open('sc_clone/data.jsonl', 'w') as f:
    for k, v in codes.items():
        f.write(json.dumps({'func': v, 'idx': k}) + '\n')

with open('sc_clone/test.txt', 'w') as f:
    for i, j, l in tests:
        f.write(i + ' ' + j + ' ' + str(l) + '\n')
