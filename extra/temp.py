import json
import numpy as np

with open('data.json') as f:
    data = json.load(f)

x1 = []
x2 = []
for i in data:
    if i['id'] < '1999':
        x1.append(len(i['code1']))
        x1.append(len(i['code2']))
    else:
        x2.append(len(i['code1']))
        x2.append(len(i['code2']))
X1 = np.array(x1)
X2 = np.array(x2)
print(X1.min(), X1.mean(), X1.max())
print(X2.min(), X2.mean(), X2.max())