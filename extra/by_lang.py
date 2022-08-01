import json

with open('runs.json') as f:
    data = json.load(f)

codes = [{}, {}, {}]
for dp in data:
    if dp['lang'] == 3:
        lang = 2
    elif dp['lang'] < 2:
        lang = dp['lang']
    else:
        continue
    prob = str(dp['section_id']) + '_' + str(dp['problem_id'])
    if prob not in codes[lang]:
        codes[lang][prob] = []
    codes[lang][prob].append(dp['source'])

for i in range(3):
    new_codes = {}
    for k, v in codes[i].items():
        if len(v) >= 3:
            new_codes[k] = codes[i][k]
    with open('lang' + str(i) + '.json', 'w') as f:
        json.dump(new_codes, f, indent=4)
