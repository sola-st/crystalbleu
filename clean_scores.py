import json

with open('scores2.json') as f:
    content = json.load(f)

res = {str(i): j for i, j in enumerate(content) if j is not None}

with open('scores_16Aug.json', 'w') as f:
    json.dump(res, f, indent=2)