import os

list_of_files = []
for (dirpath, dirnames, filenames) in os.walk('python-corpus/cleaned/'):
    for filename in filenames:
        if filename.endswith('.py'):
            list_of_files.append(os.sep.join([dirpath, filename]))
    if len(list_of_files) > 5000:
        break

filenames = list_of_files
with open('python_data.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            try:
                for line in infile:
                    outfile.write(line)
            except:
                pass
        outfile.write('\n--------------------------=====================---------------------------------\n')