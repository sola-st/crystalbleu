import os

list_of_files = []
for (dirpath, dirnames, filenames) in os.walk('java-small/training'):
    for filename in filenames:
        if filename.endswith('.java'):
            list_of_files.append(os.sep.join([dirpath, filename]))

filenames = list_of_files
count = 0
with open('java_data.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            try:
                for line in infile:
                    outfile.write(line)
            except:
                pass
        outfile.write('\n--------------------------=====================---------------------------------\n')
        count += 1
        if count > 6000:
            break