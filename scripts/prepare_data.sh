if [[ $@ =~ "java" ]]; then
    echo "Preparing Java dataset..."
    mkdir java_dataset
    cd java_dataset
    wget https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz
    tar -xzf java-small.tar.gz
    cp ../java_corpus.py .
    python java_corpus.py
    cd ..
    echo "done"
fi
if [[ $@ =~ "python" ]]; then
    echo "Preparing Python dataset..."
    mkdir python_dataset
    cd python_dataset
    wget http://files.srl.inf.ethz.ch/data/py150_files.tar.gz
    tar -xzf py150_files.tar.gz
    tar -xzf data.tar.gz
    cp ../python_corpus.py .
    python python_corpus.py
    cd ..
    echo "done"
fi
if [[ $@ =~ "c" ]]; then
    echo "Preparing C dataset..."
    mkdir c_dataset
    cd c_dataset
    pip install gdown
    gdown https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU
    tar -xzf programs.tar.gz
    cp ../c_corpus.py .
    pip install tqdm
    python c_corpus.py
    cd ..
    echo "done"
fi
if [[ $@ =~ "english" ]]; then
    echo "Preparing English dataset..."
    mkdir english_dataset
    cd english_dataset
    wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip
    unzip brown.zip
    cd ..
    echo "done"
fi
if [[ $@ =~ "french" ]]; then
    echo "Preparing French dataset..."
    mkdir french_dataset
    cd french_dataset
    wget https://www.statmt.org/europarl/v7/fr-en.tgz
    tar -xzf fr-en.tgz
    cd ..
    echo "done"
fi