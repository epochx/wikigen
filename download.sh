mkdir -p ~/results/wikigen/doc2vec
mkdir -p ~/data/

wget https://zenodo.org/record/3525794/files/wikigen.zip?download=1 ~/data

unzip ~/data/wikigen.zip

wget https://ndownloader.figshare.com/files/9301984 ~/data/wikigen/wikiclass.tar.gz

tar -xvzf wikiclass.tar.gz