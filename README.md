# Wikigen

This repository contains the code and data for the paper ["An Edit-centric Approach for Wikipedia Article Quality Assessment"](https://arxiv.org/abs/1909.08880). If you use or code or data please consider citing our work.



## Setup

1. Clone this repo:                                                                                                           

   ```bash
   git clone https://github.com/epochx/wikigen
   cd wikigen
   ```

2. Create a conda environment and activate it

   ```bash
   conda create -n <name> python=3.6
   conda activate <name>
   ```

   Where you can replace `<name>` by whatever you want.

3. Install dependencies everything

   ```bash
   sh ./install.sh
   ```

   This script will install all the dependencies using conda and/or pip.

4. Download data

   ```	bash
   sh ./download.sh
   ```

   By default, the data will be downloaded in `~/data/wikigen/` If you want to change this, make sure to edit the value of`DATA_PATH` accordingly in  the file `~/wikigen/wikigen/settings.py`



## Running

By default the output of training a model will go to `~/results/wikigen` If you want to change this, please modify the value of `RESULTS_PATH` in the file `~/wikigen/wikigen/settings.py` or change the `results_path` parameter when running a model. 



1. To train and evaluate a classifier model execute the following command.

   ```bash
   python train_classifier.py --config wikigen/config/classifier.yaml
   ```

   - You can modify the or provide parameters by changing the `classifier.yaml` file, or by using the command line. For help, run `python train_classifier.py --help`  for additional details.

2. To train and evaluate models including the auxiliary generative tasks, run:

```bash
python train_seq2seq.py --config wikigen/config/seq2seq.yaml
```

- You can modify the or provide parameters by changing the `seq2seq.yaml` file, or by using the command line. For help, run `python train_seq2seq.py --help`  for additional details.



## Running the doc2vec baseline

We are releasing the pre-processed datasets that we utilize in our paper, including a pre-processed version of the "Wikiclass" dataset, officially known as  the "English Wikipedia Quality Assessment Dataset". However, to run our baseline you also need to download the original version.  

. To obtain this dataset please go to this [link](https://figshare.com/articles/English_Wikipedia_Quality_Assessment_Dataset/1375406) and download the file `2017_english_wikipedia_quality_dataset.tar.bz2`. There are two versions of the dataset, but in our work we use the most recent, 2017 version since the one from 2015 is maintained for historical reasons. 



1. Pre-process the Wikiclass dataset and train doc2vec using the following command:

```
python preprocess_doc2vec_wikiclass.py
```

This process should take approximately 30 mins.

3. Load the vectors obtained by doc2vec and run the classifier following command:

```bash
python train_doc2vec_wikiclass_classifier.py
```

