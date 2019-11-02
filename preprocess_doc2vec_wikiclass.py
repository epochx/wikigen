# coding: utf-8
from wikigen.settings import DATA_PATH, DOC2VEC_PATH
import pandas as pd
import os
import mwparserfromhell
import tqdm
import json
import argparse

import gensim
from gensim.models import Doc2Vec
import smart_open
import logging
import sys

WIKICLASS_PATH = os.path.join(DATA_PATH, 'wikiclass')

train_file_path = os.path.join(
    WIKICLASS_PATH, 'datasets', 'training-set.tsv')

test_file_path = os.path.join(
    WIKICLASS_PATH, 'datasets', 'test-set.tsv')

wikiclass_files_path = os.path.join(
    WIKICLASS_PATH, 'revisiondata')

train_clean_wikiclass_files_path = os.path.join(
    WIKICLASS_PATH, 'train_clean_revisiondata')

test_clean_wikiclass_files_path = os.path.join(
    WIKICLASS_PATH, 'test_clean_revisiondata')

train_doc_file_path = os.path.join(
    train_clean_wikiclass_files_path, 'documents.txt')

train_label_file_path = os.path.join(
    train_clean_wikiclass_files_path, 'labels.txt')

train_revids_file_path = os.path.join(
    train_clean_wikiclass_files_path, 'revids.txt')

test_doc_file_path = os.path.join(
    test_clean_wikiclass_files_path, 'documents.txt')

test_label_file_path = os.path.join(
    test_clean_wikiclass_files_path, 'labels.txt')

test_revids_file_path = os.path.join(
    test_clean_wikiclass_files_path, 'revids.txt')


if __name__ == '__main__':

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_test', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(train_clean_wikiclass_files_path):
        os.makedirs(train_clean_wikiclass_files_path)

    if not os.path.exists(test_clean_wikiclass_files_path):
        os.makedirs(test_clean_wikiclass_files_path)

    train_data = pd.read_csv(train_file_path, sep='\t')

    train_rev_ids = train_data['article_revid'].tolist()
    train_ratings = train_data['rating'].tolist()


    if not os.path.exists(train_doc_file_path):
        print('Writing training data')

        doc_file = open(train_doc_file_path, 'w')
        label_file = open(train_label_file_path, 'w')
        revids_file = open(train_revids_file_path, 'w')

        tuples = list(zip(train_rev_ids, train_ratings))

        for rev_id, rating in tqdm.tqdm(tuples):

            file_path = os.path.join(
                wikiclass_files_path, str(rev_id))

            with open(file_path) as f:
                data  = f.read()

            clean_data = mwparserfromhell.parse(data).strip_code()
            clean_data = clean_data.replace('\n', ' ')

            doc_file.write(f'{clean_data}\n')
            label_file.write(f'{rating}\n')
            revids_file.write(f'{rev_id}\n')

        doc_file.close()
        label_file.close()
        revids_file.close()

    test_data = pd.read_csv(test_file_path, sep='\t')

    test_rev_ids = test_data['article_revid'].tolist()
    test_ratings = test_data['rating'].tolist()

    if not os.path.exists(test_doc_file_path):

        print('Writing testing data')

        doc_file = open(test_doc_file_path, 'w')
        label_file = open(test_label_file_path, 'w')
        revids_file = open(test_revids_file_path, 'w')

        tuples = list(zip(test_rev_ids, test_ratings))

        for rev_id, rating in tqdm.tqdm(tuples):

            file_path = os.path.join(
                wikiclass_files_path, str(rev_id))

            with open(file_path) as f:
                data  = f.read()

            clean_data = mwparserfromhell.parse(data).strip_code()
            clean_data = clean_data.replace('\n',  ' ')

            doc_file.write(f'{clean_data}\n')
            label_file.write(f'{rating}\n')
            revids_file.write(f'{rev_id}\n')

        doc_file.close()
        label_file.close()
        revids_file.close()


    def read_corpus(fname, tokens_only=False):
        with smart_open.open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                if tokens_only:
                    yield gensim.utils.simple_preprocess(line)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(
                        gensim.utils.simple_preprocess(line), [i])

    print('Reading corpus...')
    train_corpus = list(read_corpus(train_doc_file_path))

    test_corpus = list(read_corpus(test_doc_file_path,
                                   tokens_only=not args.use_test))

    if args.use_test:
        corpus = train_corpus + test_corpus
    else:
        corpus = train_corpus

    if args.use_test:
        doc2vec_save_path = os.path.join(
            DOC2VEC_PATH, 'wikiclass_train.d2v')

        train_vector_file_path = os.path.join(
            train_clean_wikiclass_files_path, 'vectors.jsonl')

        test_vector_file_path = os.path.join(
            test_clean_wikiclass_files_path, 'vectors.jsonl')

    else:
        doc2vec_save_path = os.path.join(
            DOC2VEC_PATH, 'wikiclass_train.no_test.d2v')

        train_vector_file_path = os.path.join(
            train_clean_wikiclass_files_path,
            'vectors.no_test.jsonl')

        test_vector_file_path = os.path.join(
            test_clean_wikiclass_files_path,
            'vectors.no_test.jsonl')

    if not os.path.exists(doc2vec_save_path):

        model = Doc2Vec(min_count=1, window=10, size=500,
                        sample=1e-4, negative=5, epochs=50,
                        workers=12)

        model.build_vocab(corpus)

        model.train(corpus,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        model.save(doc2vec_save_path)

    model = Doc2Vec.load(doc2vec_save_path)

    len_train = len(train_corpus)
    len_test = len(test_corpus)

    if not os.path.exists(train_vector_file_path):
        print('Writing train vectors...')

        with open(train_vector_file_path, 'w') as f:
            for i in range(len_train):
                vector_i = model.docvecs[i].tolist()
                json_vector_i = json.dumps(vector_i)
                f.write(f'{json_vector_i}\n')


    if not os.path.exists(test_vector_file_path):
        print('Writing test vectors...')

        if args.use_test:
            with open(train_vector_file_path, 'w') as f:
                for i in range(len_train, len_train + len_test):
                    vector_i = model.docvecs[i].tolist()
                    json_vector_i = json.dumps(vector_i)
                    f.write(f'{json_vector_i}\n')

        else:
            with open(test_vector_file_path, 'w') as f:
                for example in test_corpus:
                    vector_i = model.infer_vector(example).tolist()
                    json_vector_i = json.dumps(vector_i)
                    f.write(f'{json_vector_i}\n')

    print('ALL TASKS DONE')
