
import json
import argparse
import os

import torch
import tqdm
import numpy as np

from wikigen.settings import CLASS_LABELS
from wikigen.data import Dataset
from wikigen.config import AttrDict
from wikigen.logger import Logger
from wikigen.model.batch import ClassifierBatch, BatchIterator
from wikigen.main.classifier import generate_output

from wikigen.settings import RESULTS_PATH
from sklearn.metrics import (
    f1_score, accuracy_score, multilabel_confusion_matrix)

parser = argparse.ArgumentParser(
    description='test_classifier.py')

parser.add_argument(
    '--model',
    help="Path to load model checkpoint")

parser.add_argument(
    '--output',
    default='output.json')

run_args = parser.parse_args()

if not os.path.isdir(run_args.model):
    raise(IOError('Please provide the folder'
                  ' where the trained model is.'))

model_save_path = os.path.join(
    run_args.model, 'model.pth.best')

hyperparams_save_path = os.path.join(
    run_args.model, 'hyperparams.json')
output_save_path = run_args.output

with open(hyperparams_save_path) as f:
    args = AttrDict(json.load(f))

model_id = args.hash

logger = Logger(
    hash_value=model_id,
    write_mode='DATABASE',
    args=args)

device = torch.device(args.device)
if args.device == 'cpu':
    torch.manual_seed(args.seed)
else:
    torch.cuda.manual_seed(args.seed)

dataset = Dataset(
    RESULTS_PATH,
    args.dataset,
    args.src_min_freq,
    src_min_freq=args.src_min_freq,
    joint=False,  # fixing as it is not required
    max_len=args.max_len,
    force_reload=False,
    lowercase=args.lowercase,
    use_temporal_splits=args.use_temporal_splits)

src_ignore_ids = [
    dataset.vocab.src.PAD.hash,
    dataset.vocab.src.BOS.hash,
    dataset.vocab.src.EOS.hash]

src_tag_ignore_ids = None
if dataset.vocab.src_tag:
    src_tag_ignore_ids = [
        dataset.vocab.src_tag.PAD.hash,
        dataset.vocab.src_tag.BOS.hash,
        dataset.vocab.src_tag.EOS.hash]


valid_data = [
    example for example in
    dataset.valid]

test_data =  [
    example for example in
    dataset.test]


# loading best model on valid
model = torch.load(model_save_path).to(device=device)

print(model)

window_size = 300
n = 1

while n*window_size <= (len(test_data) + window_size):


    batch_i = 0
    total_test_loss = 0
    gen_probs = []
    src_sequences = []
    src_tag_sequences = []
    tgt_probs = []
    input_ids = []

    eval_data = valid_data + test_data[:n * window_size]

    test_batches = BatchIterator(
        dataset.vocab,
        eval_data,
        2,
        ClassifierBatch,
        max_len=args.max_len)


    #for batch in tqdm.tqdm(test_batches, desc='Testing... '):

    for batch in test_batches:
        model.eval()

        src_batch_sequences = batch.src_batch.sequences
        src_batch_tag_sequences = batch.src_batch.tag_sequences

        batch.to_torch(device=device)

        loss, probs = model.forward(
            batch.src_batch,
            batch.tgt_batch,
            src_message_batch_tuple=batch.src_message_batch)

        total_test_loss += loss.item()

        for id_sequence in src_batch_sequences:
            src_sequence_i = dataset.vocab.src.indices2tokens(
                id_sequence, ignore_ids=src_ignore_ids)
            src_sequences.append(' '.join(src_sequence_i))

        if src_batch_tag_sequences is not None:
            for id_sequence in src_batch_tag_sequences:
                src_tag_sequence_i = dataset.vocab.src_tag.indices2tokens(
                        id_sequence, ignore_ids=src_tag_ignore_ids)
                src_tag_sequences.append(' '.join(src_tag_sequence_i))

        for class_probs_i in probs:
            gen_probs.append(class_probs_i.tolist())

        for tgt_probs_i in batch.tgt_batch.classes:
            tgt_probs.append(tgt_probs_i.tolist())

        input_ids += batch.ids_batch
        batch_i += 1

        total_test_loss /= batch_i


    output = generate_output(
        input_ids,
        src_sequences,
        gen_probs=gen_probs,
        tgt_probs=tgt_probs,
        src_tag_sequences=src_tag_sequences)

    target_classes = np.array(
        [item['tgt_probs'] for item in output])

    pred_classes = np.array(
        [item["gen_probs"] for item in output])


    test_f1 = f1_score(
        target_classes.argmax(1),
        pred_classes.argmax(1),
        average='macro')

    test_acc = accuracy_score(
        target_classes.argmax(1),
        pred_classes.argmax(1))

    test_conf_matrix = multilabel_confusion_matrix(
        target_classes.argmax(1),
        pred_classes.argmax(1))


    test_bleu = 0
    test_meteor = 0
    test_meant = 0

    datadict = {
        'Test/Loss': f'{0.0:.4f}',
        'Test/Class_Loss': f'{total_test_loss:.4f}',
        'Test/Class_F1': f'{test_f1:.4f}',
        'Test/Class_Acc': f'{test_acc:.4f}',
        'Test/BLEU': f'{test_bleu:.4f}',
        'Test/METEOR': f'{test_meteor:.4f}',
        'Test/MEANT': f'{test_meant:.4f}'}

    print(f'---------------------{n}----------------------')

    for key, value in datadict.items():
        print(f'{key}: {value}')

    # if test_conf_matrix is not None:
    #     for i, item in enumerate(test_conf_matrix):
    #         print(CLASS_LABELS[i])
    #         print(item)
    #         print('')

    n += 1

    print('')