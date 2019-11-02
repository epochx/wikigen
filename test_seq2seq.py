
import json
import os
import torch
import tqdm
import numpy as np
import argparse
from itertools import chain

from wikigen.settings import CLASS_LABELS
from wikigen.data import Dataset
from wikigen.config import AttrDict
from wikigen.model.beam import beam_search_decode
from wikigen.model.batch import BatchIterator, Seq2SeqBatch
from wikigen.main.seq2seq import generate_output

from wikigen.eval.bleu import bleu
# from ..eval.meteor import meteor
# from ..eval.meant import meant

from wikigen.logger import Logger
from wikigen.settings import RESULTS_PATH
from sklearn.metrics import (
    f1_score, accuracy_score, multilabel_confusion_matrix)

parser = argparse.ArgumentParser(
    description='test_seq2seq.py')

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

print(args)

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
    args.tgt_min_freq,
    src_min_freq=args.src_min_freq,
    joint=args.joint,
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

tgt_ignore_ids = [
    dataset.vocab.tgt.PAD.hash,
    dataset.vocab.tgt.BOS.hash,
    dataset.vocab.tgt.EOS.hash]

# loading best model on valid
model = torch.load(model_save_path).to(device=device)

print(model)

lamb = args.class_lambda

valid_data = [
    example for example in
    dataset.valid]

test_data =  [
    example for example in
    dataset.test]

window_size = 300
n = 1

while n*window_size <= (len(test_data) + window_size):

    batch_i = 0
    total_test_loss = 0
    total_test_class_loss = 0
    gen_sequences = []
    gen_probs = []
    src_sequences = []
    src_tag_sequences = []
    tgt_sequences = []
    tgt_probs = []
    input_ids = []
    attentions = []

    eval_data = valid_data #+ test_data[:n*window_size]

    #print(len(eval_data))

    test_batches = BatchIterator(
        dataset.vocab,
        eval_data,
        2,
        Seq2SeqBatch,
        max_len=args.max_len)


#    for batch in tqdm.tqdm(test_batches, desc='Testing... '):

    for batch in test_batches:
        model.eval()

        src_batch_sequences = batch.src_batch.sequences
        src_batch_lengths = batch.src_batch.lengths
        src_batch_tag_sequences = batch.src_batch.tag_sequences
        tgt_batch_sequences = batch.tgt_batch.sequences[:, 1:]

        batch.to_torch(device=device)

        (loss,
         predictions,
         class_loss,
         class_probs) = model.forward(
            batch.src_batch,
            batch.tgt_batch)

        if args.num_classes is not None:
            if lamb is not None:
                loss = (1 - lamb) * loss + lamb * class_loss
            else:
                loss += class_loss
            total_test_class_loss += class_loss.item()

        total_test_loss += loss.item()

        (test_predictions,
         test_attentions) = beam_search_decode(
            model,
            dataset.vocab,
            batch.src_batch,
            args.max_len,
            args.beam_size)

        for id_sequence in test_predictions:
            gen_sequence_i = dataset.vocab.tgt.indices2tokens(
                id_sequence[0], ignore_ids=tgt_ignore_ids)
            gen_sequences.append(' '.join(gen_sequence_i))

        for id_sequence in src_batch_sequences:
            src_sequence_i = dataset.vocab.src.indices2tokens(
                id_sequence, ignore_ids=src_ignore_ids)
            src_sequences.append(' '.join(src_sequence_i))

        for i, test_attention in enumerate(test_attentions):
            len_i = src_batch_lengths[i]
            test_attention_i = test_attention[0]
            test_attention_i = test_attention_i[:, :len_i].t().tolist()
            attentions.append(test_attention_i)

        if src_batch_tag_sequences is not None:
            for id_sequence in src_batch_tag_sequences:
                src_tag_sequence_i = dataset.vocab.src_tag.indices2tokens(
                    id_sequence, ignore_ids=src_tag_ignore_ids)
                src_tag_sequences.append(' '.join(src_tag_sequence_i))

        for id_sequence in tgt_batch_sequences:
            tgt_sequence_i = dataset.vocab.tgt.indices2tokens(
                id_sequence, ignore_ids=tgt_ignore_ids)
            tgt_sequences.append(' '.join(tgt_sequence_i))

        if args.num_classes is not None:
            for class_probs_i in class_probs:
                gen_probs.append(class_probs_i.tolist())

            for tgt_probs_i in batch.tgt_batch.classes:
                tgt_probs.append(tgt_probs_i.tolist())

        input_ids += batch.ids_batch
        batch_i += 1

    total_test_loss /= batch_i

    output = generate_output(
        input_ids,
        src_sequences,
        tgt_sequences,
        gen_sequences,
        gen_probs=gen_probs,
        tgt_probs=tgt_probs,
        src_tag_sequences=src_tag_sequences,
        attentions=attentions)

    gold_sequences = {
        item['id']: item['tgt'] for item in output}

    pred_sequences = {
        item['id']: item['gen'] for item in output}

    test_class_f1 = 0
    test_class_acc = 0
    test_conf_matrix = None
    if args.num_classes is not None:
        total_test_class_loss /= batch_i

        target_classes = np.array(
            [item['tgt_probs'] for item in output])

        pred_classes = np.array(
            [item["gen_probs"] for item in output])

        import ipdb; ipdb.set_trace(context=10)

        test_class_f1 = f1_score(
            target_classes.argmax(1),
            pred_classes.argmax(1),
            average='macro')

        test_class_acc = accuracy_score(
            target_classes.argmax(1),
            pred_classes.argmax(1))

        print((target_classes.argmax(1) == pred_classes.argmax(1)).sum())
        print(target_classes.shape[0])

        test_conf_matrix = multilabel_confusion_matrix(
            target_classes.argmax(1),
            pred_classes.argmax(1))


    test_bleu = bleu(gold_sequences, pred_sequences)

    # test_meant = meant(gold_sequences, pred_sequences)['fscore']
    # test_meteor = meteor(gold_sequences, pred_sequences)

    test_meteor = 0
    test_meant = 0

    datadict = {
        'Test/Loss': f'{total_test_loss:.4}',
        'Test/Class_Loss': f'{total_test_class_loss:.4f}',
        'Test/Class_F1': f'{test_class_f1:.4f}',
        'Test/Class_Acc': f'{test_class_acc:.4f}',
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