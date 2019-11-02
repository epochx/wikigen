
import json
import os
import subprocess

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import tqdm
import numpy as np

from ..data import Dataset
from ..logger import Logger

from ..model.optimizers import optimizers
from ..model.scheduler import Scheduler
from ..model.seq2seq import Seq2seq
from ..model.beam import beam_search_decode
from ..model.batch import BatchIterator, Seq2SeqBatch

from ..eval.bleu import bleu
from sklearn.metrics import f1_score, accuracy_score


def generate_output(input_ids,
                    src_sequences,
                    tgt_sequences,
                    gen_sequences,
                    tgt_probs=None,
                    gen_probs=None,
                    src_tag_sequences=None,
                    attentions=None,
                    p_gens=None):

    length = len(input_ids)
    output = []
    for i in range(length):

        output_i = {
            'id': input_ids[i],
            'src': src_sequences[i],
            'tgt': tgt_sequences[i],
            'gen': gen_sequences[i]}

        if gen_probs:
            output_i['gen_probs'] = gen_probs[i]

        if tgt_probs:
            output_i['tgt_probs'] = tgt_probs[i]

        if src_tag_sequences:
            output_i['src_tags'] = src_tag_sequences[i]

        if attentions:
            output_i['attention'] = attentions[i]

        if p_gens:
            output_i['p_gen'] = p_gens[i]

        output.append(output_i)

    return output


def main(args):

    device = torch.device(args.device)
    if args.device == 'cpu':
        torch.manual_seed(args.seed)
    else:
        torch.cuda.manual_seed(args.seed)

    dataset = Dataset(
        args.results_path,
        args.dataset,
        args.tgt_min_freq,
        src_min_freq=args.src_min_freq,
        joint=args.joint,
        max_len=args.max_len,
        force_reload=args.force_dataset_reload,
        lowercase=args.lowercase)

    logger = Logger(
        args,
        model_name='Seq2Seq',
        write_mode=args.write_mode)

    model_id = logger.hash
    results_path = logger.run_savepath

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

    train_batches = BatchIterator(
        dataset.vocab, dataset.train, args.batch_size,
        Seq2SeqBatch, max_len=args.max_len)

    valid_batches = BatchIterator(
        dataset.vocab, dataset.valid, args.test_batch_size,
        Seq2SeqBatch, max_len=args.max_len)

    test_batches = BatchIterator(
        dataset.vocab, dataset.test, 2,
        Seq2SeqBatch, max_len=args.max_len)

    weights = [1]*len(dataset.vocab.tgt)
    weights[dataset.vocab.tgt.PAD.hash] = 0
    weights = torch.tensor(weights, dtype=torch.float)

    encoder_embeddings = nn.Embedding(
        len(dataset.vocab.src),
        args.encoder_embedding_size,
        padding_idx=dataset.vocab.src.PAD.hash)

    if args.joint:
        decoder_embeddings = encoder_embeddings
    else:
        decoder_embeddings = nn.Embedding(
            len(dataset.vocab.tgt),
            args.decoder_embedding_size,
            padding_idx=dataset.vocab.tgt.PAD.hash)

    if args.train_encoder_embeddings:
        print('Setting encoder_embeddings as trainable parameters')
        for param in encoder_embeddings.parameters():
            param.requires_grad_(True)
    else:
        print('Setting encoder_embeddings as non-trainable parameters')
        for param in encoder_embeddings.parameters():
            param.requires_grad_(False)

    encoder_tag_embeddings = None
    if '_meta' in dataset.name and args.use_tag_embeddings:
        encoder_tag_embeddings = nn.Embedding(
            len(dataset.vocab.src_tag),
            args.encoder_tag_embedding_size,
            padding_idx=dataset.vocab.src_tag.PAD.hash)

    model = Seq2seq(
        encoder_embeddings,
        decoder_embeddings,
        weights,
        args.encoder,
        args.decoder_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size,
        attention=args.attention,
        aggregator=args.aggregator,
        teacher_forcing_p=args.teacher_forcing_p,
        encoder_num_layers=args.encoder_num_layers,
        encoder_input_dropout=args.encoder_input_dropout,
        encoder_dropout=args.encoder_dropout,
        encoder_output_dropout=args.encoder_output_dropout,
        decoder_input_dropout=args.decoder_input_dropout,
        decoder_output_dropout=args.decoder_output_dropout,
        bidirectional=args.bidirectional,
        num_classes=args.num_classes,
        tag_embeddings=encoder_tag_embeddings
    ).to(device=device)

    model_save_path = os.path.join(
        results_path, 'model.pth')

    state_dict_save_path = os.path.join(
        results_path, 'state_dict.pth')

    train_output_save_path = os.path.join(
        results_path, 'train.output.json')

    valid_output_save_path = os.path.join(
        results_path, 'valid.output.json')

    test_output_save_path = os.path.join(
        results_path, 'test.output.json')

    tmp_train_output_save_path = '/tmp/train.output.json'
    tmp_valid_output_save_path = '/tmp/valid.output.json'

    writer = SummaryWriter(results_path)

    dic_args = vars(args)
    header = 'parameter|value\n - | -\n'
    parameters_string = header + '\n'.join(
        [f'{key}|{value}' for key, value in dic_args.items()])
    writer.add_text('wikigen/parameters', parameters_string, 0)

    # ---------------------------- TRAIN -----------------------------------------------

    def save_best():

        subprocess.call(
            ['cp', state_dict_save_path,
             state_dict_save_path + ".best"])

        subprocess.call(
            ['cp', model_save_path,
             model_save_path + ".best"])

        subprocess.call(
            ['cp', valid_output_save_path,
             valid_output_save_path + ".best"])

    lamb = args.class_lambda

    parameters = [
        p for p in model.parameters()
        if p.requires_grad]

    Optimizer = optimizers[args.optim]
    optimizer = Optimizer(parameters, args.lr)

    scheduler = Scheduler(
        optimizer,
        mode='min' if args.metric == 'loss' else 'max',
        factor=args.decay,
        patience=args.patience,
        threshold=0.0001,
        threshold_mode='abs',
        min_lr=1e-04)

    stop = False
    best_metric = float('inf') if args.metric == 'loss' else 0.0

    print(model)

    with tqdm.trange(args.epochs, desc=model_id) as pbar:

        for epoch in pbar:

            total_train_loss = 0
            total_train_class_loss = 0
            batch_i = 0
            gen_sequences = []
            gen_probs = []
            src_sequences = []
            src_tag_sequences = []
            tgt_probs = []
            tgt_sequences = []
            input_ids = []

            for batch in tqdm.tqdm(train_batches, desc='Training... '):

                model.train()
                model.zero_grad()

                src_batch_sequences = batch.src_batch.sequences
                src_batch_tag_sequences = batch.src_batch.tag_sequences
                tgt_batch_sequences = batch.tgt_batch.sequences[:, 1:]

                try:

                    batch.to_torch(device=device)

                    (loss,
                     predictions,
                     class_loss,
                     class_probs) = model.forward(
                        batch.src_batch,
                        batch.tgt_batch)

                    if args.num_classes is not None:
                        if lamb is not None:
                            loss = (1-lamb)*loss + lamb*class_loss
                        else:
                            loss += class_loss
                        total_train_class_loss += class_loss.item()

                    total_train_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip)
                    optimizer.step()

                except RuntimeError as e:
                    # catch out of memory exceptions during fwd/bck (skip batch)
                    if 'out of memory' in str(e):
                        pbar.write(
                            '| WARNING: ran out of memory, skipping batch. '
                            'if this happens frequently, decrease batch_size or '
                            'truncate the inputs to the model.')
                        continue
                    else:
                        raise e

                for id_sequence in predictions:
                    gen_sequence_i = dataset.vocab.tgt.indices2tokens(
                        id_sequence, ignore_ids=tgt_ignore_ids)
                    gen_sequences.append(' '.join(gen_sequence_i))

                for id_sequence in src_batch_sequences:
                    src_sequence_i = dataset.vocab.src.indices2tokens(
                        id_sequence, ignore_ids=src_ignore_ids)
                    src_sequences.append(' '.join(src_sequence_i))

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

            total_train_loss /= batch_i

            if args.device == 'cuda':
                torch.cuda.empty_cache()

            output = generate_output(
                input_ids,
                src_sequences,
                tgt_sequences,
                gen_sequences,
                tgt_probs=tgt_probs,
                gen_probs=gen_probs,
                src_tag_sequences=src_tag_sequences)

            with open(train_output_save_path, 'w') as f:
                json.dump(output, f)
            with open(tmp_train_output_save_path, 'w') as f:
                json.dump(output, f)

            gold_sequences = {
                item['id']: item["tgt"] for item in output}

            pred_sequences = {
                item['id']: item["gen"] for item in output}

            train_class_f1 = 0
            train_class_acc = 0

            if args.num_classes is not None:
                total_train_class_loss /= batch_i

                target_classes = np.array(
                    [item['tgt_probs'] for item in output])

                pred_classes = np.array(
                    [item["gen_probs"] for item in output])

                train_class_f1 = f1_score(
                    target_classes.argmax(1),
                    pred_classes.argmax(1),
                    average='macro')

                train_class_acc = accuracy_score(
                    target_classes.argmax(1),
                    pred_classes.argmax(1))

            train_bleu = bleu(gold_sequences, pred_sequences)

            # ------------------------- VALID ------------------------------------------------

            batch_i = 0
            total_valid_loss = 0
            total_valid_class_loss = 0
            gen_sequences = []
            gen_probs = []
            src_sequences = []
            src_tag_sequences = []
            tgt_sequences = []
            tgt_probs = []
            input_ids = []
            attentions = []

            for batch in tqdm.tqdm(valid_batches, desc='Validation... '):

                model.eval()

                src_batch_sequences = batch.src_batch.sequences
                src_batch_lengths = batch.src_batch.lengths
                src_batch_tag_sequences = batch.src_batch.tag_sequences
                tgt_batch_sequences = batch.tgt_batch.sequences[:, 1:]

                try:
                    batch.to_torch(device=device)

                    (loss,
                     predictions,
                     class_loss,
                     class_probs) = model.forward(
                        batch.src_batch, batch.tgt_batch)

                    if args.num_classes is not None:
                        if lamb is not None:
                            loss = (1 - lamb) * loss + lamb * class_loss
                        else:
                            loss += class_loss
                        total_valid_class_loss += class_loss.item()

                    total_valid_loss += loss.item()

                    (valid_predictions,
                     valid_attentions) = beam_search_decode(
                        model,
                        dataset.vocab,
                        batch.src_batch,
                        args.max_len,
                        args.beam_size)

                except RuntimeError as e:
                    # catch out of memory exceptions during fwd/bck (skip batch)
                    if 'out of memory' in str(e):
                        pbar.write(
                            '| WARNING: ran out of memory, skipping batch. '
                            'if this happens frequently, decrease test_batch_size or '
                            'truncate the inputs to the model.')
                        continue
                    else:
                        raise e

                for id_sequence in valid_predictions:
                    gen_sequence_i = dataset.vocab.tgt.indices2tokens(
                        id_sequence[0], ignore_ids=tgt_ignore_ids)
                    gen_sequences.append(' '.join(gen_sequence_i))

                for i, valid_attention in enumerate(valid_attentions):
                    len_i = src_batch_lengths[i]
                    valid_attention_i = valid_attention[0]
                    valid_attention_i = valid_attention_i[:,:len_i].t().tolist()
                    attentions.append(valid_attention_i)

                for id_sequence in src_batch_sequences:
                    src_sequence_i = dataset.vocab.src.indices2tokens(
                        id_sequence, ignore_ids=src_ignore_ids)
                    src_sequences.append(' '.join(src_sequence_i))

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

            total_valid_loss /= batch_i

            if args.device == 'cuda':
                torch.cuda.empty_cache()

            output = generate_output(
                input_ids,
                src_sequences,
                tgt_sequences,
                gen_sequences,
                gen_probs=gen_probs,
                tgt_probs=tgt_probs,
                src_tag_sequences=src_tag_sequences,
                attentions=attentions)

            with open(valid_output_save_path, 'w') as f:
                json.dump(output, f)
            with open(tmp_valid_output_save_path, 'w') as f:
                json.dump(output, f)

            gold_sequences = {
                item['id']: item["tgt"] for item in output}

            pred_sequences = {
                item['id']: item["gen"] for item in output}

            valid_class_f1 = 0
            valid_class_acc = 0

            if args.num_classes is not None:
                total_valid_class_loss /= batch_i

                target_classes = np.array(
                    [item['tgt_probs'] for item in output])

                pred_classes = np.array(
                    [item["gen_probs"] for item in output])

                valid_class_f1 = f1_score(
                    target_classes.argmax(1),
                    pred_classes.argmax(1),
                    average='macro')

                valid_class_acc = accuracy_score(
                    target_classes.argmax(1),
                    pred_classes.argmax(1))

            valid_bleu = bleu(gold_sequences, pred_sequences)

            current_log = {
                'Epoch': epoch,
                'Train/Loss': total_train_loss,
                'Train/Class_Loss': total_train_class_loss,
                'Train/Class_F1': train_class_f1,
                'Train/Class_Acc': train_class_acc,
                'Train/BLEU': train_bleu,
                'Valid/Loss': total_valid_loss,
                'Valid/Class_Loss': total_valid_class_loss,
                'Valid/Class_F1': valid_class_f1,
                'Valid/Class_Acc': valid_class_acc,
                'Valid/BLEU': valid_bleu}

            pbar.write('Epoch {} '.format(epoch) + '#' * 22)

            # logging tensorboard
            for key, value in current_log.items():
                if key != 'Epoch':
                    writer.add_scalar(f'wikigen/{key}', float(value), epoch)
                    pbar.write(f'{key}: {value:14.3f}')
            pbar.write('\n')

            logger.update_results(current_log)

            torch.save(
                model, os.path.join(results_path, 'model.pth'))
            torch.save(
                model.state_dict(), state_dict_save_path)

            if args.metric == 'bleu':
                metric = current_log['Valid/BLEU']

            elif args.metric == 'loss':
                metric = current_log['Valid/Loss']

            elif args.metric == 'acc':
                metric = current_log['Valid/Class_Acc']

            elif args.metric == 'f1':
                metric = current_log['Valid/Class_F1']

            else:
                raise NotImplementedError

            is_best, new_lrs = scheduler.step(metric, epoch)

            if is_best:
                best_metric = metric
                datadict = {
                    'best_metric': f'{best_metric:.4f}_{epoch}'}
                logger.update_results(datadict)
                save_best()
            else:
                pass

            for i, new_lr in enumerate(new_lrs):
                pbar.write(f'Reduced learning rate of group {i} to {new_lr:.4e}.')
                if new_lr <= 1e-4:
                    stop = True
            if stop:
                break

    # --------------------------- TEST ------------------------------------------------------

    del model

    if args.device == 'cuda':
        torch.cuda.empty_cache()

    # loading best model on valid
    model = torch.load(
        model_save_path + ".best").to(device=device)

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

    for batch in tqdm.tqdm(test_batches, desc='Testing... '):

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

    with open(test_output_save_path, 'w') as f:
        json.dump(output, f)

    gold_sequences = {
        item['id']: item['tgt'] for item in output}

    pred_sequences = {
        item['id']: item['gen'] for item in output}

    test_class_f1 = 0
    test_class_acc = 0
    if args.num_classes is not None:

        total_test_class_loss /= batch_i

        target_classes = np.array(
            [item['tgt_probs'] for item in output])

        pred_classes = np.array(
            [item["gen_probs"] for item in output])

        test_class_f1 = f1_score(
            target_classes.argmax(1),
            pred_classes.argmax(1),
            average='macro')

        test_class_acc = accuracy_score(
            target_classes.argmax(1),
            pred_classes.argmax(1))

    test_bleu = bleu(gold_sequences, pred_sequences)

    datadict = {
        'Test/Loss': f'{total_test_loss:.4}',
        'Test/Class_Loss': f'{total_test_class_loss:.4f}',
        'Test/Class_F1': f'{test_class_f1:.4f}',
        'Test/Class_Acc': f'{test_class_acc:.4f}',
        'Test/BLEU': f'{test_bleu:.4f}'}

    logger.update_results(datadict)
