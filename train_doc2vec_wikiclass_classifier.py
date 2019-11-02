import torch
import torch.nn as nn
import json
import random
import argparse
import tqdm
import os
from collections import namedtuple
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from wikigen.model.scheduler import Scheduler
from wikigen.data import split_list

torch.manual_seed(2)
torch.cuda.manual_seed(2)
random.seed(2)

Example = namedtuple("Example", ["idx", "vector", "label"])


class BatchIterator(object):
    def __init__(
        self, vocabs, examples, batch_size, batch_builder, shuffle=False
    ):

        self.vocabs = vocabs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.examples = examples
        if self.shuffle:
            self.examples = random.shufle(self.examples)
        self.num_batches = (len(self.examples) + batch_size - 1) // batch_size
        self.batch_builder = batch_builder

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        examples_slice = []
        for i, example in enumerate(self.examples, 1):
            examples_slice.append(example)
            if i > 0 and i % (self.batch_size) == 0:
                yield self.batch_builder(examples_slice, self.vocabs)
                examples_slice = []

        if examples_slice:
            yield self.batch_builder(examples_slice, self.vocabs)

        if self.shuffle:
            self.examples = random.shuffle(self.examples)


class Batch(object):
    def __init__(self, examples, vocabs):
        self.ids_batch = [int(item.idx) for item in examples]
        src_examples = [item.vector for item in examples]
        tgt_examples = [vocabs[item.label] for item in examples]

        self.src_batch = torch.tensor(src_examples)
        self.tgt_batch = torch.tensor(tgt_examples)

    def to_torch_(self, device):
        self.src_batch = self.src_batch.to(device)
        self.tgt_batch = self.tgt_batch.to(device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_test", action="store_true")

    args = parser.parse_args()

    from preprocess_doc2vec_wikiclass import (
        train_label_file_path,
        test_label_file_path,
        train_clean_wikiclass_files_path,
        test_clean_wikiclass_files_path,
        test_doc_file_path,
        train_doc_file_path,
    )

    if args.use_test:

        train_vector_file_path = os.path.join(
            train_clean_wikiclass_files_path, "vectors.jsonl"
        )

        test_vector_file_path = os.path.join(
            test_clean_wikiclass_files_path, "vectors.jsonl"
        )
    else:

        train_vector_file_path = os.path.join(
            train_clean_wikiclass_files_path, "vectors.no_test.jsonl"
        )

        test_vector_file_path = os.path.join(
            test_clean_wikiclass_files_path, "vectors.no_test.jsonl"
        )

    with open(train_vector_file_path) as f:
        train_vectors = [json.loads(line.strip()) for line in f.readlines()]

    with open(test_vector_file_path) as f:
        test_vectors = [json.loads(line.strip()) for line in f.readlines()]

    with open(train_label_file_path) as f:
        train_labels = [line.strip() for line in f.readlines()]

    with open(test_label_file_path) as f:
        test_labels = [line.strip() for line in f.readlines()]

    with open(train_doc_file_path) as f:
        train_docs = [line.strip() for line in f.readlines()]

    with open(test_doc_file_path) as f:
        test_docs = [line.strip() for line in f.readlines()]

    assert len(train_vectors) == len(train_labels)
    assert len(test_vectors) == len(test_labels)

    # In[8]:

    train_lens = [len(item) for item in train_docs]
    print(np.mean(train_lens))

    train_examples = []
    for i in range(len(train_vectors)):
        example = Example(i, train_vectors[i], train_labels[i])
        train_examples.append(example)

    test_examples = []
    for i in range(len(test_vectors)):
        example = Example(i, test_vectors[i], test_labels[i])
        test_examples.append(example)

    train_examples, valid_examples, _ = split_list(
        train_examples, train_ratio=0.8, valid_ratio=0.2, test_ratio=None
    )

    index2label = ["Stub", "Start", "C", "B", "GA", "FA"]
    label2index = {label: i for i, label in enumerate(index2label)}
    batch_size = 100
    input_size = 500
    output_size = len(label2index)
    hidden_units = [2000, 1000, 500, 200]
    decay = 0.5
    patience = 5
    learning_rate = 0.001
    epochs = 100
    min_lr = 1e-5
    device = torch.device("cuda")

    model_save_path = "/tmp/model.pth"

    train_batches = BatchIterator(
        label2index, train_examples, len(train_examples), Batch
    )

    valid_batches = BatchIterator(
        label2index, valid_examples, len(train_examples), Batch
    )

    test_batches = BatchIterator(
        label2index, test_examples, len(test_examples), Batch
    )

    layers = []
    i_num_units = input_size
    for o_num_units in hidden_units:
        layers.append(nn.Linear(i_num_units, o_num_units))
        layers.append(nn.ReLU())
        i_num_units = o_num_units

    layers.append(nn.Linear(o_num_units, output_size))

    model = nn.Sequential(*layers)
    model = model.to(device)
    print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = Scheduler(
        optimizer,
        mode="max",
        factor=decay,
        patience=patience,
        threshold=0.0001,
        threshold_mode="abs",
        min_lr=min_lr,
    )

    stop = False

    pbar = tqdm.trange(epochs)

    for epoch in pbar:

        train_epoch_loss = 0
        i = 0

        train_predictions = []
        train_targets = []

        model.train()

        for train_batch in train_batches:
            train_batch.to_torch_(device)

            optimizer.zero_grad()
            outputs = model(train_batch.src_batch)
            loss = loss_function(outputs, train_batch.tgt_batch)
            loss.backward()

            optimizer.step()

            _, predictions = outputs.max(1)
            train_predictions.append(predictions)
            train_targets.append(train_batch.tgt_batch)

            train_epoch_loss += loss.item()

            i += 1

        train_predictions = torch.cat(train_predictions).cpu().numpy()
        train_targets = torch.cat(train_targets).cpu().numpy()

        train_f1 = f1_score(train_targets, train_predictions, average="macro")

        train_acc = accuracy_score(train_targets, train_predictions)

        valid_epoch_loss = 0
        i = 0

        valid_predictions = []
        valid_targets = []

        model.eval()

        for valid_batch in valid_batches:
            valid_batch.to_torch_(device)

            optimizer.zero_grad()
            outputs = model(valid_batch.src_batch)
            loss = loss_function(outputs, valid_batch.tgt_batch)

            _, predictions = outputs.max(1)
            valid_predictions.append(predictions)
            valid_targets.append(valid_batch.tgt_batch)

            valid_epoch_loss += loss.item()

            i += 1

        valid_predictions = torch.cat(valid_predictions).cpu().numpy()
        valid_targets = torch.cat(valid_targets).cpu().numpy()

        valid_f1 = f1_score(valid_targets, valid_predictions, average="macro")

        valid_acc = accuracy_score(valid_targets, valid_predictions)

        pbar.write(f"Epoch {epoch}")
        pbar.write(f"Train Loss: {train_epoch_loss / len(train_batches)}")
        pbar.write(f"Train Accuracy: {train_acc}")
        pbar.write(f"Train F1: {train_f1}")

        pbar.write(f"Valid Loss: {valid_epoch_loss / len(valid_batches)}")
        pbar.write(f"Valid Accuracy: {valid_acc}")
        pbar.write(f"Valid  F1: {valid_f1}")

        is_best, new_lrs = scheduler.step(valid_acc, epoch)

        if is_best:
            torch.save(model, model_save_path)

        for i, new_lr in enumerate(new_lrs):
            pbar.write(f"Reduced learning rate of group {i} to {new_lr:.4e}.")
            if new_lr <= 1e-4:
                stop = True
        if stop:
            break

    model = torch.load(model_save_path)

    epoch_loss = 0
    i = 0

    test_predictions = []
    test_targets = []

    model.eval()

    for test_batch in test_batches:
        test_batch.to_torch_(device)

        optimizer.zero_grad()
        outputs = model(test_batch.src_batch)
        loss = loss_function(outputs, test_batch.tgt_batch)

        _, predictions = outputs.max(1)
        test_predictions.append(predictions)
        test_targets.append(test_batch.tgt_batch)

        epoch_loss += loss.item()

        i += 1

    test_predictions = torch.cat(test_predictions).cpu().numpy()
    test_targets = torch.cat(test_targets).cpu().numpy()

    test_f1 = f1_score(test_targets, test_predictions, average="macro")

    test_acc = accuracy_score(test_targets, test_predictions)

    print(f"Test Loss: {epoch_loss / len(test_batches)}")
    print(f"Test Accuracy: {test_acc}")
    print(f"Test F1: {test_f1}")

    test_lens = [len(item) for item in test_docs]
    tuples = [
        item
        for item in zip(
            test_lens, test_targets.tolist(), test_predictions.tolist()
        )
    ]

    sorted_tuples = sorted(tuples, key=lambda x: x[0])

    chunk_size = 100

    for i in range(chunk_size, len(sorted_tuples), chunk_size):

        sorted_tuples_i = sorted_tuples[i : i + chunk_size]
        lens_i, targets_i, predictions_i = zip(*sorted_tuples_i)
        acc_i = accuracy_score(targets_i, predictions_i)
        mean_len_i = round(sum(lens_i) / len(lens_i), 1)
        print(mean_len_i, acc_i)
