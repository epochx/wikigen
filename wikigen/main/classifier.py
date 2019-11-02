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
from ..model.classifier import Classifier
from ..model.scheduler import Scheduler
from ..model.batch import ClassifierBatch, BatchIterator
from sklearn.metrics import f1_score, accuracy_score


def generate_output(
    input_ids,
    src_sequences,
    tgt_probs=None,
    gen_probs=None,
    src_tag_sequences=None,
):

    length = len(input_ids)
    output = []
    for i in range(length):

        output_i = {"id": input_ids[i]}

        if src_sequences:
            output_i["src"] = src_sequences[i]

        if gen_probs:
            output_i["gen_probs"] = gen_probs[i]

        if tgt_probs:
            output_i["tgt_probs"] = tgt_probs[i]

        if src_tag_sequences:
            output_i["src_tags"] = src_tag_sequences[i]

        output.append(output_i)

    return output


def main(args):

    device = torch.device(args.device)
    if args.device == "cpu":
        torch.manual_seed(args.seed)
    else:
        torch.cuda.manual_seed(args.seed)

    dataset = Dataset(
        args.results_path,
        args.dataset,
        args.src_min_freq,
        src_min_freq=args.src_min_freq,
        joint=False,
        max_len=args.max_len,
        force_reload=args.force_dataset_reload,
        lowercase=args.lowercase,
    )

    logger = Logger(args, model_name="Classifier", write_mode=args.write_mode)

    model_id = logger.hash
    results_path = logger.run_savepath

    src_ignore_ids = [
        dataset.vocab.src.PAD.hash,
        dataset.vocab.src.BOS.hash,
        dataset.vocab.src.EOS.hash,
    ]

    src_tag_ignore_ids = None
    if dataset.vocab.src_tag:
        src_tag_ignore_ids = [
            dataset.vocab.src_tag.PAD.hash,
            dataset.vocab.src_tag.BOS.hash,
            dataset.vocab.src_tag.EOS.hash,
        ]

    train_batches = BatchIterator(
        dataset.vocab,
        dataset.train,
        args.batch_size,
        ClassifierBatch,
        max_len=args.max_len,
    )

    valid_batches = BatchIterator(
        dataset.vocab,
        dataset.valid,
        args.test_batch_size,
        ClassifierBatch,
        max_len=args.max_len,
    )

    test_batches = BatchIterator(
        dataset.vocab, dataset.test, 2, ClassifierBatch, max_len=args.max_len
    )

    weights = [1] * len(dataset.vocab.tgt)
    weights[dataset.vocab.tgt.PAD.hash] = 0

    encoder_embeddings = nn.Embedding(
        len(dataset.vocab.src),
        args.encoder_embedding_size,
        padding_idx=dataset.vocab.src.PAD.hash,
    )

    if args.train_encoder_embeddings:
        print("Setting encoder_embeddings as trainable parameters")
        for param in encoder_embeddings.parameters():
            param.requires_grad_(True)
    else:
        print("Setting encoder_embeddings as non-trainable parameters")
        for param in encoder_embeddings.parameters():
            param.requires_grad_(False)

    encoder_tag_embeddings = None
    if "_meta" in dataset.name and args.use_tag_embeddings:
        encoder_tag_embeddings = nn.Embedding(
            len(dataset.vocab.src_tag),
            args.encoder_tag_embedding_size,
            padding_idx=dataset.vocab.src_tag.PAD.hash,
        )

    encoder_message_embeddings = None
    if args.use_message_embeddings:
        encoder_message_embeddings = nn.Embedding(
            len(dataset.vocab.tgt),
            args.encoder_message_embedding_size,
            padding_idx=dataset.vocab.tgt.PAD.hash,
        )

    model = Classifier(
        encoder_embeddings,
        args.encoder_hidden_size,
        args.num_classes,
        aggregator=args.aggregator,
        encoder_num_layers=args.encoder_num_layers,
        encoder_input_dropout=args.encoder_input_dropout,
        encoder_dropout=args.encoder_dropout,
        encoder_output_dropout=args.encoder_output_dropout,
        bidirectional=args.bidirectional,
        tag_embeddings=encoder_tag_embeddings,
        message_embeddings=encoder_message_embeddings,
        message_encoder_hidden_size=args.encoder_message_hidden_size,
    ).to(device=device)

    print(model)

    model_save_path = os.path.join(results_path, "model.pth")

    state_dict_save_path = os.path.join(results_path, "state_dict.pth")

    train_output_save_path = os.path.join(results_path, "train.output.json")

    valid_output_save_path = os.path.join(results_path, "valid.output.json")

    test_output_save_path = os.path.join(results_path, "test.output.json")

    tmp_train_output_save_path = "/tmp/train.output.json"
    tmp_valid_output_save_path = "/tmp/valid.output.json"

    writer = SummaryWriter(results_path)

    dic_args = vars(args)
    header = "parameter|value\n - | -\n"
    parameters_string = header + "\n".join(
        [f"{key}|{value}" for key, value in dic_args.items()]
    )
    writer.add_text("wikigen/parameters", parameters_string, 0)

    # ---------------------------- TRAIN -----------------------------------------------

    def save_best():
        subprocess.call(
            ["cp", state_dict_save_path, state_dict_save_path + ".best"]
        )
        subprocess.call(["cp", model_save_path, model_save_path + ".best"])
        subprocess.call(
            ["cp", valid_output_save_path, valid_output_save_path + ".best"]
        )

    parameters = [p for p in model.parameters() if p.requires_grad]

    Optimizer = optimizers[args.optim]
    optimizer = Optimizer(parameters, args.lr)

    scheduler = Scheduler(
        optimizer,
        mode="min" if args.metric == "loss" else "max",
        factor=args.decay,
        patience=args.patience,
        threshold=0.0001,
        threshold_mode="abs",
        min_lr=1e-04,
    )

    stop = False
    best_metric = 0.0

    print(model)

    with tqdm.trange(args.epochs, desc=model_id) as pbar:

        for epoch in pbar:

            total_train_loss = 0
            batch_i = 0
            gen_probs = []
            src_sequences = []
            src_tag_sequences = []
            tgt_probs = []
            input_ids = []

            for batch in tqdm.tqdm(train_batches, desc="Training... "):

                model.train()
                model.zero_grad()

                src_batch_sequences = batch.src_batch.sequences
                src_batch_tag_sequences = batch.src_batch.tag_sequences

                try:

                    batch.to_torch(device=device)

                    loss, probs = model.forward(
                        batch.src_batch,
                        batch.tgt_batch,
                        src_message_batch_tuple=batch.src_message_batch,
                    )

                    total_train_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()

                except RuntimeError as e:
                    # catch out of memory exceptions during fwd/bck (skip batch)
                    if "out of memory" in str(e):
                        pbar.write(
                            "| WARNING: ran out of memory, skipping batch. "
                            "if this happens frequently, decrease batch_size or "
                            "truncate the inputs to the model."
                        )
                        continue
                    else:
                        raise e

                for id_sequence in src_batch_sequences:
                    src_sequence_i = dataset.vocab.src.indices2tokens(
                        id_sequence, ignore_ids=src_ignore_ids
                    )
                    src_sequences.append(" ".join(src_sequence_i))

                if src_batch_tag_sequences is not None:
                    for id_sequence in src_batch_tag_sequences:
                        src_tag_sequence_i = dataset.vocab.src_tag.indices2tokens(
                            id_sequence, ignore_ids=src_tag_ignore_ids
                        )
                        src_tag_sequences.append(" ".join(src_tag_sequence_i))

                for class_probs_i in probs:
                    gen_probs.append(class_probs_i.tolist())

                for tgt_probs_i in batch.tgt_batch.classes:
                    tgt_probs.append(tgt_probs_i.tolist())

                input_ids += batch.ids_batch
                batch_i += 1

            if args.device == "cuda":
                torch.cuda.empty_cache()

            output = generate_output(
                input_ids,
                src_sequences,
                tgt_probs=tgt_probs,
                gen_probs=gen_probs,
                src_tag_sequences=src_tag_sequences,
            )

            with open(train_output_save_path, "w") as f:
                json.dump(output, f)
            with open(tmp_train_output_save_path, "w") as f:
                json.dump(output, f)

            total_train_loss = 1.0 * total_train_loss / batch_i

            target_classes = np.array([item["tgt_probs"] for item in output])
            pred_classes = np.array([item["gen_probs"] for item in output])

            train_f1 = f1_score(
                target_classes.argmax(1), pred_classes.argmax(1), average="macro"
            )

            train_acc = accuracy_score(
                target_classes.argmax(1), pred_classes.argmax(1)
            )

            # ------------------------- VALID ------------------------------------------------

            batch_i = 0
            total_valid_loss = 0
            gen_probs = []
            src_sequences = []
            src_tag_sequences = []
            tgt_probs = []
            input_ids = []

            for batch in tqdm.tqdm(valid_batches, desc="Validation... "):

                model.eval()

                src_batch_sequences = batch.src_batch.sequences
                src_batch_tag_sequences = batch.src_batch.tag_sequences

                try:
                    batch.to_torch(device=device)

                    loss, probs = model.forward(
                        batch.src_batch,
                        batch.tgt_batch,
                        src_message_batch_tuple=batch.src_message_batch,
                    )

                    total_valid_loss += loss.item()

                except RuntimeError as e:
                    # catch out of memory exceptions during fwd/bck (skip batch)
                    if "out of memory" in str(e):
                        pbar.write(
                            "| WARNING: ran out of memory, skipping batch. "
                            "if this happens frequently, decrease test_batch_size or "
                            "truncate the inputs to the model."
                        )
                        continue
                    else:
                        raise e

                for id_sequence in src_batch_sequences:
                    src_sequence_i = dataset.vocab.src.indices2tokens(
                        id_sequence, ignore_ids=src_ignore_ids
                    )
                    src_sequences.append(" ".join(src_sequence_i))

                if src_batch_tag_sequences is not None:
                    for id_sequence in src_batch_tag_sequences:
                        src_tag_sequence_i = dataset.vocab.src_tag.indices2tokens(
                            id_sequence, ignore_ids=src_tag_ignore_ids
                        )
                        src_tag_sequences.append(" ".join(src_tag_sequence_i))

                for class_probs_i in probs:
                    gen_probs.append(class_probs_i.tolist())

                for tgt_probs_i in batch.tgt_batch.classes:
                    tgt_probs.append(tgt_probs_i.tolist())

                input_ids += batch.ids_batch
                batch_i += 1

            total_valid_loss /= batch_i

            if args.device == "cuda":
                torch.cuda.empty_cache()

            output = generate_output(
                input_ids,
                src_sequences,
                gen_probs=gen_probs,
                tgt_probs=tgt_probs,
                src_tag_sequences=src_tag_sequences,
            )

            with open(valid_output_save_path, "w") as f:
                json.dump(output, f)
            with open(tmp_valid_output_save_path, "w") as f:
                json.dump(output, f)

            target_classes = np.array([item["tgt_probs"] for item in output])

            pred_classes = np.array([item["gen_probs"] for item in output])

            valid_f1 = f1_score(
                target_classes.argmax(1), pred_classes.argmax(1), average="macro"
            )

            valid_acc = accuracy_score(
                target_classes.argmax(1), pred_classes.argmax(1)
            )

            current_log = {
                "Epoch": epoch,
                "Train/Loss": 0,
                "Train/Class_Loss": total_train_loss,
                "Train/Class_F1": train_f1,
                "Train/Class_Acc": train_acc,
                "Train/BLEU": 0,
                "Valid/Loss": 0,
                "Valid/Class_Loss": total_valid_loss,
                "Valid/Class_F1": valid_f1,
                "Valid/Class_Acc": valid_acc,
                "Valid/BLEU": 0
            }

            pbar.write("Epoch {} ".format(epoch) + "#" * 22)

            # logging tensorboard
            for key, value in current_log.items():
                if key != "Epoch":
                    writer.add_scalar(f"wikigen/{key}", float(value), epoch)
                    pbar.write(f"{key}: {value:14.3f}")
            pbar.write("\n")

            logger.update_results(current_log)

            torch.save(model, os.path.join(results_path, "model.pth"))
            torch.save(model.state_dict(), state_dict_save_path)

            if args.metric == "loss":
                metric = current_log["Valid/Class_Loss"]

            if args.metric == "acc":
                metric = current_log["Valid/Class_Acc"]

            elif args.metric == "f1":
                metric = current_log["Valid/Class_F1"]

            else:
                raise NotImplementedError

            is_best, new_lrs = scheduler.step(metric, epoch)

            if is_best:
                best_metric = metric
                datadict = {"best_metric": f"{best_metric:.4f}_{epoch}"}
                logger.update_results(datadict)
                save_best()
            else:
                pass

            for i, new_lr in enumerate(new_lrs):
                pbar.write(f"Reduced learning rate of group {i} to {new_lr:.4e}.")
                if new_lr <= 1e-4:
                    stop = True
            if stop:
                break

    # --------------------------- TEST ------------------------------------------------------

    del model

    if args.device == "cuda":
        torch.cuda.empty_cache()

    # loading best model on valid
    model = torch.load(model_save_path + ".best").to(device=device)

    batch_i = 0
    total_test_loss = 0
    gen_probs = []
    src_sequences = []
    src_tag_sequences = []
    tgt_probs = []
    input_ids = []

    for batch in tqdm.tqdm(test_batches, desc="Testing... "):

        model.eval()

        src_batch_sequences = batch.src_batch.sequences
        src_batch_tag_sequences = batch.src_batch.tag_sequences

        batch.to_torch(device=device)

        loss, probs = model.forward(
            batch.src_batch,
            batch.tgt_batch,
            src_message_batch_tuple=batch.src_message_batch,
        )

        total_test_loss += loss.item()

        for id_sequence in src_batch_sequences:
            src_sequence_i = dataset.vocab.src.indices2tokens(
                id_sequence, ignore_ids=src_ignore_ids
            )
            src_sequences.append(" ".join(src_sequence_i))

        if src_batch_tag_sequences is not None:
            for id_sequence in src_batch_tag_sequences:
                src_tag_sequence_i = dataset.vocab.src_tag.indices2tokens(
                    id_sequence, ignore_ids=src_tag_ignore_ids
                )
                src_tag_sequences.append(" ".join(src_tag_sequence_i))

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
        src_tag_sequences=src_tag_sequences,
    )

    with open(test_output_save_path, "w") as f:
        json.dump(output, f)

    target_classes = np.array([item["tgt_probs"] for item in output])

    pred_classes = np.array([item["gen_probs"] for item in output])

    test_f1 = f1_score(
        target_classes.argmax(1), pred_classes.argmax(1), average="macro"
    )

    test_acc = accuracy_score(target_classes.argmax(1), pred_classes.argmax(1))

    test_bleu = 0.0

    datadict = {
        "Test/Loss": f"{0.0:.4f}",
        "Test/Class_Loss": f"{total_test_loss:.4f}",
        "Test/Class_F1": f"{test_f1:.4f}",
        "Test/Class_Acc": f"{test_acc:.4f}",
        "Test/BLEU": f"{test_bleu:.4f}",
    }

    logger.update_results(datadict)
