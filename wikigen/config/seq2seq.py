import argparse
import sys
import colored_traceback

from wikigen.logger import Logger
from wikigen.settings import DATASET_NAMES, RESULTS_PATH
from ..model.optimizers import optimizers

colored_traceback.add_hook(always=True)

parser = argparse.ArgumentParser(description="train_seq2seq.py")

parser.add_argument(
    "--dataset",
    required="--config" not in sys.argv,
    help="Name of the dataset, " "choose from " + " ".join(DATASET_NAMES),
)

parser.add_argument(
    "--results_path",
    default=RESULTS_PATH,
    help="Path to store results of the model and checkpoints",
)

parser.add_argument(
    "--encoder_embedding_size",
    type=int,
    default=400,
    help="Word embedding sizes. Default 400",
)

parser.add_argument(
    "--train_encoder_embeddings",
    action="store_true",
    help="Make encoder embeddings trainable",
)

parser.add_argument(
    "--encoder_num_layers",
    type=int,
    default=1,
    help="Layers of encoder LSTM. Default: 1",
)

parser.add_argument(
    "--encoder_dropout",
    default=0,
    help="Dropout of encoder LSTM (only use on multi-layer). Default: None",
)

parser.add_argument(
    "--use_tag_embeddings",
    action="store_true",
    help="Use embeddings for diff tags",
)

parser.add_argument(
    "--encoder_tag_embedding_size",
    type=int,
    default=10,
    help="Embedding size for diff tags. Default 10",
)

parser.add_argument(
    "--encoder_hidden_size",
    type=int,
    default=200,
    help="Hidden dimension of the encoder lstm. Default: 200",
)

parser.add_argument(
    "--encoder_input_dropout",
    type=float,
    default=0,
    help="Dropout probability; applied before encoder LSTM stacks. Default: 0",
)

parser.add_argument(
    "--encoder",
    default="bow",
    choices=["bow", "lstm", "linear"],
    help="Type of encoder to use. Default: bow",
)

parser.add_argument(
    "--bidirectional",
    action="store_true",
    help="Use a bidirectional version, when using any RNN encoder",
)

parser.add_argument(
    "--encoder_output_dropout",
    type=float,
    default=0,
    help="Dropout probability; applied after encoder LSTM. Default: 0",
)

parser.add_argument(
    "--clip", type=float, default=5, help="Gradient clipping. Default: 5."
)

parser.add_argument(
    "--aggregator",
    choices=["last", "max", "mean", "sum"],
    default=None,
    help="How to initialize the decoder. Default: None (zero)",
)

parser.add_argument(
    "--attention",
    type=str,
    choices=["dot", "general", "concat"],
    default="general",
    help="Attention type. Default: general",
)

parser.add_argument(
    "--decoder_embedding_size",
    type=int,
    default=400,
    help="Word embedding sizes. Default: 400",
)

parser.add_argument(
    "--decoder_hidden_size",
    type=int,
    default=400,
    help="Word embedding sizes. Default: 400",
)

parser.add_argument(
    "--decoder_input_dropout",
    type=float,
    default=0.5,
    help="Dropout probability; applied before decoder LSTM stacks. Default: 0.5",
)

parser.add_argument(
    "--decoder_output_dropout",
    type=float,
    default=0.5,
    help="Dropout probability; applied after decoder LSTM. Default: 0.5",
)

parser.add_argument(
    "--teacher_forcing_p",
    type=float,
    default=0.5,
    help="Teacher fording prob. Default 0.5",
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=0,
    help="For classification. Default: 0 (do not perform classification",
)

parser.add_argument(
    "--class_lambda",
    type=float,
    default=None,
    help="Multiplier for classification loss. Default: None, losses are added.",
)

parser.add_argument(
    "--lr", type=float, default=0.5, help="initial learning rate. Default 0.5"
)

parser.add_argument(
    "--metric",
    type=str,
    default="loss",
    choices=["loss", "bleu", "f1", "acc"],
    help="Metric for early stopping and model selection. Default: loss",
)

parser.add_argument(
    "--decay",
    type=float,
    default=0.5,
    help="learning rate decay. 1 means no decay.Default 0.5 ",
)

parser.add_argument(
    "--patience",
    type=int,
    default=5,
    help="Max number of epochs without improvement in valid to apply decay. Default 5 ",
)

parser.add_argument("--epochs", type=int, default=100, help="Upper epoch limit.")

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for train and valid. Default 32",
)

parser.add_argument(
    "--test_batch_size",
    type=int,
    default=8,
    help="Test batch size, for beam search. Default 8",
)

parser.add_argument(
    "--max_len", type=int, default=100, help="Maximum length. Default: 100"
)

parser.add_argument(
    "--lowercase", action="store_true", help="Lowercase all data."
)

parser.add_argument(
    "--src_min_freq", type=int, default=1, help="Source min freq. Default: 1"
)

parser.add_argument(
    "--tgt_min_freq", type=int, default=1, help="Target min length. Default: 1"
)

parser.add_argument(
    "--joint",
    action="store_true",
    help="Use same embeddings for source and target.",
)

parser.add_argument(
    "--beam_size",
    type=int,
    default=10,
    help="Size of beam for beam search. Default: 10",
)

parser.add_argument(
    "--optim",
    default="SGD",
    choices=optimizers.keys(),
    type=str,
    help="Optimizer. Default: SGD",
)

parser.add_argument("--seed", type=int, default=1111, help="random seed")

parser.add_argument(
    "--device", default="cuda", choices=["cpu", "cuda"], help="Device"
)

parser.add_argument(
    "--verbose",
    type=int,
    default=100,
    metavar="N",
    help="Report interval, in number of batches. Default: 100",
)

parser.add_argument(
    "--overwrite", action="store_true", help="Overwrite if existing."
)

parser.add_argument(
    "--force_dataset_reload",
    action="store_true",
    help="Force dataset reload if exists.",
)

parser.add_argument(
    "--config",
    type=str,
    default="",
    help="Read config from json file",
    metavar="",
)

parser.add_argument(
    "--write_mode",
    type=str,
    default="NONE",
    choices=Logger.WRITE_MODES,
    help="Mode for writing logging info",
)


def check_args(args):
    if "_meta" not in args.dataset:
        args.use_tag_embeddings = False
        args.encoder_tag_embedding_size = None

    else:
        if not args.use_tag_embeddings:
            args.encoder_tag_embedding_size = None

    if args.joint:
        args.src_min_freq = None

    if args.class_lambda == 0:
        args.class_lambda = None

    if args.num_classes == 0:
        args.num_classes = None
        args.class_lambda = None

    if args.encoder != "lstm":
        args.encoder_num_layers = 1

    if args.encoder_num_layers == 1:
        args.encoder_dropout = 0

