#!/usr/bin/env python
# -*-coding: utf8 -*-

import os

CODE_ROOT = os.path.dirname(os.path.realpath(__file__))

HOME = os.environ["HOME"]

DATA_PATH = os.path.join(HOME, "data", "wikigen")

RESULTS_PATH = os.path.join(HOME, "results", "wikigen")

PARSED_EDITS_PATH = os.path.join(DATA_PATH, "parsed_edits")

SPLITS_PATH = os.path.join(DATA_PATH, "splits")

DOC2VEC_PATH = os.path.join(RESULTS_PATH, "doc2vec")

try:
    DATASET_NAMES = [
        name.replace(".jsonl", "") for name in os.listdir(PARSED_EDITS_PATH)
    ]
except FileNotFoundError:
    DATASET_NAMES = []

CLASS_LABELS = ["Stub", "Start", "C", "B", "GA", "FA"]

_DB_NAME = "runs.db"

PARAM_IGNORE_LIST = [
    "results_path",
    "overwrite",
    "force_dataset_reload",
    "verbose",
    "write_mode",
]

DATABASE_CONNECTION_STRING = "sqlite:///" + os.path.join(RESULTS_PATH, _DB_NAME)
