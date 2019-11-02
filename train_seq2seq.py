#!/usr/bin/python
# coding: utf-8

from wikigen.main.seq2seq import main
from wikigen.config.seq2seq import parser, check_args
from wikigen.config import read_config

if __name__ == '__main__':
    args = parser.parse_args()

    if args.config:
        args = read_config(args.config)

    check_args(args)

    main(args)
