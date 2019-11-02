#!/usr/bin/python
# coding: utf-8

from wikigen.main.classifier import main
from wikigen.config.classifier import parser, check_args
from wikigen.config import read_config

if __name__ == '__main__':
    args = parser.parse_args()

    if args.config:
        args = read_config(args.config)

    check_args(args)

    main(args)
