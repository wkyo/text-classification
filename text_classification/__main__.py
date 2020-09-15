# coding: utf-8
import os

from .text_classification import TextClassifier


def subcommand_init(args):
    """Initialize a text classification project"""
    project = args.project
    if not os.path.exists(project):
        os.mkdir(project)
    for item in ['data', 'models']:
        os.mkdir(os.path.join(project, item))
    with open(os.path.join(project, 'config.yml'), 'wt'):
        pass


def subcommand_train(args):
    tc = TextClassifier(proj_dir=args.project, lang=args.lang)
    tc.train(args.model)


def subcommand_predict(args):
    tc = TextClassifier(proj_dir=args.project)
    tc.load_model(args.model)
    for label in tc.predict(args.texts):
        print(label)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--project', default='.')
    # parser.add_argument('--data', default='data')
    # parser.add_argument('--models', default='models')
    # parser.add_argument('--config', default='config.yml')
    parser.add_argument('--lang', default=None, choices=['cn', 'en'])

    parsers = parser.add_subparsers(dest='cmd', help='subcommand')

    subparser_init = parsers.add_parser('init')

    subparser_train = parsers.add_parser('train')
    subparser_train.add_argument('--model', default='TextCNN')

    subparser_predict = parsers.add_parser('predict')
    subparser_predict.add_argument('--model', default=None, help='model path')
    subparser_predict.add_argument('texts', nargs='+')

    args = parser.parse_args()

    actions = {
        'init': subcommand_init,
        'train': subcommand_train,
        'predict': subcommand_predict
    }

    subcommand = actions.get(args.cmd)
    if subcommand:
        subcommand(args)


if __name__ == "__main__":
    main()
