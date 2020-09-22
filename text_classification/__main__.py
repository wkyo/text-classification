# coding: utf-8
import os
import warnings
import logging
import sys

from .classifier import Classifier
from .model_loader import scan_models


def main():
    from argparse import ArgumentParser
    from argparse import FileType

    parser = ArgumentParser()
    parser.add_argument('--data', default='data.csv', help='data path')
    parser.add_argument('--data-field-label', default='label')
    parser.add_argument('--data-field-text', default='text')
    parser.add_argument('--models', default='models', help='models directory')
    parser.add_argument('--load-model', default=None,
                        help='the path of the model to be loaded')
    parser.add_argument('--model', default='text_cnn',
                        help='model name, used in training phase')
    parser.add_argument('--train', default=False,
                        action='store_true', help='train model')
    parser.add_argument('--ls', default=False,
                        action='store_true', help='list all supported models')
    parser.add_argument('--stdin', default=None, const=sys.stdin, action='store_const',
                        help='read text line by line from standard input')
    parser.add_argument('texts', nargs='*')

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format=r'%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )

    if args.ls:
        print(scan_models())
    elif args.train:
        text_classifier = Classifier(
            data_path=args.data,
            models_path=args.models,
            data_field_label=args.data_field_label,
            data_field_text=args.data_field_text,
            model_auto_load=False
        )
        text_classifier.train(args.model)
    else:
        
        text_classifier = Classifier(
            models_path=args.models,
            model_path=args.load_model
        )
        if args.stdin is None:
            if not args.texts:
                warnings.warn('No input texts found')
            else:
                labels = text_classifier.predict(args.texts)
                for x in labels:
                    print(x)
        else:
            for line in args.stdin:
                line = line.strip()
                if line:
                    labels = text_classifier.predict([line])
                    print(labels[0])


if __name__ == "__main__":
    main()
