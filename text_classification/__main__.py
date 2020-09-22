# coding: utf-8
import os
import warnings
import logging

from .classifier import Classifier
from .model_loader import scan_models


def main():
    from argparse import ArgumentParser

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
            data_field_text=args.data_field_text
        )
        text_classifier.train(args.model)
    else:
        if not args.texts:
            warnings.warn('No input texts found')
        text_classifier = Classifier(
            models_path=args.models,
            model_path=args.load_model
        )
        labels = text_classifier.predict(args.texts)
        print('Predict results:')
        print('>> {}'.format(labels))


if __name__ == "__main__":
    main()
