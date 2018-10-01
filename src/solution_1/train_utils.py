import numpy as np
from imgaug import augmenters as iaa
from datasets import MTFLDataset
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard, CSVLogger

from .generator import DataGenerator

def _train_valid_test_split(pathways, labels, valid_len, test_len):
    valid_test_len = valid_len + test_len

    train_pathways_with_labels = dict(
        zip(
            pathways[: -valid_test_len],
            labels[: -valid_test_len]
        )
    )
    valid_pathways_with_labels = dict(
        zip(
            pathways[-valid_test_len : -test_len],
            labels[-valid_test_len : -test_len]
        )
    )
    test_pathways_with_labels = dict(
        zip(
            pathways[-test_len :],
            labels[-test_len:]
        )
    )

    return (train_pathways_with_labels, valid_pathways_with_labels, test_pathways_with_labels)

def _get_augmentations_pipeline(mode='easy'):

    if mode == 'hard':
        augmentations_pipline = iaa.Sequential([
            iaa.Sometimes(0.8, iaa.OneOf([
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.ContrastNormalization((0.5, 1.5), per_channel=0.8),
                iaa.Grayscale(alpha=(0.5, 1.0)),
            ])),

            iaa.OneOf([
                iaa.Affine(scale=(1, 1.2), rotate=(0, 360), shear=(0, 20), backend='cv2'),
                iaa.PerspectiveTransform(scale=(0.01, 0.10)),
                iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                iaa.Sequential([
                    iaa.Fliplr(0.6),
                    iaa.Flipud(0.05),
                ])
            ])
        ])
    elif mode == 'easy':
        augmentations_pipline = iaa.Sequential([
            iaa.Sometimes(0.8, iaa.OneOf([
                iaa.Affine(scale=(1, 1.15), backend='cv2'),
                iaa.Affine(shear=(0, 5), backend='cv2'),
                iaa.Affine(translate_percent=0.1, backend='cv2'),
                iaa.Fliplr(0.6)
            ]))
        ])

    else:
        raise ValueError('There is no such mode, available: easy, hard')

    return augmentations_pipline


def train_valid_test_generators(
        valid_proportion,
        test_proportion,
        seed,
        shape,
        batch_size=32,
        shuffle=True):

    mtfl_dataset = MTFLDataset(
        '../data/MTFL/',
        'D:/Repositories/Project/data/AFLW.csv',
        'D:/Repositories/Project/data/net.csv'
    )
    mtfl_dataset.shuffle(seed)
    valid_len = int(len(mtfl_dataset.image_pathways) * valid_proportion)
    test_len = int(len(mtfl_dataset.image_pathways) * test_proportion)

    pathways_with_smile_labels =_train_valid_test_split(
        mtfl_dataset.image_pathways,
        mtfl_dataset.smile_labels,
        valid_len,
        test_len
    )
    pathways_with_open_mouth_labels = _train_valid_test_split(
        mtfl_dataset.image_pathways,
        mtfl_dataset.open_mouth_labels,
        valid_len,
        test_len
    )

    generators = {
        'hard_train_generator': DataGenerator(
            pathways_with_smile_labels[0],
            pathways_with_open_mouth_labels[0],
            augmentations_pipline=_get_augmentations_pipeline(mode='hard'),
            shape=shape,
            batch_size=batch_size,
            shuffle=shuffle
        ),
        'easy_train_generator': DataGenerator(
            pathways_with_smile_labels[0],
            pathways_with_open_mouth_labels[0],
            augmentations_pipline=_get_augmentations_pipeline(mode='easy'),
            shape=shape,
            batch_size=batch_size,
            shuffle=shuffle
        ),
        'valid_generator': DataGenerator(
            pathways_with_smile_labels[1],
            pathways_with_open_mouth_labels[1],
            augmentations_pipline=None,
            shape=shape,
            batch_size=batch_size,
            shuffle=shuffle
        ),
        'test_generator': DataGenerator(
            pathways_with_smile_labels[2],
            pathways_with_open_mouth_labels[2],
            augmentations_pipline=None,
            shape=shape,
            batch_size=batch_size,
            shuffle=shuffle
        )
    }
    return generators

def callbacks_factory(callbacks_list, model_mask):
    callbacks = list()

    if 'best_model_checkpoint' in callbacks_list:
        best_model_filepath = '{0}/best_{1}.h5'.format('../nn_models/', model_mask)
        best_model_checkpoint = ModelCheckpoint(
            filepath=best_model_filepath,
            monitor='val_smile_output_f1_score',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            period=1
        )
        callbacks.append(best_model_checkpoint)

    if 'early_stopping' in callbacks_list:
        early_stopping = EarlyStopping(
            monitor='val_smile_output_f1_score',
            min_delta=0,
            patience=3,
            verbose=1,
            mode='max'
        )
        callbacks.append(early_stopping)

    if 'tensorboard' in callbacks_list:
        tensorboard = TensorBoard(log_dir='../logs/{0}'.format(model_mask))
        callbacks.append(tensorboard)

    if 'csv_logger' in callbacks_list:
        csv_logger = CSVLogger(filename='../logs/{0}.log'.format(model_mask))
        callbacks.append(csv_logger)

    if 'learning_rate_scheduler' in callbacks_list:
        def exp_decay(epoch):
            initial_learning_rate = 0.001
            k = 0.1
            learning_rate = initial_learning_rate * np.exp(-k * epoch)

            return learning_rate

        callbacks.append(
            LearningRateScheduler(
                exp_decay,
                verbose=1
            )
        )

    return callbacks


























