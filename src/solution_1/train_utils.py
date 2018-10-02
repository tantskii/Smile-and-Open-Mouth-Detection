from imgaug import augmenters as iaa
from datasets import MTFLDataset
from utils import train_valid_test_split

from .generator import DataGenerator

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
        '../data/AFLW.csv',
        '../data/net.csv'
    )
    mtfl_dataset.shuffle(seed)
    valid_len = int(len(mtfl_dataset.image_pathways) * valid_proportion)
    test_len = int(len(mtfl_dataset.image_pathways) * test_proportion)

    pathways_with_smile_labels =train_valid_test_split(
        mtfl_dataset.image_pathways,
        mtfl_dataset.smile_labels,
        valid_len,
        test_len
    )
    pathways_with_open_mouth_labels = train_valid_test_split(
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