import imgaug as ia
from imgaug import augmenters as iaa
from datasets import MTFLDataset

from .generator import DataGenerator
from .model import create_mobilenetv2

SEED = 147
VALID_PROPORTION = 0.1
BATCH_SIZE = 8


def create_train_valid_generators(augmentations_pipline, batch_size=32, shuffle=True):
    mtfl_dataset = MTFLDataset('../data/MTFL/',
                               'D:/Repositories/Project/data/AFLW.csv',
                               'D:/Repositories/Project/data/net.csv')
    mtfl_dataset.shuffle(SEED)
    valid_len = int(len(mtfl_dataset.image_pathways) * VALID_PROPORTION)

    train_pathways_with_smile_labels = dict(
        zip(mtfl_dataset.image_pathways[:-valid_len], mtfl_dataset.smile_labels[:-valid_len]))
    valid_pathways_with_smile_labels = dict(
        zip(mtfl_dataset.image_pathways[-valid_len:], mtfl_dataset.smile_labels[-valid_len:]))

    train_pathways_with_open_mouth_labels = dict(
        zip(mtfl_dataset.image_pathways[:-valid_len], mtfl_dataset.open_mouth_labels[:-valid_len]))
    valid_pathways_with_open_mouth_labels = dict(
        zip(mtfl_dataset.image_pathways[-valid_len:], mtfl_dataset.open_mouth_labels[-valid_len:]))

    train_generator = DataGenerator(train_pathways_with_smile_labels,
                                    train_pathways_with_open_mouth_labels,
                                    augmentations_pipline=augmentations_pipline,
                                    batch_size=batch_size,
                                    shuffle=shuffle)
    valid_generator = DataGenerator(valid_pathways_with_smile_labels,
                                    valid_pathways_with_open_mouth_labels,
                                    augmentations_pipline=None,
                                    batch_size=batch_size,
                                    shuffle=shuffle)

    return train_generator, valid_generator

def get_augmentations_pipeline():
    augmentations_pipline = iaa.Sequential([
        iaa.Sometimes(0.8, iaa.OneOf([
            iaa.Affine(scale=(1, 1.8), rotate=(0, 90), shear=(0, 20), backend='cv2'),
            iaa.PerspectiveTransform(scale=(0.01, 0.10)),
            iaa.PiecewiseAffine(scale=(0.01, 0.05)),
            iaa.Fliplr(0.6)
        ]))
    ])

    return augmentations_pipline

def train():
    augmentations_pipeline = get_augmentations_pipeline()
    train_generator, valid_generator = create_train_valid_generators(augmentations_pipeline,
                                                                     batch_size=BATCH_SIZE)

    model = create_mobilenetv2((None, None, 3))




if __name__ == '__main__':
    train()



