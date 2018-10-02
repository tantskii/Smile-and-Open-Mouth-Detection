
from datasets import MTFLDataset
from utils import train_valid_test_split
from .generator import DataGenerator

def train_valid_test_generators(
        valid_proportion,
        test_proportion,
        seed,
        crop_shape,
        batch_size=32,
        shuffle=True):
    """
    Create train, valid and test keras sequence generators
    :param valid_proportion: validation fraction
    :param test_proportion: test fraction
    :param seed: random state
    :param crop_shape: target shape of cropped images with face
    :param batch_size: size of batches
    :param shuffle: shuffle image pathways before epochs
    :return: generators dict
    """
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
        'train_generator': DataGenerator(
            pathways_with_smile_labels[0],
            pathways_with_open_mouth_labels[0],
            crop_shape=crop_shape,
            batch_size=batch_size,
            shuffle=shuffle
        ),
        'valid_generator': DataGenerator(
            pathways_with_smile_labels[1],
            pathways_with_open_mouth_labels[1],
            crop_shape=crop_shape,
            batch_size=batch_size,
            shuffle=shuffle
        ),
        'test_generator': DataGenerator(
            pathways_with_smile_labels[2],
            pathways_with_open_mouth_labels[2],
            crop_shape=crop_shape,
            batch_size=batch_size,
            shuffle=shuffle
        )
    }

    return generators