from imgaug import augmenters as iaa
from datasets import MTFLDataset
from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler,\
    ReduceLROnPlateau, TensorBoard, ModelCheckpoint

from .generator import DataGenerator

def create_train_valid_generators(augmentations_pipline, valid_proportion, seed, batch_size=32, shuffle=True):
    mtfl_dataset = MTFLDataset('../data/MTFL/',
                               'D:/Repositories/Project/data/AFLW.csv',
                               'D:/Repositories/Project/data/net.csv')
    mtfl_dataset.shuffle(seed)
    valid_len = int(len(mtfl_dataset.image_pathways) * valid_proportion)

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

def callbacks_factory(callbacks_list):
    callbacks = list()

    if 'best_model_checkpoint' in callbacks_list:
        best_model_filepath = '{0}/best_{1}.h5'.format('..models/', 'mobilenetv2_multiclassification')
        best_model_checkpoint = ModelCheckpoint(filepath=best_model_filepath,
                                                monitor='val_smile_output_f1_score',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='max',
                                                period=1)
        callbacks.append(best_model_checkpoint)

    if 'early_stopping' in callbacks_list:
        early_stopping = EarlyStopping(monitor='val_smile_output_f1_score',
                                       min_delta=0,
                                       patience=3,
                                       verbose=1,
                                       mode='max')
        callbacks.append(early_stopping)

    return callbacks


























