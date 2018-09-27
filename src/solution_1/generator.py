import numpy as np
from keras.utils import Sequence
from skimage.util import pad
from jpeg4py import JPEG
from operator import itemgetter

class DataGenerator(Sequence):
    def __init__(self,
                 pathways_with_smile_labels,
                 pathways_with_open_mouth_labels,
                 augmentations_pipline=None,
                 batch_size=32,
                 shuffle=True):

        self.pathways_with_smile_labels = pathways_with_smile_labels
        self.pathways_with_open_mouth_labels = pathways_with_open_mouth_labels
        self.pathways = list(pathways_with_smile_labels.keys())
        self.augmentations_pipline = augmentations_pipline
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pathways) / float(self.batch_size)))

    # TODO: Whitening instead normalization
    def __getitem__(self, index):
        batch_pathways = self.pathways[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = list()
        batch_y_smile = list()
        batch_y_open_mouth = list()

        for i, pathway in enumerate(batch_pathways):
            image = JPEG(pathway).decode()
            batch_x.append(image)
            batch_y_smile.append(self.pathways_with_smile_labels[pathway])
            batch_y_open_mouth.append(self.pathways_with_open_mouth_labels[pathway])

        image_shapes = [image.shape[:2] for image in batch_x]
        pad_shapes = self._get_pad_shapes(image_shapes)
        batch_x = [pad(image, pad_shape, mode='constant') for image, pad_shape in zip(batch_x, pad_shapes)]

        batch_x = np.asarray(batch_x, dtype=np.uint8)
        batch_y_smile = np.asarray(batch_y_smile)
        batch_y_open_mouth = np.asarray(batch_y_open_mouth)

        if self.augmentations_pipline:
            batch_x = self.augmentations_pipline.augment_images(batch_x)

        return batch_x / 255., batch_y_smile, batch_y_open_mouth

    def _get_pad_shapes(self, image_shapes):
        max_height = max(image_shapes, key=itemgetter(0))[0]
        max_width = max(image_shapes, key=itemgetter(1))[1]

        pad_shapes = [(( int( (max_height-image_shape[0])/2 ), max_height-image_shape[0]-int( (max_height-image_shape[0])/2 ) ),
                       ( int( (max_width-image_shape[1])/2 ), max_width-image_shape[1]-int( (max_width-image_shape[1])/2 ) ),
                       ( 0, 0 )) for image_shape in image_shapes]

        return pad_shapes

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.pathways)