import numpy as np
from keras.utils import Sequence
from jpeg4py import JPEG
from scipy.misc import imresize
from face_detector import safe_detect_face_bboxes
from mtcnn.mtcnn import MTCNN
from utils import crop_image

class DataGenerator(Sequence):
    def __init__(
            self,
            pathways_with_smile_labels,
            pathways_with_open_mouth_labels,
            augmentations_pipline=None,
            shape=(256, 256),
            batch_size=32,
            shuffle=True):

        self.pathways_with_smile_labels = pathways_with_smile_labels
        self.pathways_with_open_mouth_labels = pathways_with_open_mouth_labels
        self.pathways = sorted(list(pathways_with_smile_labels.keys()))
        self.augmentations_pipline = augmentations_pipline
        self.shape=shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mtcnn = MTCNN()
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
            bboxes = safe_detect_face_bboxes(image, self.mtcnn)

            if bboxes.shape[0] == 0:
                continue
            else:
                cropped_image = crop_image(image, bboxes.clip(min=0), bbox_number=0)
                batch_x.append(imresize(cropped_image, self.shape))
                batch_y_smile.append(self.pathways_with_smile_labels[pathway])
                batch_y_open_mouth.append(self.pathways_with_open_mouth_labels[pathway])

        batch_x = np.asarray(batch_x, dtype=np.uint8)
        batch_y_smile = np.asarray(batch_y_smile)
        batch_y_open_mouth = np.asarray(batch_y_open_mouth)

        if self.augmentations_pipline:
            batch_x = self.augmentations_pipline.augment_images(batch_x)

        return batch_x / 255., {'smile_output': batch_y_smile, 'open_mouth_output': batch_y_open_mouth}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.pathways)