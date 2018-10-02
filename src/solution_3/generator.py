import dlib
import numpy as np
from keras.utils import Sequence
from jpeg4py import JPEG
from sklearn.preprocessing import StandardScaler
from facemarks_detector import detect_facemarks_coords
from face_detector import safe_detect_face_bboxes
from mtcnn.mtcnn import MTCNN
from utils import crop_facemarks_coords, resize_facemarks_coords

class DataGenerator(Sequence):
    def __init__(
            self,
            pathways_with_smile_labels,
            pathways_with_open_mouth_labels,
            crop_shape=(100, 100),
            batch_size=32,
            shuffle=True):

        self.pathways_with_smile_labels = pathways_with_smile_labels
        self.pathways_with_open_mouth_labels = pathways_with_open_mouth_labels
        self.pathways = sorted(list(pathways_with_smile_labels.keys()))
        self.crop_shape = crop_shape
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.target_facemarks = list(range(17, 27)) + list(range(36, 68))
        self.mtcnn = MTCNN()
        self.facemark_predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pathways) / float(self.batch_size)))

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
                facemarks_coords = detect_facemarks_coords(
                    image,
                    bboxes.clip(min=0),
                    facemark_predictor_init=self.facemark_predictor
                )
                cropped_facemarks_coords = crop_facemarks_coords(
                    facemarks_coords,
                    bboxes,
                    bbox_number=0
                )
                resized_cropped_facemarks_coords = resize_facemarks_coords(
                    cropped_facemarks_coords,
                    original_crop_shape=(bboxes[0][3], bboxes[0][2]),
                    target_crop_shape=self.crop_shape
                )
                face_features = resized_cropped_facemarks_coords[self.target_facemarks, :].ravel()
                batch_x.append(face_features)
                batch_y_smile.append(self.pathways_with_smile_labels[pathway])
                batch_y_open_mouth.append(self.pathways_with_open_mouth_labels[pathway])

        batch_x = np.asarray(batch_x)
        batch_x = StandardScaler().fit_transform(batch_x)
        batch_y_smile = np.asarray(batch_y_smile)
        batch_y_open_mouth = np.asarray(batch_y_open_mouth)

        return batch_x, {'smile_output': batch_y_smile, 'open_mouth_output': batch_y_open_mouth}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.pathways)