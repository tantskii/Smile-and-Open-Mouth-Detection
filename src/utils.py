import os
import numpy as np
from scipy.misc import imresize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard, CSVLogger

def get_absolute_file_pathways(directory):
    """
    Getting full image pathways
    :param directory: directory pathway
    :return: list of iamge pathways
    """
    file_names = os.listdir(directory)

    return [os.path.join(directory, file_name) for file_name in file_names]

def crop_image(image, bboxes, bbox_number=0, padding=0):
    """
    Crop face from image by bounding box
    :param image: rgb image
    :param bboxes: bounding boxes
    :param bbox_number: first face bounding box
    :param padding: padding
    :return: cropped image
    """
    crop_image = image[
                 bboxes[bbox_number][1]-padding : bboxes[bbox_number][1]+bboxes[bbox_number][3]+padding,
                 bboxes[bbox_number][0]-padding : bboxes[bbox_number][0]+bboxes[bbox_number][2]+padding,
                 :]

    return crop_image

def imresize_with_proportion(image, target_height):
    """
    Resizing with preservation of proportions
    :param image: rgb image
    :param target_height: target height of new image
    :return: resized image
    """
    fraction = target_height / image.shape[0]

    return imresize(image, fraction)

def crop_facemarks_coords(facemarks_coords, bboxes, bbox_number=0):
    """
    Offset face landmarks to zero coordinates
    :param facemarks_coords: face landmarks coordinates
    :param bboxes: bounding boxes
    :param bbox_number: first face bounding box
    :return: face landmarks with new coordinates
    """
    facemarks_coords = facemarks_coords.copy()
    facemarks_coords[bbox_number][:, 0] = facemarks_coords[bbox_number][:, 0] - bboxes[bbox_number][0]
    facemarks_coords[bbox_number][:, 1] = facemarks_coords[bbox_number][:, 1] - bboxes[bbox_number][1]

    return facemarks_coords[bbox_number]

def resize_facemarks_coords(face_facemarks_coords, original_crop_shape, target_crop_shape):
    """
    Resize face landmarks
    :param face_facemarks_coords: face landmarks coordinates
    :param original_crop_shape: original cropped image with face shape
    :param target_crop_shape: target cropped image with face shape
    :return:
    """
    face_facemarks_coords = face_facemarks_coords.copy()
    face_facemarks_coords[:, 0] = face_facemarks_coords[:, 0] * (target_crop_shape[1] / original_crop_shape[1])
    face_facemarks_coords[:, 1] = face_facemarks_coords[:, 1] * (target_crop_shape[0] / original_crop_shape[0])

    return face_facemarks_coords

def train_valid_test_split(pathways, labels, valid_len, test_len):
    """
    Simple holdout splitting
    :param pathways: image pathways
    :param labels: labels with classes
    :param valid_len: validation lenght
    :param test_len: test lenght
    :return: dictionaries
    """
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

def callbacks_factory(callbacks_list, model_mask):
    """
    Keras callbacks list creating
    :param callbacks_list: selected callbacks
    :param model_mask: mask for files
    :return:
    """
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