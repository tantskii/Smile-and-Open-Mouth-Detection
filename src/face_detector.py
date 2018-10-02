import cv2
import dlib
import numpy as np
from mtcnn.mtcnn import MTCNN

def detect_face_by_cascade(
        image,
        detector_type,
        haarcascade_xml_pathway,
        lbpcascade_xml_pathway,
        cascade_scale_factor,
        cascade_min_neighbors):
    """
    Detect face by opencv haar cascades
    :param image: rgb image
    :param detector_type: type of cascades
    :param haarcascade_xml_pathway: haar data file
    :param lbpcascade_xml_pathway: lbp data file
    :param cascade_scale_factor:
    :param cascade_min_neighbors:
    :return: bounding boxes
    """
    face_detectors = {
        'haarcascade': cv2.CascadeClassifier(haarcascade_xml_pathway),
        'lbpcascade': cv2.CascadeClassifier(lbpcascade_xml_pathway)
    }
    face_detector = face_detectors[detector_type]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bboxes = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=cascade_scale_factor,
        minNeighbors=cascade_min_neighbors
    )

    if isinstance(bboxes, tuple):
        return np.asarray(list())
    else:
        return bboxes


def detect_face_by_hogsvm_cnn(
        image,
        detector_type,
        cnn_dat_pathway,
        dlib_upsample):
    """
    Detect face by dlib hog + svm
    :param image: rgb image
    :param detector_type: cnn or svm type
    :param cnn_dat_pathway: cnn source file
    :param dlib_upsample: image upsample factor
    :return: bounding boxes
    """
    face_detectors = {
        'hogsvm': dlib.get_frontal_face_detector(),
        'cnn': dlib.cnn_face_detection_model_v1(cnn_dat_pathway)
    }
    face_detector = face_detectors[detector_type]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectangles = face_detector(gray_image, dlib_upsample)
    if isinstance(rectangles, dlib.mmod_rectangles):
        rectangles = [mmod_rectangle.rect for mmod_rectangle in rectangles]

    bboxes = list()
    for rectangle in rectangles:
        bboxes.append([
            rectangle.left(),
            rectangle.top(),
            rectangle.right() - rectangle.left(),
            rectangle.bottom() - rectangle.top()
        ])

    return np.asarray(bboxes)


def detect_face_by_mtcnn(
        image,
        mtcnn_init,
        mtcnn_confidence_threshold):
    """
    Detect face by multitask cascaded convolutional networks
    :param image: rgb image
    :param mtcnn_init: pre initialization for speed
    :param mtcnn_confidence_threshold: network confidence threshold
    :return: bounding boxes
    """
    if mtcnn_init:
        face_detector = mtcnn_init
    else:
        face_detector = MTCNN()

    face_objects = face_detector.detect_faces(image)
    bboxes = [face_object['box'] for face_object in face_objects if
              face_object['confidence'] > mtcnn_confidence_threshold]

    return np.asarray(bboxes)


def detect_face_bboxes(
        image,
        detector_type='haarcascade',
        haarcascade_xml_pathway='../models/haarcascades/haarcascade_frontalface_alt2.xml',
        lbpcascade_xml_pathway='../models/lbpcascades/lbpcascade_frontalface_improved.xml',
        cnn_dat_pathway='../models/mmod_human_face_detector.dat',
        cascade_scale_factor=1.5,
        cascade_min_neighbors=5,
        dlib_upsample=1,
        mtcnn_init=None,
        mtcnn_confidence_threshold=0.95):
    """
    Сommon function for all detectors
    :param image: rgb image
    :param detector_type: select detector type
    :param haarcascade_xml_pathway: haar data file
    :param lbpcascade_xml_pathway: lbp data file
    :param cnn_dat_pathway: cnn source file
    :param cascade_scale_factor:
    :param cascade_min_neighbors:
    :param dlib_upsample: image upsample factor
    :param mtcnn_init: pre initialization for speed
    :param mtcnn_confidence_threshold: network confidence threshold
    :return: bounding boxes
    """

    if detector_type == 'haarcascade' or detector_type == 'lbpcascade':
        return detect_face_by_cascade(
            image,
            detector_type,
            haarcascade_xml_pathway,
            lbpcascade_xml_pathway,
            cascade_scale_factor,
            cascade_min_neighbors
        )

    elif detector_type == 'hogsvm' or detector_type == 'cnn':
        return detect_face_by_hogsvm_cnn(
            image,
            detector_type,
            cnn_dat_pathway,
            dlib_upsample
        )

    elif detector_type == 'mtcnn':
        return detect_face_by_mtcnn(
            image,
            mtcnn_init,
            mtcnn_confidence_threshold
        )

    else:
        raise ValueError('There is no such detector, available: haarcascade, lbpcascade, hogsvm, cnn, mtcnn')


def safe_detect_face_bboxes(image, mtcnn, include_cnn=False):
    """
    Alternate detector selection until a face is found
    :param image: rgb image
    :param mtcnn: pre initialization for speed
    :param include_cnn: disable cnn for speed
    :return: bounding boxes
    """
    bboxes = detect_face_bboxes(image, detector_type='mtcnn', mtcnn_init=mtcnn)

    if bboxes.shape[0] == 0:
        bboxes = detect_face_bboxes(image, detector_type='hogsvm')

    if bboxes.shape[0] == 0:
        bboxes = detect_face_bboxes(image, detector_type='haarcascade')

    if include_cnn:
        if bboxes.shape[0] == 0:
            bboxes = detect_face_bboxes(image, detector_type='cnn')

    return bboxes