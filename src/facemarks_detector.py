import cv2
import dlib
import numpy as np

def facemarks_to_coords(facemarks, dtype=np.int):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (facemarks.part(i).x, facemarks.part(i).y)

    return coords


def detect_facemarks_coords(image,
                            bboxes,
                            facemarks_data_pathway='shape_predictor_68_face_landmarks.dat'):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facemarks_predictor = dlib.shape_predictor(facemarks_data_pathway)
    rectangles = dlib.rectangles()
    rectangles.extend([dlib.rectangle(left=bbox[0],
                                      top=bbox[1],
                                      right=bbox[2] + bbox[0],
                                      bottom=bbox[3] + bbox[1]) for bbox in bboxes])
    facemarks_coords = list()
    for rectangle in rectangles:
        facemarks = facemarks_predictor(gray_image, rectangle)
        facemarks_coords.append(facemarks_to_coords(facemarks))

    return facemarks_coords