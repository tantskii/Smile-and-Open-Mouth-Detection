import os
import time
import dlib
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from metrics import f1_score
from keras.utils import CustomObjectScope
from keras.models import load_model
from jpeg4py import JPEG
from sklearn.externals import joblib
from tqdm import tqdm
from face_detector import safe_detect_face_bboxes
from mtcnn.mtcnn import MTCNN
from facemarks_detector import detect_facemarks_coords
from utils import crop_facemarks_coords, resize_facemarks_coords

def predict(args):
    smile_faces = list()
    open_mouth_faces = list()
    facemark_inference_measurements = list()
    prediction_inference_measurements = list()
    image_names = os.listdir(args.images_directory)

    crop_shape = (args.height, args.width)
    target_facemarks = list(range(17, 27)) + list(range(36, 68))
    mtcnn = MTCNN()
    scaler = joblib.load('../models/solution_3_scaler.save')
    facemark_predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')

    with CustomObjectScope({'f1_score': f1_score}):
        model = load_model('../nn_models/best_mlp_multiclassification.h5')

    for image_name in tqdm(image_names):
        image = JPEG(os.path.join(args.images_directory, image_name)).decode()

        start_facemarks_time = time.time()
        bboxes = safe_detect_face_bboxes(image, mtcnn)
        if bboxes.shape[0] == 0:
            continue
        else:
            facemarks_coords = detect_facemarks_coords(
                image,
                bboxes.clip(min=0),
                facemark_predictor_init=facemark_predictor
            )
            facemark_inference_measurements.append(time.time() - start_facemarks_time)

            start_predict_time = time.time()
            cropped_facemarks_coords = crop_facemarks_coords(
                facemarks_coords,
                bboxes,
                bbox_number=0
            )
            resized_cropped_facemarks_coords = resize_facemarks_coords(
                cropped_facemarks_coords,
                original_crop_shape=(bboxes[0][3], bboxes[0][2]),
                target_crop_shape=crop_shape
            )
            face_features = resized_cropped_facemarks_coords[target_facemarks, :].ravel()
            face_features = scaler.transform(face_features.reshape(1, -1))
            predictions = model.predict(face_features)
            prediction_inference_measurements.append(time.time() - start_predict_time)
            predictions = [float(prediction) for prediction in predictions]

            if predictions[0] >= 0.985:
                smile_faces.append(image_name)

            if predictions[1] >= 0.92:
                open_mouth_faces.append(image_name)

    print('\nAverage facemark searching inference time: {0} sec.'.format(
        np.round(np.mean(facemark_inference_measurements), 3))
    )
    print('\nAverage prediction inference time: {0} sec.'.format(
        np.round(np.mean(prediction_inference_measurements), 3))
    )

    print('\nIMAGES WITH SMILE')
    print('-----------------')
    for image in smile_faces:
        print('  {0}'.format(image))

    print('\nIMAGES WITH OPEN MOUTH')
    print('----------------------')
    for image in open_mouth_faces:
        print('  {0}'.format(image))