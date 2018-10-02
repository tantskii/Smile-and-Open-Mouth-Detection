import os
import time
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.utils import CustomObjectScope
from keras.models import load_model
from metrics import f1_score
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from jpeg4py import JPEG
from face_detector import safe_detect_face_bboxes
from utils import crop_image
from scipy.misc import imresize

def predict(args):
    """
    Solution 1 prediction script for directory with images for smile and open mouth detection
    :param args: argparse arguments
    :return: prints inference measured time and two lists of images that have passed the filter
    """
    mtcnn = MTCNN()
    image_names = os.listdir(args.images_directory)
    smile_faces = list()
    open_mouth_faces = list()
    inference_measurements = list()

    with CustomObjectScope({'f1_score': f1_score}):
        model = load_model('../nn_models/best_mobilenetv2_multiclassification.h5')

    for image_name in tqdm(image_names):
        image = JPEG(os.path.join(args.images_directory, image_name)).decode()
        start_time = time.time()
        bboxes = safe_detect_face_bboxes(image, mtcnn)

        if bboxes.shape[0] == 0:
            continue
        else:
            cropped_image = crop_image(image, bboxes.clip(min=0), bbox_number=0)
            cropped_image = imresize(cropped_image, (args.height, args.width)) / 255.
            predictions = model.predict(np.expand_dims(cropped_image, axis=0))
            inference_measurements.append(time.time() - start_time)
            predictions = [float(prediction) for prediction in predictions]

            if predictions[0] >= args.smile_prediction_threshold:
                smile_faces.append(image_name)

            if predictions[1] >= args.mouth_open_prediction_threshold:
                open_mouth_faces.append(image_name)

    print('\nAverage end to end inference time: {0} sec.'.format(np.round(np.mean(inference_measurements), 3)))

    print('\nIMAGES WITH SMILE')
    print('-----------------')
    for image in smile_faces:
        print('  {0}'.format(image))

    print('\nIMAGES WITH OPEN MOUTH')
    print('----------------------')
    for image in open_mouth_faces:
        print('  {0}'.format(image))