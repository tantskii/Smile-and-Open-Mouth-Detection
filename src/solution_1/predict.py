import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import matplotlib.pyplot as plt
from metrics import f1_score
from keras.utils import CustomObjectScope
from keras.models import load_model
from datasets import TestDataset
from jpeg4py import JPEG
from sklearn.metrics import f1_score as f1
from tqdm import tqdm
from face_detector import safe_detect_face_bboxes
from mtcnn.mtcnn import MTCNN
from utils import crop_image
from scipy.misc import imresize

def predict():
    with CustomObjectScope({'f1_score': f1_score}):
        model = load_model('../nn_models/best_mobilenetv2_multiclassification.h5')

    test_dataset = TestDataset('../data_test/example_data/example_data')
    smile_predictions = list(); smile_predictions2 = list()
    open_mouth_predictions = list(); open_mouth_predictions2 = list()

    mtcnn = MTCNN()
    for image_pathway in tqdm(test_dataset.image_pathways):
        image = JPEG(image_pathway).decode()
        bboxes = safe_detect_face_bboxes(image, mtcnn)
        cropped_image = crop_image(image, bboxes.clip(min=0), bbox_number=0)
        cropped_image = imresize(cropped_image, (192, 192)) / 255.

        predictions = model.predict(np.expand_dims(cropped_image, axis=0))
        predictions = [float(pred) for pred in predictions]

        # plt.imshow(cropped_image)
        # plt.show()
        # print(predictions)

        smile_predictions.append(0) if predictions[0] < 0.955 else smile_predictions.append(1) #0.96
        open_mouth_predictions.append(0) if predictions[1] < 0.5 else open_mouth_predictions.append(1)

        smile_predictions2.append(predictions[0])
        open_mouth_predictions2.append(predictions[1])


    print(f1(test_dataset.smile_labels, smile_predictions))
    print(f1(test_dataset.open_mouth_labels, open_mouth_predictions))

    a = 4

if __name__ == '__main__':
    predict()

    # from sklearn.metrics import precision_recall_curve
    # import matplotlib.pyplot as plt
    #
    # prec, rec, tre = precision_recall_curve(test_dataset.smile_labels, smile_predictions)
    #
    #
    # def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    #     plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    #     plt.plot(thresholds, recalls[:-1], 'g--', label='recall')
    #     plt.xlabel('Threshold')
    #     plt.legend(loc='upper left')
    #     plt.ylim([0, 1])
    #
    #
    # plot_prec_recall_vs_tresh(prec, rec, tre)
    # plt.show()

    # 0.6562499999999999
    # 0.9166666666666667