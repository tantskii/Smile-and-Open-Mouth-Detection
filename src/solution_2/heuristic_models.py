import dlib
import time
import numpy as np
from face_detector import safe_detect_face_bboxes
from facemarks_detector import detect_facemarks_coords
from jpeg4py import JPEG
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import euclidean
from tqdm import tqdm


def _get_optimal_threshold(
        data_of_positive_labels,
        data_of_negative_labels,
        hist_bins=15,
        clip=0.3,
        round_to=3):
    data_of_positive_labels = [round(data, round_to) for data in data_of_positive_labels]
    data_of_negative_labels = [round(data, round_to) for data in data_of_negative_labels]
    bins = np.linspace(0, clip, hist_bins, endpoint=True)

    data_of_positive_labels_hist = np.histogram(
        data_of_positive_labels,
        bins=bins,
        density=True,
        normed=True
    )
    data_of_negative_labels_hist = np.histogram(
        data_of_negative_labels,
        bins=bins,
        density=True,
        normed=True
    )

    min_hists_differnce = np.inf
    data_at_interaction = None
    for x_data, y_positive, y_negative in zip(
            data_of_positive_labels_hist[1][3:-1],
            data_of_positive_labels_hist[0][3:],
            data_of_negative_labels_hist[0][3:]):

        hists_difference = np.abs(y_positive - y_negative)
        if hists_difference < min_hists_differnce:
            min_hists_differnce = hists_difference
            data_at_interaction = x_data

    return data_at_interaction


class HeuristicMouthStateDetector(object):
    def __init__(self, mouth_aspect_ratio_threshold=None):
        self.mouth_aspect_ratio_threshold = mouth_aspect_ratio_threshold
        self.mtcnn = MTCNN()
        self.facemark_predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
        self.facemark_inference_measurements = list()
        self.predict_inference_measurements = list()

    def fit(self, image_pathways, mouth_open_labels):
        if not self.mouth_aspect_ratio_threshold:
            open_mouth_mars = list()
            close_mouth_mars = list()

            for image_pathway, mouth_open_label in tqdm(zip(image_pathways, mouth_open_labels)):
                image = JPEG(image_pathway).decode()
                bboxes = safe_detect_face_bboxes(image, self.mtcnn, include_cnn=True)

                if bboxes.shape[0] == 0:
                    continue
                else:
                    facemarks_coords = detect_facemarks_coords(
                        image,
                        bboxes.clip(min=0),
                        facemark_predictor_init=self.facemark_predictor
                    )
                    mouth_aspect_ratio = self._get_mouth_aspect_ratio(facemarks_coords[0])
                    open_mouth_mars.append(mouth_aspect_ratio)\
                        if mouth_open_label == 1\
                        else close_mouth_mars.append(mouth_aspect_ratio)

            self.mouth_aspect_ratio_threshold = _get_optimal_threshold(
                open_mouth_mars,
                close_mouth_mars,
                hist_bins=15,
                clip=0.3,
                round_to=2
            )

    def predict(self, image_pathways):
        if self.mouth_aspect_ratio_threshold:
            self.facemark_inference_measurements = list()
            self.predict_inference_measurements = list()
            predictions = list()

            for image_pathway in tqdm(image_pathways):
                image = JPEG(image_pathway).decode()


                start_facemarks_time = time.time()
                bboxes = safe_detect_face_bboxes(image, self.mtcnn, include_cnn=True)
                if bboxes.shape[0] == 0:
                    predictions.append(0)
                else:
                    facemarks_coords = detect_facemarks_coords(image, bboxes.clip(min=0), self.facemark_predictor)
                    self.facemark_inference_measurements.append(time.time() - start_facemarks_time)


                    start_predict_time = time.time()
                    mouth_aspect_ratio = self._get_mouth_aspect_ratio(facemarks_coords[0])
                    self.predict_inference_measurements.append(time.time() - start_predict_time)


                    predictions.append(0)\
                        if mouth_aspect_ratio < self.mouth_aspect_ratio_threshold\
                        else predictions.append(1)

            return np.asarray(predictions)

        else:
            raise ValueError('Train or set the mouth_aspect_ratio value')

    def _get_mouth_aspect_ratio(self, facemarks_coords):
        mouth_inner_facemarks_coords = facemarks_coords[[60, 62, 64, 66], :]

        height = euclidean(mouth_inner_facemarks_coords[1], mouth_inner_facemarks_coords[3])
        width = euclidean(mouth_inner_facemarks_coords[0], mouth_inner_facemarks_coords[2])
        mouth_aspect_ratio = height / width

        return mouth_aspect_ratio


class HeuristicSmileDetector(object):
    def __init__(self, smile_deviations_sum_threshold=None):
        self.smile_deviations_sum_threshold = smile_deviations_sum_threshold
        self.mtcnn = MTCNN()
        self.facemark_predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
        self.lower_lip_point_pairs = [(48, 60), (58, 67), (57, 66), (56, 65), (54, 64)]
        self.facemark_inference_measurements = list()
        self.predict_inference_measurements = list()

    def fit(self, image_pathways, smile_labels):
        if not self.smile_deviations_sum_threshold:
            smile_deviations_sum = list()
            not_smile_deviations_sum = list()

            for image_pathway, smile_label in tqdm(zip(image_pathways, smile_labels)):
                image = JPEG(image_pathway).decode()
                bboxes = safe_detect_face_bboxes(image, self.mtcnn, include_cnn=True)

                if bboxes.shape[0] == 0:
                    continue
                else:
                    facemarks_coords = detect_facemarks_coords(
                        image,
                        bboxes.clip(min=0),
                        facemark_predictor_init=self.facemark_predictor
                    )
                    lower_lip_points = self._calculate_line_points(facemarks_coords[0], self.lower_lip_point_pairs)
                    deviations_sum = self._get_deviations_sum(lower_lip_points)
                    smile_deviations_sum.append(deviations_sum)\
                        if smile_label == 1\
                        else not_smile_deviations_sum.append(deviations_sum)

            self.smile_deviations_sum_threshold = _get_optimal_threshold(
                smile_deviations_sum,
                not_smile_deviations_sum,
                hist_bins=15,
                clip=0.06,
                round_to=3
            )

    def predict(self, image_pathways):
        if self.smile_deviations_sum_threshold:
            self.facemark_inference_measurements = list()
            self.predict_inference_measurements = list()
            predictions = list()

            for image_pathway in tqdm(image_pathways):
                image = JPEG(image_pathway).decode()


                start_facemarks_time = time.time()
                bboxes = safe_detect_face_bboxes(image, self.mtcnn, include_cnn=True)
                if bboxes.shape[0] == 0:
                    predictions.append(0)
                else:
                    facemarks_coords = detect_facemarks_coords(image, bboxes.clip(min=0), self.facemark_predictor)
                    self.facemark_inference_measurements.append(time.time() - start_facemarks_time)


                    start_predict_time = time.time()
                    lower_lip_points = self._calculate_line_points(facemarks_coords[0], self.lower_lip_point_pairs)
                    deviations_sum = self._get_deviations_sum(lower_lip_points)
                    self.predict_inference_measurements.append(time.time() - start_predict_time)


                    predictions.append(0)\
                        if deviations_sum < self.smile_deviations_sum_threshold\
                        else predictions.append(1)

            return np.asarray(predictions)

        else:
            raise ValueError('Train or set the smile_deviations_sum_threshold value')


    def _calculate_line_points(self, facemarks_coords, point_pairs):
        line_points = list()

        for point_pair in point_pairs:
            line_points.append((facemarks_coords[point_pair[0], :] + facemarks_coords[point_pair[1], :]) / 2)

        return np.asarray(line_points, dtype=np.int)

    def _normalize_points(self, line_points):
        row_sums = line_points.sum(axis=0)
        line_points = line_points / row_sums

        return line_points

    def _get_deviations_sum(self, line_points):
        line_points = self._normalize_points(line_points)
        dist = line_points[4] - line_points[0]
        deviations_sum = 0

        for i in range(1, 4):
            deviation = np.linalg.norm(np.cross(dist, line_points[0] - line_points[i])) / np.linalg.norm(dist)
            deviations_sum += deviation

        return deviations_sum