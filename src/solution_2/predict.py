import json
import numpy as np
from utils import get_absolute_file_pathways
from .heuristic_models import HeuristicSmileDetector, HeuristicMouthStateDetector

def predict_separately(model, image_pathways):
    predictions = model.predict(image_pathways)
    image_pathways = np.asarray(image_pathways)
    predicted_image_pathways = image_pathways[np.where(predictions == 1)].tolist()
    predicted_image_names = [predicted_image_pathway.split('\\')[1]
                             for predicted_image_pathway
                             in predicted_image_pathways]

    print('\n\nAverage facemark searching inference time: {0} sec.'.format(
        np.round(np.mean(model.facemark_inference_measurements), 3))
    )
    print('Average prediction inference time: {0} sec.\n\n'.format(
        np.round(np.mean(model.predict_inference_measurements), 3))
    )

    return predicted_image_names


def predict(args):
    image_pathways = get_absolute_file_pathways(args.images_directory)
    with open('../models/heuristic_models.json') as in_file:
        thresholds = json.load(in_file)

    heuristic_smile_detector = HeuristicSmileDetector(thresholds['smile_deviations_sum_threshold'])
    heuristic_mouth_state_detector = HeuristicMouthStateDetector(thresholds['mouth_aspect_ratio_threshold'])

    smile_faces = predict_separately(heuristic_smile_detector, image_pathways)
    open_mouth_faces = predict_separately(heuristic_mouth_state_detector, image_pathways)

    print('\nIMAGES WITH SMILE')
    print('-----------------')
    for image in smile_faces:
        print('  {0}'.format(image))

    print('\nIMAGES WITH OPEN MOUTH')
    print('----------------------')
    for image in open_mouth_faces:
        print('  {0}'.format(image))