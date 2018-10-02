import json
import warnings
import pandas as pd
from sklearn.metrics import f1_score
from datasets import MTFLDataset
from .heuristic_models import HeuristicMouthStateDetector, HeuristicSmileDetector
warnings.filterwarnings('ignore')

def get_train_test(test_proportion, seed):
    """
    Simple holdout creation
    :param test_proportion: fraction of test
    :param seed: random state
    :return: dict of train and test datasets
    """
    mtfl_dataset = MTFLDataset(
        '../data/MTFL/',
        '../data/AFLW.csv',
        '../data/net.csv'
    )
    mtfl_dataset.shuffle(seed)
    test_len = int(len(mtfl_dataset.image_pathways) * test_proportion)

    train_test_datasets = {
        'train_image_pathways': mtfl_dataset.image_pathways[: -test_len],
        'train_open_mouth_labels': mtfl_dataset.open_mouth_labels[: -test_len],
        'train_smile_labels': mtfl_dataset.smile_labels[: -test_len],
        'test_image_pathways': mtfl_dataset.image_pathways[-test_len :],
        'test_open_mouth_labels': mtfl_dataset.open_mouth_labels[-test_len :],
        'test_smile_labels': mtfl_dataset.smile_labels[-test_len :]
    }

    return train_test_datasets

def train(args):
    """
    Finding the optimal threshold for each of the two models on the training dataset
    :param args: argparse arguments
    :return: save test evaluation results in logs
    """
    print('Create datasets')
    train_test_datasets = get_train_test(args.test_proportion, args.seed)

    print('Train HeuristicMouthStateDetector')
    heuristic_mouth_state_detector = HeuristicMouthStateDetector(args.mouth_aspect_ratio_treshold)
    heuristic_mouth_state_detector.fit(
        image_pathways=train_test_datasets['train_image_pathways'],
        mouth_open_labels=train_test_datasets['train_open_mouth_labels']
    )
    mouth_open_predictions = heuristic_mouth_state_detector.predict(train_test_datasets['test_image_pathways'])

    print('Train HeuristicSmileDetector')
    heuristic_smile_detector = HeuristicSmileDetector(args.smile_deviations_sum_threshold)
    heuristic_smile_detector.fit(
        image_pathways=train_test_datasets['train_image_pathways'],
        smile_labels=train_test_datasets['train_smile_labels']
    )
    smile_predictions = heuristic_smile_detector.predict(train_test_datasets['test_image_pathways'])

    print('Save test evaluation')
    pd.DataFrame({
        'MetricsNames': ['smile_f1_score', 'mouth_open_f1_score'],
        'Results': [
            f1_score(train_test_datasets['test_smile_labels'], smile_predictions),
            f1_score(train_test_datasets['test_open_mouth_labels'], mouth_open_predictions),
        ]
    }).to_csv('../logs/solution_2_test_evaluation.csv', index=False)

    with open('../models/heuristic_models.json', 'w') as out_file:
        thresholds = {
            'mouth_aspect_ratio_threshold': heuristic_mouth_state_detector.mouth_aspect_ratio_threshold,
            'smile_deviations_sum_threshold': heuristic_smile_detector.smile_deviations_sum_threshold
        }
        json.dump(thresholds, out_file)