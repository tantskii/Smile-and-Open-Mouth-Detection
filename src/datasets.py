import os
import random
import pandas as pd
from itertools import chain
from operator import itemgetter
from utils import get_absolute_file_pathways

class TestDataset(object):
    """
    MULTI-PIE dataset
    """
    def __init__(self, dataset_dir):
        self._images_dir = os.path.join(dataset_dir, 'images')
        self._landmarks_dir = os.path.join(dataset_dir, 'landmarks')
        self._open_mouth_dir = os.path.join(dataset_dir, 'open_mouth')
        self._smile_dir = os.path.join(dataset_dir, 'smile')

        self._init_image_pathways()
        self._init_landmarks()
        self._init_smile_labels()
        self._init_open_mouth_labels()

    def _init_image_pathways(self):
        """
        Get absolute image pathways
        :return:
        """
        self.image_pathways = get_absolute_file_pathways(self._images_dir)

    def _init_landmarks(self):
        """
        Face landmarks preprocessing
        :return:
        """
        self.landmarks = list()

        for image_landmarks_pathway in get_absolute_file_pathways(self._landmarks_dir):
            with open(image_landmarks_pathway, 'r') as file_straem:
                string = file_straem.read()
                landmarks_coords = list(map(float, string.split('\n')[1].split(' ')[:-1]))
                landmarks_coords = [landmarks_coords[x : x+2] for x in range(0, len(landmarks_coords), 2)]
                self.landmarks.append(landmarks_coords)

    def _init_open_mouth_labels(self):
        """
        Open mouth labels preprocessing
        :return:
        """
        self.open_mouth_labels = self.__return_labels(self._open_mouth_dir)

    def _init_smile_labels(self):
        """
        Smile labels preprocessing
        :return:
        """
        self.smile_labels = self.__return_labels(self._smile_dir)

    def __return_labels(self, directory):
        """
        Convert directory image to labels
        :param directory:
        :return:
        """
        labels = list()
        target_image_names = os.listdir(directory)

        for image_name in os.listdir(self._images_dir):
            if image_name in target_image_names:
                labels.append(1)
            else:
                labels.append(0)

        return labels


class MTFLDataset(object):
    """
    Multi-Task Facial Landmarks dataset
    """
    def __init__(self,dataset_dir, AFLW_labels, net_labels):

        self._AFLW_images_dir = os.path.join(dataset_dir, 'AFLW')
        self._net_images_dir = os.path.join(dataset_dir, 'net_7876')

        self._training = os.path.join(dataset_dir, 'training.txt')
        self._testing = os.path.join(dataset_dir, 'testing.txt')

        self._AFLW_open_mouth_labels = pd.read_csv(AFLW_labels)
        self._net_open_mouth_labels = pd.read_csv(net_labels)

        self._init_image_pathways()
        self._init_smile_labels()
        self._init_open_mouth_labels()

    def shuffle(self, seed=None):
        """
        Shuffle the dataset
        :param seed: random state
        :return:
        """
        pathways_with_labels = list(
            zip(self.image_pathways, self.smile_labels, self.open_mouth_labels))

        random.Random(seed).shuffle(pathways_with_labels)
        self.image_pathways, self.smile_labels, self.open_mouth_labels = map(list,zip(*pathways_with_labels))


    def _init_image_pathways(self):
        """
        Get absolute image pathways which has labels in txt files
        :return:
        """
        image_pathways = get_absolute_file_pathways(self._AFLW_images_dir) + get_absolute_file_pathways(self._net_images_dir)
        image_names = [image_pathway.split('\\')[1] for image_pathway in image_pathways]

        training_image_names = list(self.__parse_txt(self._training).keys())
        testing_image_names = list(self.__parse_txt(self._testing).keys())

        correct_image_pathways_indices = list()
        for i in range(len(image_names)):
            if image_names[i] in training_image_names or image_names[i] in testing_image_names:
                correct_image_pathways_indices.append(i)

        self.image_pathways = itemgetter(*correct_image_pathways_indices)(image_pathways)

    def _init_smile_labels(self):
        """
        Smile labels preprocessing
        :return:
        """
        training_image_labels = self.__parse_txt(self._training)
        testing_image_labels = self.__parse_txt(self._testing)
        image_labels = dict(chain.from_iterable(d.items() for d in (training_image_labels, testing_image_labels)))

        self.smile_labels = self.__return_labels(image_labels)

    def _init_open_mouth_labels(self):
        """
        Open mouth labels preprocessing
        :return:
        """
        open_mouth_labels = pd.concat([self._AFLW_open_mouth_labels, self._net_open_mouth_labels])
        open_mouth_labels['Label'] = open_mouth_labels['Label'].map({1: 1, 2: 0})
        image_labels = pd.Series(open_mouth_labels['Label'].values,
                                 index=open_mouth_labels['ImageName']).to_dict()

        self.open_mouth_labels = self.__return_labels(image_labels)

    def __return_labels(self, image_labels):
        """
        Convert to labels
        :param image_labels: labels
        :return: labels
        """
        labels = list()

        for image_pathway in self.image_pathways:
            image_name = image_pathway.split('\\')[-1]
            labels.append(image_labels[image_name])

        return labels

    def __parse_txt(self, pathway):
        """
        Parse txt file with data
        :param pathway: pathway to txt file
        :return: labels
        """
        image_labels = dict()

        with open(pathway, 'r') as file_stream:
            for line in file_stream:
                if line != ' ':
                    splitter = '/' if '/' in line else '\\'
                    line = line.split(splitter)[1].split(' ')
                    image_name = line[0]
                    label = 0 if line[12] == '2' else int(line[12])

                    image_labels[image_name] = label

        return image_labels