import os
from scipy.misc import imresize

def get_absolute_file_pathways(directory):
    file_names = os.listdir(directory)

    return [os.path.join(directory, file_name) for file_name in file_names]

def crop_image(image, bboxes, bbox_number=0, padding=0):
    crop_image = image[
                 bboxes[bbox_number][1]-padding : bboxes[bbox_number][1]+bboxes[bbox_number][3]+padding,
                 bboxes[bbox_number][0]-padding : bboxes[bbox_number][0]+bboxes[bbox_number][2]+padding,
                 :]

    return crop_image

def imresize_with_proportion(image, target_height):
    fraction = target_height / image.shape[0]

    return imresize(image, fraction)