import os

def get_absolute_file_pathways(directory):
    file_names = os.listdir(directory)

    return [os.path.join(directory, file_name) for file_name in file_names]