import argparse

from solution_1 import predict as predict_solution_1
from solution_2 import predict as predict_solution_2
from solution_3 import predict as predict_solution_3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--solution', type=int, default=1)
    arg('--images_directory', default='../data_test/example_data/example_data/images')
    arg('--height', type=int, default=192)
    arg('--width', type=int, default=192)
    arg('--smile_prediction_threshold', type=float, default=0.955)
    arg('--mouth_open_prediction_threshold', type=float, default=0.5)

    args = parser.parse_args()

    if args.solution == 1:
        predict_solution_1(args)

    elif args.solution == 2:
        predict_solution_2(args)

    elif args.solution == 3:
        predict_solution_3(args)