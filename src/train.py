import argparse

from solution_1 import train as train_solution_1
from solution_2 import train as train_solution_2
from solution_3 import train as train_solution_3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--solution', type=int, default=1)
    arg('--valid_proportion', type=float, default=0.1)
    arg('--test_proportion', type=float, default=0.1)
    arg('--batch_size', type=int, default=32)
    arg('--height', type=int, default=192)
    arg('--width', type=int, default=192)
    arg('--seed', type=int, default=147)
    arg('--l2_reg', type=float, default=0.001)
    arg('--optimizer_lr', type=float, default=0.001)
    arg('--optimizer_decay', type=float, default=0.001)
    arg('--epochs', type=int, default=6)

    arg('--alpha', type=float, default=1.)
    arg('--depth_multiplier', type=int, default=1)

    arg('--mouth_aspect_ratio_treshold', type=float, default=None)
    arg('--smile_deviations_sum_threshold', type=float, default=None)

    args = parser.parse_args()

    if args.solution == 1:
        train_solution_1(args)
    elif args.solution == 2:
        train_solution_2(args)
    elif args.solution == 3:
        train_solution_3(args)