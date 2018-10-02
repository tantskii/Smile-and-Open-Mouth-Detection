import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
warnings.filterwarnings('ignore')

import pandas as pd
from metrics import f1_score
from keras.optimizers import Adam
from utils import callbacks_factory
from .model import create_mobilenetv2, freeze_model, unfreeze_model
from .train_utils import train_valid_test_generators

def train_pipeline(model, train_generator, valid_generator, callbacks, optimizer_lr, optimizer_decay, epochs):
    model.compile(
        optimizer=Adam(
            lr=optimizer_lr,
            decay=optimizer_decay
        ),
        loss={
            'smile_output': 'binary_crossentropy',
            'open_mouth_output': 'binary_crossentropy'
        },
        loss_weights={
            'smile_output': 0.5,
            'open_mouth_output': 0.5
        },
        metrics=[f1_score]
    )
    model.fit_generator(
        train_generator,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=valid_generator,
        verbose=1,
        workers=4,
        use_multiprocessing=False,
    )

    return model

def train(args):

    print('Create generators')
    generators = train_valid_test_generators(
        valid_proportion=args.valid_proportion,
        test_proportion=args.test_proportion,
        seed=args.seed,
        shape=(args.height, args.width),
        batch_size=args.batch_size,
        shuffle=True
    )
    print('Create model')
    model = create_mobilenetv2(
        input_shape=(args.height, args.width, 3),
        alpha=args.alpha,
        depth_multiplier=args.depth_multiplier,
        l2_reg=args.l2_reg,
        seed=args.seed
    )

    print('Training freezed model')
    freeze_model(model, 'global_max_pooling2d_1')
    callbacks = callbacks_factory(
        callbacks_list=[
            'early_stopping',
            'tensorboard',
        ],
        model_mask='mobilenetv2_multiclassification_freezed'
    )
    model = train_pipeline(
        model,
        generators['hard_train_generator'],
        generators['valid_generator'],
        callbacks,
        optimizer_lr=args.optimizer_lr,
        optimizer_decay=args.optimizer_decay,
        epochs=args.epochs
    )

    print('Training unfreezed model')
    unfreeze_model(model)
    callbacks = callbacks_factory(
        callbacks_list=[
            'best_model_checkpoint',
            'early_stopping',
            'tensorboard',
            'learning_rate_scheduler'
        ],
        model_mask='mobilenetv2_multiclassification'
    )
    model = train_pipeline(
        model,
        generators['easy_train_generator'],
        generators['valid_generator'],
        callbacks,
        optimizer_lr=args.optimizer_lr,
        optimizer_decay=args.optimizer_decay,
        epochs=3 * args.epochs
    )

    print('Save test evaluation')
    results = model.evaluate_generator(generators['test_generator'])
    pd.DataFrame({
        'MetricsNames': model.metrics_names,
        'Results': results
    }).to_csv(os.path.join('../logs/solution_1_test_generator_evaluation.csv'), index=False)