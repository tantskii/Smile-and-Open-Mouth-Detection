import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
warnings.filterwarnings('ignore')

import pandas as pd
from keras.optimizers import Adam
from metrics import f1_score
from utils import callbacks_factory
from .train_utils import train_valid_test_generators
from .model import create_mlp_model

def train(args):

    print('Create generators')
    generators = train_valid_test_generators(
        valid_proportion=args.valid_proportion,
        test_proportion=args.test_proportion,
        seed=args.seed,
        crop_shape=(args.height, args.width),
        batch_size=args.batch_size,
        shuffle=True
    )
    print('Create model')
    model = create_mlp_model(l2_reg=args.l2_reg, seed=args.seed)
    callbacks = callbacks_factory(
        callbacks_list=[
            'tensorboard',
            'best_model_checkpoint',
            'early_stopping',
            'learning_rate_scheduler'
        ],
        model_mask='mlp_multiclassification'
    )

    model.compile(
        optimizer=Adam(
            lr=args.optimizer_lr,
            decay=args.optimizer_decay
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
    print('Training model')
    model.fit_generator(
        generators['train_generator'],
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=generators['valid_generator'],
        verbose=1,
        workers=4,
        use_multiprocessing=False,
    )
    print('Save test evaluation')
    results = model.evaluate_generator(generators['test_generator'])
    pd.DataFrame({
        'MetricsNames': model.metrics_names,
        'Results': results
    }).to_csv('../logs/solution_3_test_generator_evaluation.csv', index=False)