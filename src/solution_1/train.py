import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import pandas as pd
from metrics import f1_score
from keras.optimizers import Adam
from .model import create_mobilenetv2, freeze_model, unfreeze_model
from .train_utils import train_valid_test_generators, callbacks_factory

SEED = 147
VALID_PROPORTION = 0.1
TEST_PROPORTION = 0.1
BATCH_SIZE = 32
HEIGHT = 192
WIDTH = 192
CHANNELS = 3

def train_pipeline(
        model,
        train_generator,
        valid_generator,
        callbacks,
        optimizer_lr,
        optimizer_decay,
        epochs):

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


def train():
    generators = train_valid_test_generators(
        valid_proportion=VALID_PROPORTION,
        test_proportion=TEST_PROPORTION,
        seed=SEED,
        shape=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    model = create_mobilenetv2(
        input_shape=(HEIGHT, WIDTH, CHANNELS),
        alpha=1.,
        depth_multiplier=1,
        l2_reg=0.001,
        seed=SEED
    )


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
        optimizer_lr=0.001,
        optimizer_decay=0.001,
        epochs=6
    )


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
        optimizer_lr=0.001,
        optimizer_decay=0.001,
        epochs=15
    )


    results = model.evaluate_generator(generators['test_generator'])
    pd.DataFrame({
        'MetricsNames': model.metrics_names,
        'Results': results
    }).to_csv('../logs/test_generator_evaluation.csv', index=False)

if __name__ == '__main__':
    train()