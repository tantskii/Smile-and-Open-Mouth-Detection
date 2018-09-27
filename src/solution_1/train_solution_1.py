from metrics import f1_score
from keras.optimizers import Adam
from .model import create_mobilenetv2
from .train_utils import create_train_valid_generators, get_augmentations_pipeline, callbacks_factory

SEED = 147
VALID_PROPORTION = 0.1
BATCH_SIZE = 8


def train():
    augmentations_pipeline = get_augmentations_pipeline()
    callbacks = callbacks_factory(['best_model_checkpoint, early_stopping'])
    train_generator, valid_generator = create_train_valid_generators(augmentations_pipeline,
                                                                     batch_size=BATCH_SIZE,
                                                                     seed=SEED,
                                                                     valid_proportion=VALID_PROPORTION)
    model = create_mobilenetv2((None, None, 3))
    model.compile(optimizer=Adam(lr=0.001, decay=0.001, amsgrad=True),
                  loss={'smile_output': 'binary_crossentropy',
                        'open_mouth_output': 'binary_crossentropy'},
                  loss_weights={'smile_output': 0.5,
                                'open_mouth_output': 0.5},
                  metrics=[f1_score])
    model.fit_generator(train_generator,
                        epochs=10,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        verbose=1,
                        workers=4,
                        use_multiprocessing=False)


if __name__ == '__main__':
    train()



