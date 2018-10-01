from keras.models import Model
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, BatchNormalization, Dropout
from keras.initializers import he_normal
from keras.regularizers import l2

def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == 'all':
        for layer in model.layers:
            layer.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, layer in enumerate(model.layers):
            if layer.name == freeze_before_layer:
                freeze_before_layer_index = i
        for layer in model.layers[:freeze_before_layer_index + 1]:
            layer.trainable = False

def unfreeze_model(model):
    for layer in model.layers:
        layer.trainable = True

def _create_branch(model_base_output, l2_reg, seed, name):
    branch = Dense(
        512,
        activation='relu',
        use_bias=True,
        kernel_initializer=he_normal(seed),
        kernel_regularizer=l2(l2_reg),
        bias_initializer=he_normal(seed),
        bias_regularizer=l2(l2_reg)
    ) (model_base_output)
    branch = BatchNormalization() (branch)
    branch = Dense(
        128,
        activation='relu',
        use_bias=True,
        kernel_initializer=he_normal(seed),
        kernel_regularizer=l2(l2_reg),
        bias_initializer=he_normal(seed),
        bias_regularizer=l2(l2_reg)
    ) (branch)
    branch = BatchNormalization() (branch)
    branch = Dense(
        1,
        activation='sigmoid',
        name=name
    ) (branch)

    return branch


def create_mobilenetv2(input_shape, alpha=1., depth_multiplier=1, l2_reg=0.001, seed=None):
    model_base = MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        include_top=False,
        weights='imagenet',
        pooling='max'
    )
    smile_branch = _create_branch(model_base.output, l2_reg, seed, name='smile_output')
    open_mouth_branch = _create_branch(model_base.output, l2_reg, seed, name='open_mouth_output')
    model = Model(model_base.input, [smile_branch, open_mouth_branch])

    return model