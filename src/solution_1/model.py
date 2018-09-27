from keras.models import Model, Input
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, BatchNormalization
from keras.initializers import he_normal
from keras.regularizers import l2


def create_branch(model_base_output, l2_reg, seed):
    branch = Dense(512,
                   activation='relu',
                   use_bias=False,
                   kernel_initializer=he_normal(seed),
                   kernel_regularizer=l2(l2_reg)) (model_base_output)
    branch = BatchNormalization() (branch)
    branch = Dense(128,
                   activation='relu',
                   use_bias=False,
                   kernel_initializer=he_normal(seed),
                   kernel_regularizer=l2(l2_reg)) (branch)
    branch = BatchNormalization() (branch)
    branch = Dense(1, activation='sigmoid') (branch)

    return branch


def create_mobilenetv2(input_shape, alpha=1., depth_multiplier=1, l2_reg=0.001, seed=147):
    model_base = MobileNetV2(input_shape=input_shape,
                             alpha=alpha,
                             depth_multiplier=depth_multiplier,
                             include_top=False,
                             weights='imagenet',
                             pooling='max')

    smile_branch = create_branch(model_base.output, l2_reg, seed)
    open_mouth_branch = create_branch(model_base.output, l2_reg, seed)
    model = Model(model_base.input, [smile_branch, open_mouth_branch])

    return model