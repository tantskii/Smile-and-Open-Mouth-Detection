from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.initializers import he_normal
from keras.regularizers import l2

def _create_branch(model_base_output, l2_reg, seed, name):
    """
    Сreating one of two branches for multitask classification after base fully connected layers
    :param model_base_output: output after base layers
    :param l2_reg: float l2 regularization
    :param seed: random state
    :param name: name of the branch ouput
    :return: keras layers
    """
    branch = Dense(
        30,
        activation='relu',
        use_bias=True,
        kernel_initializer=he_normal(seed),
        kernel_regularizer=l2(l2_reg),
        bias_initializer=he_normal(seed),
        bias_regularizer=l2(l2_reg)
    ) (model_base_output)
    branch = BatchNormalization() (branch)
    branch = Dense(
        15,
        activation='relu',
        use_bias=True,
        kernel_initializer=he_normal(seed),
        kernel_regularizer=l2(l2_reg),
        bias_initializer=he_normal(seed),
        bias_regularizer=l2(l2_reg)
    ) (branch)
    branch = BatchNormalization()(branch)
    branch = Dense(1,activation='sigmoid',name=name) (branch)

    return branch

def create_mlp_model(l2_reg=0.001, seed=None):
    """
    Multi layer perceptron creation with two outputs
    :param input_shape: input image shape
    :param alpha: mobilenet width (channels) multiplier
    :param depth_multiplier: mobilenet depth (height and width of feature maps) multiplie
    :param l2_reg: l2 regularization
    :param seed: random state
    :return: keras model
    """
    input = Input(shape=(84,))
    fc1 = Dense(
        60,
        activation='relu',
        use_bias=True,
        kernel_initializer=he_normal(seed),
        kernel_regularizer=l2(l2_reg),
        bias_initializer=he_normal(seed),
        bias_regularizer=l2(l2_reg)
    ) (input)
    bn1 = BatchNormalization() (fc1)

    smile_branch = _create_branch(bn1, l2_reg, seed,name='smile_output')
    open_mouth_branch = _create_branch(bn1, l2_reg, seed,name='open_mouth_output')
    model = Model(input, [smile_branch, open_mouth_branch])

    return model