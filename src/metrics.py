from keras import backend as K

def recall_score(y_true, y_pred):
    """
    Recall score calculation
    :param y_true: real class
    :param y_pred: predicted class
    :return: score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def precision_score(y_true, y_pred):
    """
    Precision score calculation
    :param y_true: real class
    :param y_pred: predicted class
    :return: score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def f1_score(y_true, y_pred):
    """
    F1 score calculation as a geometric mean of precision and recall
    :param y_true: real class
    :param y_pred: predicted class
    :return: score
    """
    precision  = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ( (precision * recall) / (precision + recall + K.epsilon()) )