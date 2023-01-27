# DSB evaluation functions
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# 计算 KL Divergence (Kullback-Leibler)
def asymmetricKL(P, Q):
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon
    P_sum = sum(P)
    Q_sum = sum(Q)
    P = P / P_sum
    Q = Q / Q_sum
    divergence = np.sum(P * np.log(P / Q))
    return divergence


def symmetricalKL(P, Q):
    """
    P = np.asarray([1.346112,1.337432,1.246655])
    Q = np.asarray([1.033836,1.082015,1.117323])
    print(asymmetricKL(P, Q), asymmetricKL(Q, P))
    print(symmetricalKL(P, Q))
    """
    return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00


# Evaluation
# Metrics: pearson & symmetricKL
def evaluation_repair_map(y_train, Y_pred):
    import numpy as np
    from sklearn.metrics import mean_squared_error
    evaluation_dict = {'pearson': [],
                       'spearman': [],
                       'symKL': [],
                       'MSE': []}
    sample_n = y_train.shape[0]
    for index in range(sample_n):
        temp_train = y_train[index, :]
        temp_pred = Y_pred[index, :]
        # pearson
        pccs = np.corrcoef(temp_train, temp_pred)[0, 1]
        evaluation_dict['pearson'].append(pccs)
        # spearman
        spear = pd.Series(temp_train).corr(pd.Series(temp_pred), method='spearman')
        evaluation_dict['spearman'].append(spear)
        # symmetricKL
        symKL = symmetricalKL(temp_train, temp_pred)
        evaluation_dict['symKL'].append(symKL)
        # MSE
        mse = mean_squared_error(temp_train, temp_pred)
        evaluation_dict['MSE'].append(mse)
    # DataFrame
    eval_df = pd.DataFrame(evaluation_dict)
    return eval_df


# 自定义损失函数
def my_categorical_crossentropy_2(labels, logits):
    import tensorflow as tf
    '''
    label = tf.constant([[0,0,1,0,0]], dtype=tf.float32)
    logit = tf.constant([[-1.2, 2.3, 4.1, 3.0, 1.4]], dtype=tf.float32)
    logits = tf.nn.softmax(logit) # 计算softmax
    my_result1 = my_categorical_cross_entropy(labels=label, logits=logits)
    my_result2 = my_categorical_crossentropy_1(label, logits)
    my_result3 = my_categorical_crossentropy_2(label, logits)
    my_result1, my_result2, my_result3
    '''
    return tf.keras.losses.categorical_crossentropy(labels, logits)


# evaluation: KL-Divergence, MSE, Pearson
def evaluation(model_path, Xdata, ydata):
    # load model
    from keras.models import load_model
    model = load_model(model_path, custom_objects={'my_categorical_crossentropy_2': my_categorical_crossentropy_2})
    # predict
    ypred = model.predict(Xdata)
    result = evaluation_repair_map(ydata, ypred)
    return result