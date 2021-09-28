import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input

warnings.filterwarnings('ignore')

def tf_model(input_shape, optimizer, metric, loss, lr):
    if 'Adam':
        optimizer = Adam(lr)
    elif 'SGD':
        optimizer = SGD(lr)
    else:
        optimizer = RMSprop(lr)

    if 'rmse' in metric:
        idx = metric.index('rmse')
        metric[idx] = RootMeanSquaredError()

    if 'rmse' in loss:
        idx = metric.index('rmse')
        loss[idx] = RootMeanSquaredError()

    inputs = Input(shape=input_shape, name='input')

    x = Dense(32, activation='relu', name='dense_layer1')(inputs)
    x = Dense(32, activation='relu', name='dense_layer2')(x)

    outputs = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='ANN_model')

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metric
    )

    model.summary()
    return model


def preprocessing(df, lags, col):
    df = df.set_index('date')
    lag_cols = np.array(
        [col+f' t-{lag}' if lag > 0 else col for lag in range(lags + 1)])
    X_train = df[col].copy()
    X_train = pd.concat([X_train.shift(lag) for lag in range(lags + 1)], axis=1)
    X_train.columns = lag_cols
    X_train = X_train[X_train[f'{col} t-{lags}'].notna()]
    return X_train


def train_model(df_train, params):
    X_train, X_val = train_test_split(
        df_train, test_size=params['val_size'], shuffle=False)

    y_train = X_train[X_train.columns[0]]
    y_val = X_val[X_val.columns[0]]

    X_train = X_train.drop(columns = [X_train.columns[0]])
    X_val = X_val.drop(columns = [X_val.columns[0]])

    save_path = './model/' + params['symbol'] + '_model_' \
        + params['col'] + '.h5'
    
    mc = ModelCheckpoint(save_path, monitor='val_mae', mode='min', 
        save_best_only=True, verbose=1)

    input_shape = (len(X_train.columns),)

    model = tf_model(input_shape, params['optimizer'],
        params['metric'], params['loss'], params['learning_rate'])

    tf.keras.utils.plot_model(model, to_file='./model/' 
        + params['symbol'] + '_model_architecture.png', 
        show_shapes=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        shuffle=False,
        callbacks=[mc],
        verbose=1
    )
        
    plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.plot(history.history[params['metric'][0]])
    plt.plot(history.history['val_' + params['metric'][0]])
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend([params['metric'][0], 'val_' + params['metric'][0]], loc='upper right')

    save_path = './graph/' + params['symbol'] + '_model_' \
        + params['col'] + '_loss.png'

    plt.savefig(save_path)
    

def eval_model(df_train, params):
    X_train, X_val = train_test_split(
        df_train, test_size=params['val_size'], shuffle=False)

    col = params['col']

    y_true = df_train[col]
    X_val = X_val.drop(columns=[col])

    path = './model/' + params['symbol'] + '_model_' \
        + params['col'] + '.h5'

    model = tf.keras.models.load_model(path)
    y_pred = model.predict(X_val)

    X_val = X_val.reset_index()
    X_val['date'] = pd.to_datetime(X_val['date'], format='%Y-%m-%d')

    plt.figure(figsize=(8, 4))
    plt.plot(y_true)
    plt.plot(X_val['date'], y_pred)
    plt.xlabel('date')
    plt.ylabel(f'{col} price')
    plt.legend(['real price', 'pred'], loc='upper right')

    save_path = './graph/' + params['symbol'] + '_model_' \
        + params['col'] + '_val.png'
    plt.savefig(save_path)


def forecast(model, df, days):
    lags = model.layers[0].input_shape[0][1]
    preds = []

    X_test = df.tail(lags)['close'].to_numpy()
    X_test = np.expand_dims(X_test, axis=0)

    for x in range(days):
        pred = model.predict(X_test)[0][0]
        preds.append(pred)
        X_test = np.delete(X_test, 0)
        X_test = np.append(X_test, pred)
        X_test = np.expand_dims(X_test, axis=0)

    return np.array(preds)
