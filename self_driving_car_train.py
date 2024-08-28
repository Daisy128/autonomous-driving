from datetime import timedelta, datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from config import Config
from self_driving_car_batch_generator import Generator
from utils import get_driving_styles
from utils_models import *

np.random.seed(0) # 0 means can be any number
# 随机种子控制随机数生成器的行为, 确保每次运行程序时，生成的随机数序列都是相同的。
# 这对于调试和结果比较非常重要，因为它消除了由于随机因素导致的结果差异。


def load_data(cfg):
    """
    Load training data_nominal and split it into training and validation set
    """
    drive = get_driving_styles(cfg)

    print("Loading training set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x = None
    y = None
    path = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    column_name = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed', 'lap', 'sector', 'cte']

    # if we have multiple driving styles, like ["normal", "recovery", "reverse"]
    # the following for loop concatenate the three csv files into one '垂直堆叠rows'
    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.TRAINING_SET_DIR,
                                cfg.TRACK,
                                drive_style,
                                'driving_log.csv')
            # 读取文件第一行
            with open(path, 'r') as f:
                reader = csv.reader(f)
                first_row = next(reader)
            # 设置列名
            if first_row == column_name:
                data_df = pd.read_csv(path)
            else:
                data_df = pd.read_csv(path, header=None) # 读取且不要将第一行当作列名
                data_df.columns = column_name
            data_df.to_csv(path, index=False)  # 省略额外的第一列indexing

            if x is None:
                x = data_df[['center', 'left', 'right']].values
                y = data_df['steering'].values
            else:
                # similar to (x = x + 1), where x refers 'x' in the parenthesis, 1 refers 'data_df[['center', 'left', 'right']].values'
                x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0) # axis=0用于将来自多个 CSV 文件的数据合并, 垂直堆叠数组，增加行数。
                y = np.concatenate((y, data_df['steering'].values), axis=0)
        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print("No driving data_nominal were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cfg.TEST_SIZE, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    duration_train = time.time() - start
    print("Loading training set completed in %s." % str(timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(x)) + " elements") # len(x) = number of rows of x
    print("Training set: " + str(len(x_train)) + " elements")
    print("Test set: " + str(len(x_test)) + " elements")
    return x_train, x_test, y_train, y_test


def train_model(model, cfg, x_train, x_test, y_train, y_test):
    """
    Train the self-driving car model
    """
    if cfg.USE_PREDICTIVE_UNCERTAINTY:
        default_prefix_name = os.path.join(cfg.TRACK + '-' + cfg.SDC_MODEL_NAME + '-mc')
    else:
        default_prefix_name = os.path.join(cfg.TRACK + '-' + cfg.SDC_MODEL_NAME)
        
    # 在每个 epoch 结束后保存模型时，插入当前的 epoch 数字，确保文件名唯一。如，epoch=5 时生成文件名 "track1-dave2-mc-005.h5", 
    name = os.path.join(cfg.SDC_MODELS_DIR, # SDC_MODELS_DIR: self-driving car models
                            default_prefix_name + '-{epoch:03d}.h5') # .h5: HDF5
    
    checkpoint = ModelCheckpoint(
        name,
        monitor='val_loss', # 每个 epoch 结束时，检查验证损失是否为最优。
        verbose=0, # 0表示不输出详细信息; 可以设置为1来显示保存信息
        save_best_only=True, # 仅保存验证损失最小的模型, 避免过拟合
        mode='auto') # 自动选择保存模型的模式。当监控值是损失（如 val_loss）时，auto 模式会自动选择 min，表示越小越好

    early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=.0005, # 只有当损失的改善超过 min_delta 时，才会认为模型有显著进步
                                               patience=10, # 如果经过10个 epoch 后损失没有显著改善，训练停止
                                               mode='auto') # loss -> mode= 'min'
    
#    clear_memory = LambdaCallback(on_epoch_end=lambda epoch, logs: tf.keras.backend.clear_session())

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=cfg.LEARNING_RATE))

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    x_train, y_train = shuffle(x_train, y_train, random_state=0) # random_state=0 设置随机种子,确保了每次打乱的顺序是一致的，从而保证实验的可重复性
    x_test, y_test = shuffle(x_test, y_test, random_state=0) # test data?

    train_generator = Generator(x_train, y_train, True, cfg)
    val_generator = Generator(x_test, y_test, False, cfg)

    # model.fit开始训练模型
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=cfg.NUM_EPOCHS_SDC_MODEL,
                        callbacks=[checkpoint, early_stop, reduce_lr], #callback
                        verbose=1) # 输出详细信息

    # summarize history for loss
    # history.history: model.fit()返回的 History 对象中存储的字典，包含了模型在每个 epoch 中的训练和验证损失
    plt.plot(history.history['loss']) # 绘制训练损失
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plot_name = f'{default_prefix_name}_{current_time}.png'
    plot_path = os.path.join('history', 'loss_plot', plot_name)
    plt.savefig(plot_path)
    plt.show()
    
    # store the data into history.csv
    hist_df = pd.DataFrame(history.history)
    hist_df['time'] = current_time
    hist_df['plot'] = plot_name
    
    hist_csv_file = os.path.join('history', 
                                 default_prefix_name + '-history.csv')
        
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f, index=False)

    # store the trained model into .h5 file
    name = os.path.join(cfg.SDC_MODELS_DIR,
                        default_prefix_name + '-final.h5')
   
    # save the last model anyway (might not be the best)
    model.save(name)
    tf.keras.backend.clear_session()


def main():
    """
    Load train/validation data_nominal set and train the model
    """
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    x_train, x_test, y_train, y_test = load_data(cfg)

    model = build_model(cfg.SDC_MODEL_NAME, cfg.USE_PREDICTIVE_UNCERTAINTY)

    train_model(model, cfg, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
