import os
import time
import csv

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from datetime import timedelta, datetime
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from config import Config
from utils_models import *
from histogram_vis import *
from utils import get_driving_styles, get_driving_path
from self_driving_car_batch_generator import Generator

np.random.seed(0) # 0 means can be any number
# 随机种子控制随机数生成器的行为, 消除由于随机因素导致的结果差异.确保每次运行程序时，生成的随机数序列都是相同的。

def load_data(cfg):
    """
    Load training data_nominal and split it into training and validation set
    """
    drive = get_driving_styles(cfg)

    print("Loading training set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x_list = []
    y_steering_list = []
    y_throttle_list = []
    column_name = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed', 'lap', 'sector', 'cte']

    # if we have multiple driving styles, like ["normal", "recovery", "reverse"]
    # the following for loop concatenate the three csv files into one '垂直堆叠rows'
    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.TRAINING_SET_DIR,
                                f"{get_driving_path(cfg)}",
                                drive_style,
                                'driving_log.csv')
            
            data_df = pd.read_csv(path, header=0)
            if list(data_df.columns) != column_name:
                data_df.columns = column_name

            y_throttle_center = data_df['throttle'].values
            y_throttle_left = y_throttle_center / 1.2   # adjust the speed for manual input
            y_throttle_right = y_throttle_center / 1.2

            y_center = data_df['steering'].values
            y_left = y_center + 0.02   # cannot be greater than 0.1
            y_right = y_center - 0.02

            new_x = np.concatenate([data_df['center'].values, data_df['left'].values, data_df['right'].values])
            new_y_steering = np.concatenate([y_center, y_left, y_right])
            new_y_throttle = np.concatenate([y_throttle_center, y_throttle_left, y_throttle_right])

            x_list.append(new_x)
            y_steering_list.append(new_y_steering)
            y_throttle_list.append(new_y_throttle)
            
        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if not x_list:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files.")
        exit()

    x = np.concatenate(x_list, axis=0)
    y_steering = np.concatenate(y_steering_list, axis=0).reshape(-1, 1)  # dimention: 1*N -> N*1
    y_throttle = np.concatenate(y_throttle_list, axis=0).reshape(-1, 1)

    # Now concatenate along axis=1 to stack them side by side
    y = np.concatenate((y_steering, y_throttle), axis=1)
    
    try:

        # if cfg.SAMPLE_DATA:
        #     sampled_indices = data_sampling(y)
        #     x_sampled = x[sampled_indices]
        #     y_sampled = y[sampled_indices]

        #     # debug
        #     # df_sampled = pd.DataFrame({'steering': y_sampled})
        #     # print_histogram(df_sampled, data_df)

        x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=cfg.TEST_SIZE, random_state=0)
        
    except TypeError:
        print("Missing header to csv files")
        exit()

    duration_train = time.time() - start
    print("Loading training set completed in %s." % str(timedelta(seconds=round(duration_train))))

    print(f"Data set: {len(x)} elements")
    print(f"Training set: {len(x_train)} elements")
    print(f"Test set: {len(x_test)} elements")

    return x_train, x_test, y_train, y_test


def train_model(model, cfg, x_train, x_test, y_train, y_test):
    """
    Train the self-driving car model
    """

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    if cfg.USE_PREDICTIVE_UNCERTAINTY:
        default_prefix_name = os.path.join(cfg.TRACK + '-' + cfg.SDC_MODEL_NAME + '-mc')
    else:
        default_prefix_name = os.path.join(cfg.TRACK + '-' + cfg.SDC_MODEL_NAME)
        
    name = os.path.join(cfg.SDC_MODELS_DIR, # SDC_MODELS_DIR: "models"
                        cfg.TRACK,
                        default_prefix_name + '-{epoch:03d}.h5')
    
    checkpoint = ModelCheckpoint(
        name,
        monitor='val_loss', # 每个 epoch 结束时，检查验证损失是否为最优。
        verbose=0, # 0表示不输出详细信息; 可以设置为1来显示保存信息
        save_best_only=False, # 仅保存验证损失最小的模型 ?
        mode='auto') # 自动选择保存模型的模式。当监控值是损失（如 val_loss）时，auto 模式会自动选择 min，表示越小越好

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=.0001, # 只有当损失的改善超过 min_delta 时，才会认为模型有显著进步
                                               patience=10, # 如果经过10个 epoch 后损失没有显著改善，训练停止
                                               mode='auto') # loss -> mode= 'min'
    
#    clear_memory = LambdaCallback(on_epoch_end=lambda epoch, logs: tf.keras.backend.clear_session())

    if cfg.WITH_BASE:
        track_path = os.path.join(cfg.SDC_MODELS_DIR, cfg.BASE_MODEL)
        model.load_weights(track_path)

        # for layer in model.layers:
        #     if 'conv' in layer.name:
        #         layer.trainable = True

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=cfg.LEARNING_RATE))

    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    #x_train, y_train = shuffle(x_train, y_train, random_state=0) # random_state=0 设置随机种子,确保了每次打乱的顺序是一致的，从而保证实验的可重复性
    #x_test, y_test = shuffle(x_test, y_test, random_state=0) # test data?

    train_generator = Generator(x_train, y_train, cfg.USE_AUGMENT, cfg)
    val_generator = Generator(x_test, y_test, False, cfg) # False: not apply augmentation

    # model.fit开始训练模型
    with tf.device('/GPU:0'):
        history = model.fit(train_generator,
                            validation_data=val_generator,
                            epochs=cfg.NUM_EPOCHS_SDC_MODEL,
                            #callbacks=[checkpoint, early_stop, reduce_lr], # callback with reducing lr
                            callbacks=[checkpoint, early_stop], #callback
                            verbose=1) # 输出详细信息

    # summarize history for loss
    # history.history: model.fit()返回的 History 对象中存储的字典，包含了模型在每个 epoch 中的训练和验证损失
    plt.plot(history.history['loss']) # 绘制训练损失
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plot_name = f'{default_prefix_name}_{current_time}.png'
    plot_path = os.path.join('history', cfg.TRACK, 'loss_plot', plot_name)
    plt.savefig(plot_path)
    plt.show()
    
    # store the data into history.csv
    hist_df = pd.DataFrame(history.history)
    hist_df['time'] = current_time
    hist_df['plot'] = plot_name
    hist_df['description'] = (
                                f"data: {get_driving_styles(cfg)}, mc: {cfg.USE_PREDICTIVE_UNCERTAINTY}, "
                                f"aug: {cfg.USE_AUGMENT} with choose_image {cfg.AUG_CHOOSE_IMAGE}, "
                                f"random_flip: {cfg.AUG_RANDOM_FLIP}, random_translate: {cfg.AUG_RANDOM_TRANSLATE}, "
                                f"random_shadow: {cfg.AUG_RANDOM_SHADOW}, random_brightness: {cfg.AUG_RANDOM_BRIGHTNESS}"
                            )    # can be changed in each train, for detailed description
    
    hist_df.loc[1:, ['description']] = np.nan # put value only to the first row of the file

    hist_csv_file = os.path.join('history', cfg.TRACK, 
                                 default_prefix_name + '-' + current_time + '-history.csv')
        
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f, index=False)

    # store the trained model into .h5 file
    name = os.path.join(cfg.SDC_MODELS_DIR,
                        cfg.TRACK,
                        default_prefix_name + '-final.h5')
   
    # save the last model anyway (might not be the best)
    model.save(name)

    final_model = os.path.join(cfg.SDC_MODELS_DIR,
                               'final_model',
                               cfg.TRACK,
                               default_prefix_name + "-" + current_time + '-final.h5') # .h5: HDF5
    model.save(final_model)

    print(current_time)
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
