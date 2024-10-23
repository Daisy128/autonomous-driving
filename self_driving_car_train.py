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
from utils import get_driving_styles
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

    x = None
    y = None
    path = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    column_name = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed', 'lap', 'sector', 'cte']
    all_steering_data = []

    # if we have multiple driving styles, like ["normal", "recovery", "reverse"]
    # the following for loop concatenate the three csv files into one '垂直堆叠rows'
    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.TRAINING_SET_DIR,
                                cfg.TRACK3_PATH,
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

            if cfg.ALL_DATA:
                y_center = data_df['steering'].values
                y_left = data_df['steering'].values + 0.02
                y_right = data_df['steering'].values - 0.02

                if x is None:
                    # 将 center, left, right 分别展平为一维数组并拼接
                    x = np.concatenate((data_df['center'].values, data_df['left'].values, data_df['right'].values), axis=0)
                    # 将 y_center, y_left, y_right 拼接成一维数组
                    y = np.concatenate((y_center, y_left, y_right), axis=0)
                    print(f"x shape: {x.shape}")
                    print(f"y shape: {y.shape}")
                else:
                    new_x = np.concatenate((data_df['center'].values, data_df['left'].values, data_df['right'].values), axis=0)
                    new_y = np.concatenate((y_center, y_left, y_right), axis=0)
                    x = np.concatenate((x, new_x), axis=0)
                    y = np.concatenate((y, new_y), axis=0)
                    all_steering_data.extend(data_df['steering'].values)
            else:
                if x is None:
                    x = data_df[['center', 'left', 'right']].values
                    y = data_df['steering'].values
                else:
                    # similar to (x = x + 1), where x refers 'x' in the parenthesis, 1 refers 'data_df[['center', 'left', 'right']].values'
                    x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0) # axis=0用于将来自多个 CSV 文件的数据合并, 垂直堆叠数组，增加行数。
                    y = np.concatenate((y, data_df['steering'].values), axis=0)

            # Used for sampling
            

        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files.")
        exit()

    try:
        # 将 steering 数据转换为 DataFrame，便于采样
        if cfg.SAMPLE_DATA:
            sampled_indices = data_sampling(y)
            # 根据平衡后的 steering 数据重新生成 x 和 y
            #sampled_indices = sampled_df.index
            x_sampled = x[sampled_indices]
            y_sampled = y[sampled_indices]

            # debug
            # df_sampled = pd.DataFrame({'steering': y_sampled})
            # print_histogram(df_sampled, data_df)

            # 将数据集划分为训练集和测试集
            x_train, x_test, y_train, y_test = train_test_split(x_sampled, y_sampled, test_size=cfg.TEST_SIZE, random_state=0)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cfg.TEST_SIZE, random_state=0)
        
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
        
    # 在每个 epoch 结束后保存模型时，插入当前的 epoch 数字，确保文件名唯一。如，epoch=5 时生成文件名 "track1-dave2-mc-005.h5", 
    name = os.path.join(cfg.SDC_MODELS_DIR, # SDC_MODELS_DIR: self-driving car models
                        cfg.TRACK,
                        default_prefix_name + '-{epoch:03d}.h5') # .h5: HDF5
    
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
        track1_path = name = os.path.join(cfg.SDC_MODELS_DIR, 'track3', cfg.BASE_MODEL)
        model.load_weights(track1_path)

        # for layer in model.layers:
        #     if 'conv' in layer.name:
        #         layer.trainable = False

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=cfg.LEARNING_RATE))

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

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
