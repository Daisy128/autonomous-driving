import os
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from utils import get_driving_styles

def data_sampling(y):
    
    # cfg = Config()
    # cfg.from_pyfile("config_my.py")
    # drive = get_driving_styles(cfg)

    # print("Loading training set " + str(cfg.TRACK) + str(drive))

    # all_steering_data = []

    # for drive_style in drive:
    #     path = os.path.join(cfg.TRAINING_DATA_DIR,
    #                         cfg.TRAINING_SET_DIR,
    #                         cfg.TRACK,
    #                         drive_style,
    #                         'driving_log.csv')
        
    #     data = pd.read_csv(path)
    #     steering_data = data['steering']
    #     all_steering_data.extend(steering_data)

    df = pd.DataFrame({'steering': y})

    df_part1 = df[abs(df['steering']) <= 0.045]
    df_part2 = df[(abs(df['steering']) > 0.045)]
    
    df_part1_sampled = df_part1.sample(frac=0.4, random_state=42, replace=len(df_part1) < 0.3 * len(df_part1))
    
    df_balanced = pd.concat([df_part1_sampled, df_part2])

    return df_balanced.index


def print_histogram(df_sampled, df):
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    axes[0].hist(df['steering'], bins=100, color='blue', edgecolor='black')
    axes[0].set_title('Original')
    axes[0].set_xlabel('Steering')
    axes[0].set_ylabel('Occurrences')

    axes[1].hist(df_sampled['steering'], bins=100, color='green', edgecolor='black')
    axes[1].set_title('Balanced')
    axes[1].set_xlabel('Steering')
    axes[1].set_ylabel('Occurrences')

    plt.tight_layout()

    plt.show()

