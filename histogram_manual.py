import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from utils import get_driving_styles

different_package = False

cfg = Config()
cfg.from_pyfile("config_my.py")

drive = get_driving_styles(cfg)

if different_package:
    cfg.TRACK = "track3von1"

column_name = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed', 'lap', 'sector', 'cte']
    
print("Loading training set " + str(cfg.TRACK) + str(drive))

all_steering_data = []

for drive_style in drive:
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
    
    data = pd.read_csv(path)
    steering_data = data['steering']
    all_steering_data.extend(steering_data)

df = pd.DataFrame({'steering': all_steering_data})

df_part1 = df[abs(df['steering']) <= 0.045]
df_part2 = df[(abs(df['steering']) > 0.045)]
#df_part3_sampled = df_part3.sample(frac=2, replace=True, random_state=42)

df_part1_sampled = df_part1.sample(frac=0.6, random_state=0, replace=len(df_part1) < 0.3 * len(df_part1))

df_balanced = pd.concat([df_part1_sampled, df_part2])

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
axes[0].hist(df['steering'], bins=100, color='blue', edgecolor='black')
axes[0].set_title('Original')
axes[0].set_xlabel('Steering')
axes[0].set_ylabel('Occurrences')

axes[1].hist(df_balanced['steering'], bins=100, color='green', edgecolor='black')
axes[1].set_title('Balanced')
axes[1].set_xlabel('Steering')
axes[1].set_ylabel('Occurrences')

plt.tight_layout()

plt.show()
