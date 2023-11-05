import os
import argparse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_data(data_parent_dir, class_names):
    # クラス数
    num_classes = len(class_names)

    # 画像データとラベルを格納するリストを初期化
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for class_idx, class_name in enumerate(class_names):
        # トレーニングデータの読み込み
        class_folder = os.path.join(data_parent_dir, class_name, 'output/train')
        image_files = [f for f in os.listdir(class_folder) if f.endswith(".jpg")]

        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            img = load_img(image_path, target_size=(32, 32))
            img = img_to_array(img)
            img = img.astype('float32') / 255.0
            x_train.append(img)
            y_train.append(class_idx)

        # テストデータの読み込み
        class_folder = os.path.join(data_parent_dir, class_name, 'output/test')
        image_files = [f for f in os.listdir(class_folder) if f.endswith(".jpg")]

        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            img = load_img(image_path, target_size=(32, 32))
            img = img_to_array(img)
            img = img.astype('float32') / 255.0
            x_test.append(img)
            y_test.append(class_idx)

    # NumPy配列に変換
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # クラスラベルをカテゴリカル形式に変換
    y_train = to_categorical(y_train, num_classes)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


