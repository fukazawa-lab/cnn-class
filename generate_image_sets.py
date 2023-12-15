import os
import random
import shutil
import argparse  # argparse ライブラリを追加
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

from PIL import Image

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("--class_name", required=True, help="Class name for data processing")
args = parser.parse_args()

class_name = args.class_name  # コマンドライン引数から class_name を受け取る

if ' ' in class_name:
    print("■■■クラス名にスペースが入っています！スペースを削除してクラス名を再度定義しなおしてください")
else:
    # オリジナルのデータセットと新しいデータセットのフォルダを設定
    source_folder = "cnn-class/image_resource/"+class_name+"/"  # 犬の画像がまとまっているフォルダ
    output_folder = "cnn-class/image_resource/"+class_name+"/output/"  # 新しいデータセットのフォルダ
    
    
    # フォルダ内の画像ファイルをリストアップ
    image_files = [f for f in os.listdir(source_folder) if f.endswith(".jpg")]
    
    # ラベルの数を初期化
    label = 0
    
    # データセットを作成
    for image_file in image_files:
        # 画像ファイルのパス
        source_path = source_folder+ image_file
        
        # 新しいファイル名を生成（CIFAR-10形式）
        new_filename = f"{class_name}_{label:04d}.jpg"
        
        # 新しいファイルの保存先パス
        target_path = "cnn-class/image_resource/"+class_name+"/output/train/"+new_filename
        
        # 画像ファイルのコピーとリネーム
        shutil.copy(source_path, target_path)
        
        # 次のラベルに進む
        label += 1
    
    # データセットの分割（トレーニング、検証、テストセット）
    images = os.listdir("cnn-class/image_resource/"+class_name+"/output/train/")
    train_images, test_images = train_test_split(images, test_size=0.4, random_state=42)
    #val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)
    
    # 検証データとテストデータを移動
    for image in test_images:
      shutil.move("cnn-class/image_resource/"+class_name+"/output/train/"+image, "cnn-class/image_resource/"+class_name+"/output/test/"+ image)
    
    for image in test_images:
      shutil.move("cnn-class/image_resource/"+class_name+"/output/train/"+image, "cnn-class/image_resource/"+class_name+"/output/validation/"+image)
    
