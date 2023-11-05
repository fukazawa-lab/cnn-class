import os
import random
import shutil
import argparse  # argparse ���C�u������ǉ�
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

from PIL import Image

# �R�}���h���C�������̐ݒ�
parser = argparse.ArgumentParser()
parser.add_argument("--class_name", required=True, help="Class name for data processing")
args = parser.parse_args()

class_name = args.class_name  # �R�}���h���C���������� class_name ���󂯎��


# �f�[�^�g���̊֐�
def load_and_preprocess_image(image_path):
    target_size=(224, 224)
    # �摜��ǂݍ��݁A�w�肵���T�C�Y�Ƀ��T�C�Y
    image_path = class_name+"/output/train/"
    img = load_img(image_path, target_size=target_size)
    # �摜�f�[�^��NumPy�z��ɕϊ�
    img_array = img_to_array(img)
    # �摜�f�[�^�����f���ɍ������`���őO����
    preprocessed_image = preprocess_input(img_array)
    return preprocessed_image

def save_augmented_image(image, save_path):
    # NumPy�z�񂩂�PIL Image�ɕϊ�
    augmented_img = Image.fromarray(image.astype('uint8'))
    # �摜���w��̃p�X�ɕۑ�
    augmented_img.save(save_path)

# �I���W�i���̃f�[�^�Z�b�g�ƐV�����f�[�^�Z�b�g�̃t�H���_��ݒ�
source_folder = class_name+"/"  # ���̉摜���܂Ƃ܂��Ă���t�H���_
output_folder = class_name+"/output/"  # �V�����f�[�^�Z�b�g�̃t�H���_

# �V�����f�[�^�Z�b�g�̃t�H���_���쐬
if not os.path.exists(class_name + "/output"):
    os.makedirs(class_name + "/output/train")
    os.makedirs(class_name + "/output/validation")
    os.makedirs(class_name + "/output/test")


# �t�H���_���̉摜�t�@�C�������X�g�A�b�v
image_files = [f for f in os.listdir(source_folder) if f.endswith(".jpg")]

# ���x���̐���������
label = 0

# �f�[�^�Z�b�g���쐬
for image_file in image_files:
    # �摜�t�@�C���̃p�X
    source_path = source_folder+ image_file
    
    # �V�����t�@�C�����𐶐��iCIFAR-10�`���j
    new_filename = f"{class_name}_{label:04d}.jpg"
    
    # �V�����t�@�C���̕ۑ���p�X
    target_path = class_name+"/output/train/"+new_filename
    
    # �摜�t�@�C���̃R�s�[�ƃ��l�[��
    shutil.copy(source_path, target_path)
    
    # ���̃��x���ɐi��
    label += 1

# �f�[�^�Z�b�g�̕����i�g���[�j���O�A���؁A�e�X�g�Z�b�g�j
images = os.listdir(class_name+"/output/train/")
train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)

# ���؃f�[�^�ƃe�X�g�f�[�^���ړ�
for image in val_images:
  shutil.move(class_name+"/output/train/"+image, class_name+"/output/validation/"+image)

for image in test_images:
  shutil.move(class_name+"/output/train/"+image, class_name+"/output/test/"+ image)
