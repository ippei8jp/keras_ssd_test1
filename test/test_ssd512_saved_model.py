#!/usr/bin/env python
# SSD512 saved_modelからの認識処理

# === 準備 ==============================================================
import sys
import os

# Jupyter環境か確認
in_jupyter = True
try :
    get_ipython().run_line_magic('matplotlib', 'inline')
except :
    # jupyter環境ではない
    in_jupyter = False

from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# === 設定 ==============================================================
# SavedModelのディレクトリ
SAVED_MODEL_DIR = "saved_model_ssd512"

# サンプル画像のパス
SAMPLE_IMAGE = '../ssd_keras/examples/fish_bike.jpg'

# 表示用カラーテーブル(枠描画色)
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
# クラス名(デフォルトと同じなので設定しなくてもいいけど...)
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# 入力イメージサイズ
img_height = 512
img_width = 512

# === モデルの読み込み ==============================================================
print('Loading model...')
model = tf.saved_model.load(SAVED_MODEL_DIR)
print('Done.')

# === 認識処理 ==============================================================
orig_images = []        # 表示用イメージ配列(バッチサイズ分)
input_images = []       # モデル入力用イメージ配列(バッチサイズ分)

# サンプルイメージ
img_path = SAMPLE_IMAGE

# 表示用イメージとして読み込み(imageioを使用)
orig_images.append(imread(img_path))

# モデル入力用イメージとして読み込み/リサイズ/ndarray化
img = image.load_img(img_path, target_size=(img_height, img_width))     # 読み込みとリサイズ
img = image.img_to_array(img)                                           # ndarray化
input_images.append(img)                                                # バッチサイズ分の配列
input_images = np.array(input_images)                                   # list→dnarray変換

# 認識処理本体
y_pred = model(input_images)

# y_pred[batch][number][classID, score, xmin, ymin, xmax, ymax]     numberはtop_kで指定した値 分出力される

# スコアの閾値
confidence_threshold = 0.5

# スコアが閾値を超えたものだけ取り出し
y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

# 結果の表示(配列)
print("Predicted boxes:")
print('   class    conf    xmin    ymin    xmax    ymax')
for a in y_pred_thresh[0] :
    print(f'{int(a[0]):8d}{a[1]:8.2f}{int(a[2]):8d}{int(a[3]):8d}{int(a[4]):8d}{int(a[5]):8d}')

# === 結果を画像として表示 ==============================================================


# 図の作成(サイズの単位はインチ)
if not in_jupyter :
    plt.figure(figsize=(10,6))
else :
    plt.figure(figsize=(20,12))

# 図にイメージを貼り付け
plt.imshow(orig_images[0])

'''
if not in_jupyter :
    # jupytereはそのまま表示されるけど、通常はpauzeで指定秒数表示
    plt.pause(2)            #  表示
'''

# 認識枠書き込みのためにcurrent axesを取得
current_axis = plt.gca()

# 認識枠書き込み
for box in y_pred_thresh[0]:            # すべての認識結果についてループ
    # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
    xmin = box[2] * orig_images[0].shape[1] / img_width                 # 左上
    ymin = box[3] * orig_images[0].shape[0] / img_height                # 左上
    xmax = box[4] * orig_images[0].shape[1] / img_width                 # 右下
    ymax = box[5] * orig_images[0].shape[0] / img_height                # 右下
    color = colors[int(box[0])]                                         # 表示色
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])           # クラス名
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))     # 認識枠描画
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})          # クラス名描画

if not in_jupyter :
    # jupytereはそのまま表示されるけど、通常はpauzeで指定秒数表示
    plt.pause(5)                # 表示
    plt.close()                 # 終了したら閉じる

print("==== DONE ====")
