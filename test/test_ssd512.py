#!/usr/bin/env python
# coding: utf-8
# オリジナルソース：ssd512_inference.ipynbをpythonにexportしたソース
# SSD512 VOCで学習済みモデル用チュートリアル

# === 準備 ==============================================================
import sys
import os

# ロードするモデル
LOAD_MODEL_AND_WEIGHT = False
if len(sys.argv) > 1 :          # メンドクサイからこんなんでいっか。
    # パラメータが何か指定されていたらモデル付き重みデータをロード
    LOAD_MODEL_AND_WEIGHT = True

# Jupyter環境か確認
in_jupyter = True
try :
    get_ipython().run_line_magic('matplotlib', 'inline')
except :
    # jupyter環境ではない
    in_jupyter = False

# ライブラリパスの追加
sys.path.append("../ssd_keras")

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# ssd_keras 内のモジュール
from keras_loss_function.keras_ssd_loss import SSDLoss                                  # 損失関数

from data_generator.object_detection_2d_data_generator import DataGenerator             # データジェネレータクラス
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels       # 色数変更関数
from data_generator.object_detection_2d_geometric_ops import Resize                     # リサイズ関数
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

if LOAD_MODEL_AND_WEIGHT :
    # これらのモデルメソッドを使うのでimportしておく
    from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
    from keras_layers.keras_layer_DecodeDetections import DecodeDetections
    from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
    from keras_layers.keras_layer_L2Normalization import L2Normalization
else :
    from models.keras_ssd512 import ssd_512

# === 設定 ==============================================================


# モデルファイルのパス
MODEL_PATH = "./VGG_VOC0712_SSD_512x512_iter_120000_with_model.h5"
WEIGHTS_PATH = '../weights/VGG_VOC0712_SSD_512x512_iter_120000.h5'      # h5ファイルのpath

# 保存するSavedModelのディレクトリ
SAVED_MODEL_DIR = "saved_model_ssd512"

# サンプル画像のパス
SAMPLE_IMAGE = '../ssd_keras/examples/fish_bike.jpg'

# イメージディレクトリ、アノテーションデータディレクトリ、データセットファイル(後半部分で使用)
VOC_2007_images_dir         = '/mnt/m/Tensorflow_data/dataset/VOCdevkit/VOC2007/JPEGImages/'
VOC_2007_annotations_dir    = '/mnt/m/Tensorflow_data/dataset/VOCdevkit/VOC2007/Annotations/'
# たくさんデータがあっても時間がかかるだけなので、適当にピックアップしたデータセットを使う
# VOC_2007_test_image_set_filename = '/mnt/m/Tensorflow_data/dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
VOC_2007_test_image_set_filename = './test.dataset'

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

# === Keras modelの生成 ==============================================================
# 最初にメモリをクリアしておく
K.clear_session() # Clear previous models from memory.

if LOAD_MODEL_AND_WEIGHT :
    # 損失関数は指定しないといけないらしい
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    
    # モデルデータ込みのモデルファイルのロード
    model = load_model(MODEL_PATH, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})
else :
    # モデルのロード
    model = ssd_512(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                   two_boxes_for_ar1=True,
                   steps=[8, 16, 32, 64, 128, 256, 512],
                   offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   clip_boxes=False,
                   variances=[0.1, 0.1, 0.2, 0.2],
                   normalize_coords=True,
                   subtract_mean=[123, 117, 104],
                   swap_channels=[2, 1, 0],
                   confidence_thresh=0.5,
                   iou_threshold=0.45,
                   top_k=200,
                   nms_max_output_size=400)
    
    # 重みデータのロード
    model.load_weights(WEIGHTS_PATH, by_name=True)
    
    # モデルのコンパイル
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)       # 最適化関数(keras.optimizers)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)                                  # 損失関数 (keras_loss_function/keras_ssd_loss.py)
    
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    
    # モデル付きでセーブする
    if not os.path.exists(MODEL_PATH) :  # 1回やったらOKなので既に存在してたらやらない
        # モデルのセーブ
        model.save(MODEL_PATH)
        
        # Tensorflow SavedModel形式で保存...はTF1ではエラーになる
        if int(tf.version.VERSION.split('.')[0]) != 1:
            model.save(SAVED_MODEL_DIR, save_format = "tf")
        

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
y_pred = model.predict(input_images)


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

# === 通常の認識処理はここまで ==============================================================

# === 教師データと認識結果の差分を見てみる ==============================================================

# 教師データが欲しいのでテスト用データベースを利用する

# データジェネレータを使用
dataset = DataGenerator()

# データセットファイルからXML解析してアノテーションデータを生成する(値はクラス内で保持)
dataset.parse_xml(images_dirs=[VOC_2007_images_dir],                            # 必須
                  image_set_filenames=[VOC_2007_test_image_set_filename],       # 必須
                  annotations_dirs=[VOC_2007_annotations_dir],                  # アノテーションデータがない場合は省略
                  classes=classes,                                              # デフォルト使用なら省略可
                  include_classes='all',                                        # 省略時は'all'
                  exclude_truncated=False,                                      # 重なった画像を除外するか？→除外しない
                  exclude_difficult=True,                                       # 難しい画像を除外するか？  →除外する
                  ret=False)                                                    # 戻り値として結果を返すか？→返さない

# GrayScale、ARGB画像をRGBに変換する関数
convert_to_3_channels = ConvertTo3Channels()

# リサイズ処理を行う関数
resize = Resize(height=img_height, width=img_width)

# ジェネレータを生成
generator = dataset.generate(batch_size=1,                                      # batch size → 1
                             # shuffle=True,                                      # シャッフル → する
                             shuffle=False,                                     # シャッフル → しない
                             transformations=[convert_to_3_channels,            # 変換関数
                                              resize],
                             returns={'processed_images',                       # 戻り値
                                      'filenames',
                                      'inverse_transform',
                                      'original_images',
                                      'original_labels'},
                             keep_images_without_gt=False)                      # 正解なしの画像を保持 → しない   デフォルトFalse

# ジェネレータから1レコード取得
for _ in range(dataset.dataset_size)  :     # データセットの数だけ回す(繰り返しデータが返ってくるので for ～ in generator としてはいけない)
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(generator)

    # ジェネレータの batch_size が 1 なので、indexは0しかない
    idx = 0
    
    # 使用画像の情報
    print(f"Image:{batch_filenames[idx]}")                  # ファイル名
    print("Ground truth boxes:")                            # 教師データ(Ground truth boxes)
    print('   class            xmin    ymin    xmax    ymax')
    for a in batch_original_labels[idx] :
        print(f'{a[0]:8d}        {a[1]:8d}{a[2]:8d}{a[3]:8d}{a[4]:8d}')
    
    
    # 認識処理本体
    y_pred = model.predict(batch_images)
    
    # スコアの閾値
    confidence_threshold = 0.5
    
    # スコアが閾値を超えたものだけ取り出し
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    
    # 認識結果をオリジナルイメージ(リサイズ前)用に変換する
    y_pred_thresh_inv = apply_inverse_transforms(y_pred_thresh, batch_inverse_transforms)
    
    # 結果の表示(配列)
    print("Predicted boxes:")
    print('   class    conf    xmin    ymin    xmax    ymax')
    for a in y_pred_thresh_inv[idx] :           # オリジナルイメージ(リサイズ前)用に変換した結果
        print(f'{int(a[0]):8d}{a[1]:8.2f}{int(a[2]):8d}{int(a[3]):8d}{int(a[4]):8d}{int(a[5]):8d}')
    
    # 図の作成(サイズの単位はインチ)
    if not in_jupyter :
        plt.figure(figsize=(10,6))
    else :
        plt.figure(figsize=(20,12))
    
    # 図にイメージを貼り付け
    plt.imshow(batch_original_images[idx])
    '''
    if not in_jupyter :
        plt.pause(2)                # 表示
    '''
    
    # 認識枠書き込みのためにcurrent axesを取得
    current_axis = plt.gca()
    
    # 教師データの枠を描画(緑色)
    for box in batch_original_labels[idx]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
    '''
    if not in_jupyter :
        plt.pause(2)                # 表示
    '''
    # 認識結果の枠を描画(各色)
    for box in y_pred_thresh_inv[idx]:
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    
    if not in_jupyter :
        plt.pause(5)                # 表示
        plt.close()                 # 終了したら閉じる
    
print("==== DONE ====")
