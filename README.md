# 前書き
SSDのkeras実装を試してみる。  
パクってきたのは。[Pierluigi Ferrari さんのSSD](https://github.com/pierluigiferrari/ssd_keras)  
[Andrey Rykov さんのSSD](https://github.com/rykov8/ssd_keras)が有名っぽいけど。  
理由は「なんとなく」。  
というのは冗談で、後段のNMS処理まで含めてモデル化されていて、pythonプログラム部分がシンプルになりそうだから。  

# 事前準備

オリジナルはTensorflow1.x + kerasを使用するが、これをTensorflow2対応にしてみた。

## 仮想環境の構築
```bash
pyenv virtualenv 3.8.9 keras_tf2
pyenv local keras_tf2 
pip install --upgrade pip setuptools
```
> ちなみに、tensorfolw1.15を使う場合はpythonは3.7以下で。  
> 3.8だと、pipに「そんなモジュールはない」と怒られる。  
> これは、pypiのサーバにpython3.8対応のtensorfolw1.15のファイルがupされていないため。  


# モジュール類のインストール
```bash
sudo apt install libxml2-dev libxmlsec1-dev
pip install tensorflow
pip install matplotlib 
pip install opencv-python
pip install imageio
pip install scikit-learn
pip install tqdm
pip install beautifulsoup4 
pip install lxml
```

# ssd_kerasリポジトリのclone と パッチ適用
```bash
git clone https://github.com/pierluigiferrari/ssd_keras.git
cd ssd_keras/
patch -p1 < ../ssd_keras_1.patch            # Tensorflow1.15対応パッチ
patch -p1 < ../ssd_keras_2.patch            # Tensorflow2.5対応パッチ
cd ..
```

# 学習済みモデルのダウンロード
[ssd_kerasのリポジトリのREADME.mdの「Download the original trained model weights」](https://github.com/pierluigiferrari/ssd_keras/blob/master/README.md#download-the-original-trained-model-weights)
に書かれた  
「1.PASCAL VOC models」の「07+12」の「SSD300*」または「SSD512*」をダウンロード  
リンク先を再掲(SSD300)：<https://drive.google.com/open?id=121-kCXaOHOkJE_Kf5lKcJvC_5q1fYb_q>  
リンク先を再掲(SSD512)：<https://drive.google.com/open?id=19NIa0baRCFYT3iRxQkOKCD7CpN6BFO8p>  
ダウンロードしたファイルを``weights``ディレクトリに保存しておく。
> Googleドライブ上の巨大なファイルなので、wgetでさくっとダウンロードできない。  
> どうしてもwgetでやりたかったら<https://qiita.com/cedar0912/items/3e0fbb0291e63317ad9e>あたりを参考にどうぞ。  

# テスト用画像のダウンロード
実際の座標と認識結果の差分を確認するために、[Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)のテストデータを使用しているため、ダウンロードしておく。  
適当なディレクトリで以下を実行
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
```

> このダウンロードするのが面倒なら、やらなくても可。  
> その場合は、下記スクリプト内の「通常の認識処理はここまで」と書かれた行以降をコメントアウトする。  

# サンプル実行

## 設定値変更
``test/test_ssdXXX.py``の以下の変数の設定を↑でテスト画像をダウンロードしたディレクトリに書き換え
- VOC_2007_images_dir
- VOC_2007_images_dir

ソース中では``/mnt/m/Tensorflow_data/dataset/``となっている部分を変更する。  

## 実行
```bash
cd test
python test_ssd512.py
# or
python test_ssd300.py
```

## モデル情報付きh5ファイルをロードする実行
上記スクリプトを1回実行すると、モデル情報付きh5ファイルとTensorflowのsaved_modelを保存する。  
モデル情報付きh5ファイルを使用して処理を実行するには、同じスクリプトに適当なパラメータを付けて実行する。  
(パラメータの数だけでチェックしているので内容は何でも良い。下の例では「1」)  

> モデル付きでセーブしてあっても、カスタムレイヤの定義は必要なのであんまりうれしくないかも...

```bash
python test_ssd512.py 1
# or
python test_ssd300.py 1
```

## saved_modelをロードする実行
同様に保存されたsaved_modelを使用して処理を実行するには、ファイル名に``_saved_model``が付いたスクリプトを実行する。  
こちらのスクリプトは簡単にするため、後半部分を省略してある。  
```bash
python test_ssd512_saved_model.py
# or
python test_ssd300_saved_model.py
```

> saved_modelからロードするときはカスタムレイヤの定義も不要。  
> ディレクトリ内の ``keras_metadata.pb`` に情報があるのかな？  
