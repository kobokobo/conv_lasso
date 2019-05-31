
更新日：2016年8月8日
更新者：Norimasa Kobori

---------------------------------------------------------------------

各フォルダごとに、コード＆データが完結しています。

■ica_ImgSeparation
画像の混合画像から，源画像を推定する。
ICAを用いて，源画像を推定する。
各コスト関数の比較を実施した。


■ica_FeatureExtraction_Figure
自然画像から基底画像を求める。
トポロジカルICA（プーリング層が入った2層式のICA）を用いて，
基底画像が近いものが集まるようにしている。
ガボールフィルタのような基底画像が類似したものほど近くに出る。


■ica_FeatureExtraction_Figure
MNISTの画像から，ICAを用いて基底画像を推定する問題。
優ガウス分布で実施，入力画像にノイズを入れると精度が上がる。


■mlp_ColorCalibration
色相を学習する多層パーセプトロン。NNはBP法。
基底となる色を学習させる。暗みのあるテスト画像が明るい正しい色に変換される。


■mlp_Classification_Figure
MNISTの画像を多層パーセプトロンで学習させる。NNはBP法。
コードは，mlp_ColorCalibrationから以下に差し替え。
https://github.com/davidstutz/matlab-mnist-two-layer-perceptron
（理由）
・mlp_ColorCalibrationでは，for文を繰り返すため糞遅い。Matrix演算に変換。
・mlp_ColorCalibrationでは，逐次学習のみで，バッチ学習していない。

　結果はLeCunの論文に近いが，やや低い。
　中間層が300層の場合，認識精度：92.89%
　中間層が700層の場合，認識精度：91.8%

---------------------------------------------------------------------

以上。
