# ゼロから作るDeep Learning Learning のサンプル実装を作ってみるリポジトリ

## ToC

- ch01. コードなし
- ch02. コードは `ch02.py` へ実装
- ch03, コードは `ch03.py` へ実装

## 免責

### requirements

ここのコードは以下の環境化で実装している

- hardware
    - M1 Mackbook Air (2020モデル)
- python
    - mini-forge MacOSX arm64版
    - `python -VV` の出力
      ```
      Python 3.9.7 | packaged by conda-forge | (default, Sep 29 2021, 19:22:19) [Clang 11.1.0 ]
      ```
    - activate は次のコマンド
      ```shell
      conda activate dl-env
      ```

# note

## ch1


## ch2

- パーセプトロンの実装
- ch02.py を参照のこと
- NODEというパーセプトロンの基本の箱にlambda関数で返却する数式を定義できるようにした


## ch3

- ch03.py を， ch02.py を利用して実装
- 活性化関数をプロットし可視化
    - step関数
    - sigmoid関数
    - ReLU関数
- 行列計算の動作確認

