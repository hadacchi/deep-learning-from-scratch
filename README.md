# ゼロから作るDeep Learning Learning のサンプル実装を作ってみるリポジトリ

## ToC

- ch01. コードなし
- ch02. コードは `ch02.py` へ実装
- ch03, 3.1 〜 3.3 のコードは `ch03_1.py` へ実装
  3.4 〜 3.7 のコードは `ch03_2.py` へ実装

## 免責

### requirements

ここのコードは以下の環境化で実装している

- hardware
    - M1 Mackbook Air (2020モデル)
- python
    - `poetry -V` の出力
        ```
        (main) $ poetry -V
        Poetry version 1.1.12
        ```
    - `poetry run python -VV` の出力
        ```
        (main) $ poetry run python -VV
        Python 3.10.1 (main, Dec  6 2021, 22:18:20) [Clang 13.0.0 (clang-1300.0.29.3)]
        ```
    - 実行は，poetry で `poetry run python ~~~`
    - `alias po`
        ```po='poetry run'```
    - `alias pp`
        ```pp='poetry run python'```
    - ~~mini-forge MacOSX arm64版~~
    - ~~`python -VV` の出力~~
      ```
      Python 3.9.7 | packaged by conda-forge | (default, Sep 29 2021, 19:22:19) [Clang 11.1.0 ]
      ```
    - ~~activate は次のコマンド~~
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

- ch03_1.py を， ch02.py を利用して実装
- 活性化関数をプロットし可視化
    - step関数
    - sigmoid関数
    - ReLU関数
- 行列計算の動作確認
- ch03_2.py を， ch03_1.py を利用して実装
    - 要素別の計算と行列計算で3層NWの実装
    - forward関数
        - NNの伝播をする関数
    - softmax関数
- ch03_3.py へ MNIST を実装
    - 学習済NWのロードと推論の実装
    - テストデータの画像表示と認識結果と正解の表示
    - 精度評価
    - batchサイズを変更した
- batch
    - 入力データを $$N$$ 個まとめて入力し，出力データを $$N$$ 個出力するような処理のこと
    - まとめてデータを入れることで，IOの処理時間に大して計算時間の比率を多くできるメリットがある

## ch04

- ch04.py を実装
- `cross_entropy_error(y, t)` 交差エントロピー誤差を実装
- バッチ推論の誤差を交差エントロピーで計算するのを実装
- `numerical_diff` 数値微分を実装
- `part_diff` 数値偏微分を実装
- `gradient` 勾配計算を実装
