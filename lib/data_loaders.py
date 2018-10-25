# -*- coding: utf-8 -*-
#
# ml-tools/lib/data_loaders.py
#
import numpy as np

# TensorFlow & keras
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist


def get_rinds_label(y, label, n):
    """指定された数値のインデックスからn個をランダムに抽出"""
    is_label = np.where(y == label)[0]
    ri = np.random.choice(is_label, n, replace=False)
    return ri


def get_petit_MNIST(X, y, n_images_per_label):
    """
    小規模なMNISTデータを読み込む

    各数値につき n_images_per_label ずつのデータを抽出する
    """

    n_labels = 10
    for label in range(n_labels):
        ri = get_rinds_label(y, label, n_images_per_label)

        # 初回
        if label == 0:
            X_petit = X[ri, :]
            y_petit = y[ri]

        # 次回以降
        else:
            X_petit = np.vstack((X_petit, X[ri, :]))
            y_petit = np.hstack((y_petit, y[ri]))

    # end of for label in range(n_labels) ...

    return X_petit, y_petit


def get_MNIST_train_test(n_images_per_label=None,
    test_size=0.2, standardize=True):
    """MNISTデータセットを読み込む"""

    # 訓練用のデータとテスト用のデータを読み込む ( keras の場合 )
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 小規模なデータセットを利用する場合
    if n_images_per_label:
        assert isinstance(n_images_per_label, int)

        # テスト用のデータ数 = 訓練用のサイズ x test_size ( 切り捨て )
        tst_size = int(n_images_per_label * test_size)

        # 指定個数だけデータとラベルをランダムに抽出する
        (X_train, y_train), (X_test, y_test) = [
            get_petit_MNIST(X, y, size)
                for X, y, size in (
                    (X_train, y_train, n_images_per_label),
                    (X_test, y_test, tst_size),
                )
        ]

    # 標準化を行う
    if standardize:
        X_train, X_test = [X / 255. for X in (X_train, X_test)]

    return X_train, y_train, X_test, y_test