# -*- coding: utf-8 -*-
#
# polyfit_for_best_order.py
#
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# サンプルで使用するライブラリのインポート
import scipy as sp


def dline(n=32):
    """二重線を引く"""
    print("=" * n)


def polyfit_for_best_order(X, y, error,
    orders=(1, 2, 3, 5, 10),
    test_size=0.3,
    plot_scatter=False,
    plot_all=True):
    """ベストな次数を得る"""
    
    # 散布図を描画する
    if plot_scatter:
        sns.set()
        plt.scatter(X, y, s=6, color="crimson")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.4)
        plt.show()

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)
    dline()
    print("n train: %d" % len(X_train))
    print("n test : %d" % len(X_test))

    # 各次数で試す
    dline()
    funcs = []
    for order in orders:

        # フィットさせる ( パラメータを得る )
        params = sp.polyfit(X_train, y_train, order)

        # 関数 ( 多項式 ) を作成
        f_order = sp.poly1d(params)
        funcs.append(f_order)

        # テストデータ投入時の誤差を求める
        error_test = error(f_order, X_test, y_test)
        print("order: %3d => error: %f" % (order, error_test))

    # 散布図と学習済みの曲線を描画する
    if plot_all:

        # 散布図を描画する
        sns.set()
        Xs = (X_train, X_test)
        ys = (y_train, y_test)
        colors = ("springgreen", "orange")
        labels = ("train", "test")
        for i in range(2):
            plt.scatter(Xs[i], ys[i], s=6, color=colors[i],
                label=labels[i])

        # 学習済みの曲線を描画する
        x_space = np.linspace(0, X[-1], 1000)
        for i in range(0, len(orders)):
            f = funcs[i]
            plt.plot(x_space, f(x_space))

        # 次数と曲線の色の関係を表示させる
        legends = [
            "d=%i" % f.order
                for f in funcs
        ]

        plt.legend(legends, loc="best")
        plt.grid(True, alpha=0.4)
        plt.show()


def error(f, X, y):
    """誤差の二乗和を求める"""
    return sp.sum((f(X) - y) ** 2)


def sample():
    """サンプル"""

    # データの読み込み
    data = sp.genfromtxt("../data/web_traffic.tsv",
        delimiter="\t")

    # ※本来なら NaN チェックを行うべき

    # 変数を取り出す
    X = data[:, 0]
    y = data[:, 1]

    # 誤差の最も小さい次数を得る
    orders = (1, 2, 3, 5, 10, 50)
    polyfit_for_best_order(X, y, error, orders,
        plot_scatter=False, plot_all=True)


if __name__ == '__main__':
    # sample()
    pass