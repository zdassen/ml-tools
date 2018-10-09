# -*- coding: utf-8 -*-
#
# learn-prep/lib/plotters.py
#
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# 分類用のライブラリのインポート
from sklearn.neighbors import KNeighborsClassifier


def plot_decision_regions(X, y, clf, resolution=0.02,
    do_plot=True):
    """決定境界をプロットする"""

    # マーカーとカラーマップの準備
    # markers = ("s", "x", "o", "^", "v")
    markers = ("o", "o", "o", "o", "o")
    colors = ("springgreen", "orange", "crimson", "gray", "cyan")
    uniq_classes = np.unique(y)
    n_uniq_classes = len(uniq_classes)
    cmap = ListedColormap(colors[:n_uniq_classes])

    # 決定境界のプロット

    # プロット用領域の設定
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # プロット領域に 0 が含まれる場合は x軸、y軸を描画する
    r1 = range(int(x1_min), int(x1_max) + 1)
    if 0 in r1:
        plt.axvline(x=0, color="gray", linewidth=0.5)

    r2 = range(int(x2_min), int(x2_max) + 1)
    if 0 in r2:
        plt.axhline(y=0, color="gray", linewidth=0.5)

    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # 予測値を算出
    # ravel() によって　x1, x2 の値の組み合わせが出来上がる
    _X = np.array([
        xx1.ravel(),
        xx2.ravel(),
    ]).T
    Z = clf.predict(_X)
    Z = Z.reshape(xx1.shape)

    # 等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)

    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロット
    for i, cls in enumerate(uniq_classes):
        is_cls = y == cls
        plt.scatter(X[is_cls, 0], X[is_cls, 1],
            c=cmap(i), alpha=0.8, marker=markers[i],
            label=cls)

    # グリッドを表示する
    plt.grid(True, alpha=0.4)

    # この時点でプロットする場合のみプロット
    if do_plot:
        plt.show()


def knn_decision_regions(X, y, n_neighbors, **kwargs):
    """近傍による分類結果をプロットする"""

    # ※[確認] .. データを標準化したか?

    # 近傍でクラス分類を行う
    neigh = KNeighborsClassifier(
        n_neighbors=n_neighbors)
    neigh.fit(X, y)

    # 決定境界をプロットする
    sns.set()
    plot_decision_regions(X, y, neigh, **kwargs)


def collate(rows, cols, *params, titles=None):
    """NumPy.ndarrayをプロットする"""

    if titles:
        
        # パラメータの数 = プロット時のタイトル数
        assert len(params) == len(titles)

    for i in range(rows):
        for j in range(cols):
            ind = cols * i + j

            title = None
            try:

                # 画像とタイトル ( あれば ) を取り出す
                img = params[ind]
                if titles:
                    title = titles[ind]

            except IndexError:
                pass
            else:

                # プロット位置の指定 + タイトル
                plt.subplot(rows, cols, ind + 1)
                if title:
                    plt.title(title)

                # 画像をプロットする
                plt.imshow(img, cmap=plt.cm.gray)

        # end of for j in range(cols) ...
    # end of for i in range(rows) ...

    plt.show()


def plot_conv_weights(conv, nr, nc):
    """畳み込み層の重みをまとめてプロットする"""

    # 重みを取り出す ( NumPy.ndarray )
    w = conv.weight.data.numpy()

    # 入出力チャンネル数、フィルタのサイズを取得する
    out_channels, in_channels, h_kernel, w_kernel = \
        w.shape

    # nr, nc で指定した分だけプロットする
    # ※実際に存在するフィルタの数の方が多い場合があるため
    ws = []
    for i in range(nr):
        for j in range(nc):
            ws.append(w[i, j, :, :])

            # ※タイトルどうする?

    collate(nr, nc, *ws)