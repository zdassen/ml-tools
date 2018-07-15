# -*- coding: utf-8 -*-
#
# piped_cv.py
#
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit,\
GridSearchCV
from sklearn.metrics import precision_recall_curve, auc

from datetime import datetime


def dbline(n=32):
    print("=" * n)


def piped_grid_cv(X, y, pipeline, param_grid, n_splits=5,
    test_size=0.3, save_result_csv=False, scoring=None,
    show_result_keys=True):
    """交差検定+グリッドサーチ"""

    # 交差検定を行う
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size)

    # グリッドサーチを行う
    gs = GridSearchCV(pipeline, param_grid, cv=ss,
        scoring=scoring, return_train_score=False)
    gs.fit(X, y)

    # 結果を csv ファイルに保存する
    cv_results = pd.DataFrame(gs.cv_results_)
    if save_result_csv:
        
        # 現在時刻をファイル名に含ませる
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        fpath = "./cv_results_%s.csv" % ts

        # csv ファイルを書き込む
        cv_results.to_csv(fpath)

    # 最適なパラメータ
    dbline()
    print("best parameter:")
    best_param_index = gs.best_index_
    print(cv_results["params"][best_param_index])

    # 測定値の名前をリストアップする
    if show_result_keys:
        dbline()
        for k in gs.cv_results_.keys():
            print(k)
        print(cv_results["split0_test_score"])

    return gs.best_estimator_


def plot_precision_recall_curve_cv(X, y, estimator, n_splits=5,
    test_size=0.3):
    """交差検定+Precision-Recall曲線を描画"""

    # Precision、Recall、AUC ( 曲線の下側の面積の割合 ) 
    # の値を格納する
    pres = []
    recs = []
    aucs = []

    # 交差検定を行う
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size)
    for train_i, test_i in ss.split(X):

        # 訓練データ
        X_train, y_train = X[train_i, :], y[train_i]

        # テスト用データ
        X_test, y_test = X[test_i, :], y[test_i]

        # 学習させる→予測を行う
        estimator.fit(X_train, y_train)
        y_proba = estimator.predict_proba(X_test)

        # Precision、Recall、閾値を得る
        pre, rec, thres = precision_recall_curve(
            y_test, y_proba[:, -1])
        pres.append(pre)
        recs.append(rec)

        # AUC ( 曲線の下側の面積の割合 ) を格納
        aucs.append(
            auc(rec, pre)
        )

    # end of for train_i, test_i in ss: ...
    
    # AUC が中央値のものの曲線を描画する
    mi = int(n_splits / 2)
    med_i = np.argsort(aucs)[mi]
    pre_med = pres[med_i]
    rec_med = recs[med_i]
    auc_med = aucs[med_i]

    sns.set()
    plt.title("Precision-Recall curve (AUC: %f)" % auc_med)
    plt.step(rec_med, pre_med, color="b", alpha=0.2,
        where="post")
    plt.fill_between(rec_med, pre_med, step="post", color="b", alpha=0.2)
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.0))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.4)
    plt.show()