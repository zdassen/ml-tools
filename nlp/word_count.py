# -*- coding: utf-8 -*-
#
# word_count.py
#
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from datetime import datetime


def word_count_ranking(lines, write_file=True):
    """単語の頻度が高い順にcsvファイルに書き込む"""

    # 単語の頻度をカウントする
    cv = CountVectorizer()
    X = cv.fit_transform(lines).toarray()

    # 各単語の頻度を合計
    word_count = np.sum(X, axis=0)

    # { 単語: 頻度 } の形式で整理する
    d = {}
    for word, wid in cv.vocabulary_.items():
        d[word] = word_count[wid]
    c = Counter(d)

    if write_file:

        # 現在時刻からファイル名を生成
        now = datetime.now()
        nows = now.strftime("%y%m%d_%H%M%S")
        fname = "word_count_ranking_%s.csv" % nows
        fpath = "./%s" % fname

        # csv ファイルに書き込む
        with open(fpath, "w") as f:

            # ヘッダ部分
            f.write("word,count\n")

            # 単語,頻度
            for word, count in c.most_common():
                f.write("%s,%d\n" % (word, count))

    # ※何かオブジェクトを返してもよい


def ccoc(lines, words, is_talkative=True):
    """単語の共起数を計算する(Count Co-OCcurrences)"""

    # 単語の頻度をベクトル化する
    cv = CountVectorizer()
    X = cv.fit_transform(lines).toarray()

    # 文章が単語を含むかどうかのフラグ配列を得る
    flags_contains = []
    for word in words:

        # 単語の、ベクトルち中のインデックス
        wi = cv.vocabulary_[word]

        # 文書が単語を含むかどうか
        contains_word = X[:, wi] >= 1

        # フラグ配列を追加する
        flags_contains.append(contains_word)

        # 単語ごとに頻度を表示する
        if is_talkative:
            print("%s: %d" % (word, np.sum(contains_word)))

    # 各単語の出現フラグにつき AND 演算をかける
    tmp = np.ones_like(flags_contains[0])
    for flag_contains in flags_contains:
        tmp = tmp & flag_contains

    return np.sum(tmp)


def sample():
    """単語の頻度をcsvファイルを書き込む"""

    # 文書データを読み込み
    tweet = pd.read_csv("../data/twitter-sanders-apple2.csv")

    # 単語の頻度を csv ファイルに書き込む
    word_count_ranking(tweet["text"].values)


def sample2():
    """単語の頻度を確認する(csvファイルを読み込む)"""
    fpath = "./word_count_ranking_180713_124932.csv"

    # データフレームにする
    wc = pd.read_csv(fpath)
    print(wc.head(n=30))


def sample3():
    """単語の共起数をカウントする"""
    tweet = pd.read_csv("../data/twitter-sanders-apple2.csv")

    # 共起数
    words = ("awesome", "crazy")
    cooccurrences = ccoc(tweet["text"].values, words)
    print(cooccurrences)


if __name__ == '__main__':
    # sample()    # 単語の頻度を csv ファイルを書き込む
    # sample2()    # 単語の頻度を確認する ( csv ファイルを読み込む )
    # sample3()    # 単語の共起数をカウントする
    pass