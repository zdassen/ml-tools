# -*- coding: utf-8 -*-
#
# utils.py
#
import numpy as np


def dline(n=32):
    """二重線を引く"""
    print("=" * n)


def shapes(*arrs):
    """行列のサイズをチェックする"""
    for i, arr in enumerate(arrs):
        if isinstance(arr, (list, tuple)):
            print("arg %2d: size %d" % (i, len(arr)))
        elif isinstance(arr, np.ndarray):
            print("arg %2d: %s" % (i, arr.shape))