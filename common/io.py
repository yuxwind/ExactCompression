# coding: utf-8

import os
import numpy as np
import pickle
import glob
import re

def mkdir(d):
    os.makedirs(d, exist_ok=True)
    return d

def mkpath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        try:
            return np.load(fp)
        except:
            return np.load(fp, allow_pickle=True)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def _dump(wfp, obj):
    suffix = _get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':
        pickle.dump(obj, open(wfp, 'wb'))
    else:
        raise Exception('Unknown Type: {}'.format(suffix))

def find_float_in_str(str_):
    numeric_const_pattern = r"""
            [-+]? # optional sign
            (?:
                (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
                |
                (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
            )
            # followed by optional exponent part if desired
            (?: [Ee] [+-]? \d+ ) ?
            """
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    f_ =  rx.findall(str_)
    f_ = np.array(f_,dtype=float)
    return f_


if __name__ == '__main__':
    s1 = 'err_R [  4.87797058  12.75937769  -3.77550213 -15.18667789]'
    s2 = 'predict R error: 4.88 rx,ry,rz: 12.76,-3.78,-15.19'
    f1 = find_float_in_str(s1)
    f2 = find_float_in_str(s2)
    print(s1,f1)
    for f in f1:
        print(f'|{f}|')
    print(s2,f2)
