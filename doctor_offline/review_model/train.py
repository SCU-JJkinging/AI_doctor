#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/14 16:23
# @Author  : JJkinging
# @File    : train.py
import pandas as pd
from collections import Counter

# 读取数据
train_data_path = './train_data.csv'
train_data = pd.read_csv(train_data_path, header=None, sep='\t')

# 打印正负标签比例
print(dict(Counter(train_data[0].values)))

# 转换数据到列表形式
train_data = train_data.values[0].tolist()
print(train_data[:10])