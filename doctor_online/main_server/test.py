#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 17:36
# @Author  : JJkinging
# @File    : test.py
import fileinput

symptom = list(map(lambda x: x.strip(),
                           fileinput.FileInput(r'D:\python_project\AI_doctor\doctor_offline\structured\reviewed\2型糖尿病.csv',
                                               openhook=fileinput.hook_encoded('utf-8'))))
print(symptom)
