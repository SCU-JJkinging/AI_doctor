#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/14 15:24
# @Author  : JJkinging
# @File    : EntityToNeo4j.py

# 引入相关包
import os
import fileinput
from neo4j import GraphDatabase
from config import NEO4J_CONFIG

driver = GraphDatabase.driver(**NEO4J_CONFIG)


def _load_data(path):
    """
    description: 将path目录下的csv文件以指定格式加载到内存
    :param path:  审核后的疾病对应症状的csv文件
    :return:      返回疾病字典，存储各个疾病以及与之对应的症状的字典
                  {疾病1: [症状1, 症状2, ...], 疾病2: [症状1, 症状2, ...]
    """
    # 获得疾病csv列表
    disease_csv_list = os.listdir(path)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。

    # 获取疾病列表 disease_list 和一个症状列表 symptom_list
    # 将后缀.csv去掉, 获得疾病列表
    disease_list = list(map(lambda x: x.split(".")[0], disease_csv_list))

    # 初始化一个症状列表, 它里面是每种疾病对应的症状列表
    symptom_list = []
    # 遍历疾病csv列表
    for disease_csv in disease_csv_list:
        # 将疾病csv中的每个症状取出存入symptom列表中
        # fileinput.FileInput 可以读取多行
        symptom = list(map(lambda x: x.strip(),
                           fileinput.FileInput(os.path.join(path, disease_csv),
                                               openhook=fileinput.hook_encoded('utf-8'))))
        # 过滤掉所有长度异常的症状名
        # filter() 函数用于过滤序列，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。
        symptom = list(filter(lambda x: 0 < len(x) < 100, symptom))
        symptom_list.append(symptom)
    # 返回指定格式的数据
    return dict(zip(disease_list, symptom_list))


def write(path):
    """
    description: 将csv数据写入到neo4j, 并形成图谱
    :param path: 数据文件路径
    """
    # 使用_load_data从持久化文件中加载数据
    disease_symptom_dict = _load_data(path)
    # 开启一个neo4j的session
    with driver.session() as session:

        for key, value in disease_symptom_dict.items():
            cypher = "MERGE (a:Disease{name:%r}) RETURN a" % key
            session.run(cypher)
            for v in value:
                cypher = "MERGE (b:Symptom{name:%r}) RETURN b" % v
                session.run(cypher)
                cypher = "MATCH (a:Disease{name:%r}) MATCH (b:Symptom{name:%r}) \
                          WITH a,b MERGE(a)-[r:dis_to_sym]-(b)" % (key, v)
                session.run(cypher)
        cypher = "CREATE INDEX ON:Disease(name)"
        session.run(cypher)
        cypher = "CREATE INDEX ON:Symptom(name)"
        session.run(cypher)

if __name__ == "__main__":
    # 输入参数path为csv数据所在路径
    path = "D:/python_project/AI_doctor/doctor_offline/structured/reviewed/"
    write(path)