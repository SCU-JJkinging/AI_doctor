#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 16:36
# @Author  : JJkinging
# @File    : neo4j-driver_test.py
from neo4j import GraphDatabase
# 关于neo4j数据库的用户名,密码信息已经配置在同目录下的config.py文件中
from config import NEO4J_CONFIG

driver = GraphDatabase.driver(**NEO4J_CONFIG)

with driver.session() as session:
    cypher = 'create (c:Company) set c.name="海成科技" return c.name'
    record = session.run(cypher)
    result = list(map(lambda x: x[0], record))
    print('result:', result)