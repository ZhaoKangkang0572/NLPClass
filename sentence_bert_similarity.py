#!/usr/bin/python
# coding:utf8
"""
@version: v1.0.0
@author: baili
@license: Apache Licence
@contact: yucong@i-i.ai
@time:  2020/9/30 5:36 下午
"""
import os
import traceback
import re
import json
import time
import logging
import math
import numpy as np
import tensorflow as tf

from nlu.config import TrainConfig, DataProcessor
from bert_module import modeling, optimization
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer, util

"""
similarity 模块： 这里主要是 自定义的相似度任务。采用sentenceBert架构
"""


def load_bert_config(path):
    """
    bert 模型配置文件，electra 与 bert 配置类参数是一致的
    """
    return modeling.BertConfig.from_json_file(path)


class SentenceBertSimilarity:
    def __init__(self, model_path):
        # self.train_config = config
        # self.data_processor = data_processor
        # self.bert_config = load_bert_config(self.train_config.bert_config_path)
        _ = f"开始载入，sentencebert初始化"
        print(_), logging.info(_)
        # print(os.getcwd())
        # model_path='/data4/azun/project_dialout/pretrained_models/distiluse-base-multilingual-cased-v2'
        # model_path='/tmp/pycharm_project_457/models/V2'
        # model_path='config_models/robot_1_version_1_similarity_sentbert'
        # self.sentence_bert_model = SentenceTransformer('/data4/azun/project_dialout/config_models/robot_1_version_1_similarity_sentbert')
        self.model_path=model_path
        self.is_loaded=False
        print(os.getcwd())
        print(model_path)
        try:
            print(model_path)
            self.sentence_bert_model = SentenceTransformer(model_path)
        except:
            _ = f"初始化时，载入模型时有异常 "
            print(_), logging.info(_)
            traceback.print_exc()

        try:
            # print()#, convert_to_tensor=True))
            _ = f"打印你好的句向量{self.sentence_bert_model.encode(['你好'])} "
            print(_), logging.info(_)
        except:
            traceback.print_exc()
        pass


    def predict_embedding1(self, text):
        _ = f"开始预测 "
        print(_), logging.info(_)
        try:
            if not self.is_loaded:
                _ = f"找不到模型，重新载入 "
                print(_), logging.info(_)
                self.sentence_bert_model=SentenceTransformer(self.model_path)
                self.is_loaded = True
            embedding_text = self.sentence_bert_model.encode([text])#, convert_to_tensor=True)
            # print("拿到了embeddingtext")
            # print(embedding_text.tolist())
        except:
            _ = f"取出句向量时发生异常 "
            print(_), logging.info(_)
            traceback.print_exc()
        return embedding_text.tolist()
#
#
# bsm_sentence_bert_model = SentenceBertSimilarity("../config_models/robot_1_version_1_similarity_sentbert")
# # bsm_sentence_bert_model = SentenceBertSimilarity('/data4/azun/project_dialout/config_models/robot_1_version_1_similarity_sentbert')
#
# print(bsm_sentence_bert_model.predict_embedding1("你好"))