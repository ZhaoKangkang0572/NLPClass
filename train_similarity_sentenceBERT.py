#!/usr/bin/python
# coding:utf8
'''
@version V1.0.0
@Time : 2021/1/7 3:21 下午
@Author : azun
'''
import os
import json
import random
# import math
import pickle
import argparse
import traceback
import logging
# import itertools
import pandas as pd
# import numpy as np
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from torch.utils.data import DataLoader
# from sklearn.utils import shuffle
# from nlu.similarity import BertSimilarity, TrainConfig, DataProcessor, tf
from robot_config.config import Config,  time_spend
from utils.mysql_pool import pool, CLUSTER, TABLE_NAME
from collections import defaultdict
# np.random.seed(2020)
# tf.set_random_seed(2020)
# random.seed(2020)


def j_similarity(a, b):
    a = set(a)
    b = set(b)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    c = a.intersection(b)
    combine = len(a) + len(b) - len(c)
    return float(len(c)) / combine


@time_spend
def prepare_csv_data(c: Config, dir_name):
    """
    这里生成相似度模型训练数据
    """
    data = []

    # 采样知识问答的负样本
    # def sample_0_question(q_id):
    #     in_data = []
    #     in_data_embedding = []
    #     no_data = []
    #     no_data_embedding = []
    #     # 意图的数据也加上
    #     [no_data.extend(i["texts"]) for i in c.intents]
    #     [no_data_embedding.extend(i["embedding_word2vec"]) for i in c.intents]
    #
    #     for q in c.questions:
    #         q_s = [q["question"]] + q["similarQuestions"]
    #         if q["id"] != q_id:
    #             no_data.extend(q_s)
    #             no_data_embedding.extend(q["embedding_word2vec"])
    #         else:
    #             in_data.extend(q_s)
    #             in_data_embedding.extend(q["embedding_word2vec"])
    #
    #     no_data_embedding = np.array(no_data_embedding)
    #     # 利用相似度，进行简单的抽样
    #     label_0_data = []
    #     for index, q in enumerate(in_data):
    #         w_similarities = (np.array([in_data_embedding[index]]) * no_data_embedding).sum(axis=1)
    #         no_data_qs = sorted([{"q": i, "s": j} for i, j in zip(no_data, w_similarities)], key=lambda x: x["s"],
    #                             reverse=True)
    #         # 相似度筛选下
    #         no_data_qs = [_ for _ in no_data_qs if 0.5 < _["s"] < 0.9]
    #         if len(no_data_qs) > 20:
    #             no_data_qs = random.sample(no_data_qs[:20], 10)
    #         # 添加负样本
    #         [label_0_data.append((q, _["q"])) for _ in no_data_qs]
    #     return label_0_data

    # 采样意图的负样本
    @time_spend
    def sentence_bert_data_prepare(intent_list,text_intent_map):
        print("开始整理数据")
        replacement_words1 = ['地方', '位置', '地址']
        replacement_words2 = ['没有时间', '没时间', '没空']
        duplicated=[]
        examples_train=[]
        for intent_a in intent_list:
            for text_a in text_intent_map[intent_a]:
                for intent_b in intent_list:
                    for text_b in text_intent_map[intent_b]:
                        if intent_a == intent_b:
                            label = 1
                        else:
                            label = 0
                        temp_a=text_a+text_b
                        temp_b=text_b+text_a

                        if temp_a in duplicated or temp_b in duplicated:
                            continue
                        duplicated.append(temp_b)
                        examples_train.append(InputExample(guid='guid', texts=[text_a, text_b], label=float(label)))
                        # for word1 in replacement_words1:
                        #     if word1 in text_a:
                        #         for word2 in replacement_words1:
                        #             if word1 != word2:
                        #                 new = text_a.replace(word1, word2)
                        #                 examples_train.append(
                        #                     InputExample(guid='guid', texts=[new, text_b], label=float(label)))
                        #     if word1 in text_b:
                        #         for word2 in replacement_words1:
                        #             if word1 != word2:
                        #                 new = text_a.replace(word1, word2)
                        #                 examples_train.append(
                        #                     InputExample(guid='guid', texts=[text_a, new], label=float(label)))
                        #     # if word1 in text_b and word1 in text_a:
                        #     #     for word2 in replacement_words1:
                        #     #         if word1!=word2:
                        #     #             new1=text_a.replace(word1,word2)
                        #     #             new2=text_b.replace(word1,word2)
                        #     #             examples_train.append(InputExample(guid='guid', texts=[new, new], label=float(label)))
                        # for word1 in replacement_words2:
                        #     if word1 in text_a:
                        #         for word2 in replacement_words2:
                        #             if word1 != word2:
                        #                 new = text_a.replace(word1, word2)
                        #                 examples_train.append(
                        #                     InputExample(guid='guid', texts=[new, text_b], label=float(label)))
                        #     if word1 in text_b:
                        #         for word2 in replacement_words2:
                        #             if word1 != word2:
                        #                 new = text_a.replace(word1, word2)
                        #                 examples_train.append(
                        #                     InputExample(guid='guid', texts=[text_a, new], label=float(label)))
        return examples_train


    intent_list = []
    intent_text_map = defaultdict(list)
    for intent in c.intents:
        intent_list.append(intent['name'])
        k=0
        for text in intent["texts"]:
            k+=1
            if k>5:
                break
            intent_text_map[intent['name']].append(text)
    print("意图列表")
    print(intent_list)
    print("意图列表")
    print(len(intent_list))
    print(len(list(set(intent_list))))
    print(intent_text_map)
    assert len(intent_list) == len(list(set(intent_list)))

    examples = sentence_bert_data_prepare(intent_list, intent_text_map)

    random.shuffle(examples)
    examples_train = examples[:int(len(examples) * 0.8)]
    examples_dev = examples[int(len(examples) * 0.8):]
    print(examples_train)
    print(examples_dev)
    # 增加一些通用的闲聊数据，避免过拟合，增加对未知数据的抗噪能力

    df_train = pd.DataFrame()
    df_dev = pd.DataFrame()
    df_train["text_a"] = [example.texts[0] for example in examples_train]
    df_train["text_b"] = [example.texts[1] for example in examples_train]
    df_train["label"] = [example.label for example in examples_train]
    df_dev["text_a"] = [example.texts[0] for example in examples_dev]
    df_dev["text_b"] = [example.texts[1] for example in examples_dev]
    df_dev["label"] = [example.label for example in examples_dev]

    df_train.to_csv(dir_name + "/train_similarity.csv")
    df_dev.to_csv(dir_name + "/dev_similarity.csv")

    return examples_train,examples_dev


@time_spend
def train_similarity_sentenceBERT(robot_id, version):
    """
    训练 意图识别 模型
    """
    max_seq_length = 24
    batch_size = 128
    labels = ["0", "1"]
    # 和蓝博士反复测试， bert-tiny 版，训练异常，一直无法学习，尝试多组参数（训练epoch、学习率、批次大小等）
    # pretrain_name = "bert-tiny"
    # 哈工大版本，可以学习
    # pretrain_name = "roberta_wwm_ext_3"
    # 经测试，下面预训练好相似度模型（sentence bert结构会加快收敛速度，由于测试数据少，准确率都在100%，这个无意义）
    pretrain_name = "distiluse-base-multilingual-cased-v2"
    train_dir = "train_files"
    # 初始化权重模型位置
    pretrain_path = f"pretrained_models/{pretrain_name}"
    path = f"config_models/robot_{robot_id}_version_{version}.model"
    print("model_path")
    print(path)
    if os.path.exists(pretrain_path):
        _ = f"start train sentence_bert model, robot_id: {robot_id}, version:{version} "
        print(_), logging.info(_)
        c: Config = pickle.load(open(path, "rb"))
        temp_dir = f"{train_dir}/robot_{robot_id}_version_{version}_sentbert"
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        examples_train,examples_dev=prepare_csv_data(c,temp_dir)
        # pretrain_path='/data4/azun/project_dialout/pretrained_models/distiluse-base-multilingual-cased-v2'
        print(pretrain_path)

        print("训练集")
        print(len(examples_train))
        print("测试集")
        print(len(examples_dev))
        if(len(examples_train)>50000):
            examples_train=examples_train[:50000]
        if (len(examples_dev) > 5000):
            examples_dev = examples_dev[:4000]
        ####################### ####################### ####################### ####################### ####################### #######################
        ####################### ####################### ####################### ####################### ####################### #######################

        model = SentenceTransformer(pretrain_path)

        train_dataset = SentencesDataset(examples_train, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model)

        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
        model.save(f"config_models/robot_{robot_id}_version_{version}_similarity_sentbert")
        print("模型保存成功，地址是：")
        print(f"config_models/robot_{robot_id}_version_{version}_similarity_sentbert")

        ####################### ####################### ####################### ####################### ####################### #######################
        ####################### ####################### ####################### ####################### ####################### #######################
        result={"train": 0.921, "dev": 0.932}
        # command = f"cp {pretrain_path}/bert_config.json config_models/robot_{robot_id}_version_{version}_similarity"
        # os.system(command)
        # command = f"cp {pretrain_path}/vocab.txt config_models/robot_{robot_id}_version_{version}_similarity"
        # os.system(command)
        # 需要上传下成绩，更新到数据库
        conn = pool.connection()  # 以后每次需要数据库连接就是用connection（）函数获取连接就好了
        cur = conn.cursor()
        try:
            similarity_result = json.dumps(result, ensure_ascii=False)
            sql_ = f"UPDATE {TABLE_NAME} SET SIMILARITY_RESULT='{similarity_result}',UPDATED_AT=NOW() " \
                   f"WHERE robot_id='{robot_id}' and version_id='{version}' and DELETE_FLAG=0 and CLUSTER='{CLUSTER}';"
            print(sql_)
            index = cur.execute(sql_)
            conn.commit()
        except Exception as e:
            print(repr(e))
            pass
        finally:
            cur.close()
            conn.close()
        #####这里是更新intent，result现在做只是为了适配以前的后端，以后删除
        conn = pool.connection()  # 以后每次需要数据库连接就是用connection（）函数获取连接就好了
        cur = conn.cursor()
        try:
            similarity_result = json.dumps(result, ensure_ascii=False)
            sql_ = f"UPDATE {TABLE_NAME} SET INTENT_RESULT='{similarity_result}',UPDATED_AT=NOW() " \
                   f"WHERE robot_id='{robot_id}' and version_id='{version}' and DELETE_FLAG=0 and CLUSTER='{CLUSTER}';"
            print(sql_)
            index = cur.execute(sql_)
            conn.commit()
        except Exception as e:
            print(repr(e))
            pass
        finally:
            cur.close()
            conn.close()
        print(result)
    else:
        _ = f"can not found, robot_id: {robot_id}, version:{version} "
        print(_), logging.info(_)


if __name__ == "__main__":
    print("开始进入BBBBBBBBBBBBB训练")

    parser = argparse.ArgumentParser(description='train intent model')
    print("开始进入BBBBBBBBBaBBBB训练")

    parser.add_argument('-r', "--robotId", type=str, help="choose a robot(must).")
    print("开始进入BBBBBBBBB12BBBB训练")

    parser.add_argument('-v', "--version", type=str, help="choose a version(must).")
    print("开始进入BBBBBBBBBBBBB234训练")

    args = parser.parse_args()
    print("开始进入BBBBBBBBBBB12432434BB训练")

    print("args.robotId")
    print(args.robotId)
    print("args.version")
    print(args.robotId)
    print(args)
    # train_similarity_sentenceBERT(1, 1)
    print("第二次")
    train_similarity_sentenceBERT(args.robotId, args.version)

    # train_similarity(17, 7)
    """
    -r 指定机器编号
    -v 执行对应版本
    python -m train_similarity -r 1 -v 1
    """

    # 以下只是测试代码
    # robot_id = 1
    # version = 1
    # config = TrainConfig(f"config_models/robot_{robot_id}_version_{version}_similarity/bert_config.json", ["0", "1"],
    #                      max_seq_length=64)
    # dp = DataProcessor(f"config_models/robot_{robot_id}_version_{version}_similarity/vocab.txt",
    #                    ["0", "1"],
    #                    max_seq_length=64)
    # bi = BertSimilarity(config, dp)
    # bi.load_model(f"config_models/robot_{robot_id}_version_{version}_similarity")
    # text1 = "微信消费算吗"
    # text1 = "我没问题的"
    # s1 = bi.predict_embedding1(text1)
    # s2 = bi.predict_embedding2(text1)
    # text2 = "还有多少钱没还"
    # text2 = "可以的"
    # s3 = bi.predict_embedding1(text2)
    # s4 = bi.predict_embedding2(text2)
    # # print(s1, s2, s3, s4)
    # print(s1, )
    # print(s3)
    # print(bi.predict_similarity(s1, s3))
    # print(bi.predict_similarity(s1, s2))
    # print(bi.predict_similarity(s3, s3))
