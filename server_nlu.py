#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-11-13 18:04
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import re
import json
import jieba
import time
import logging
import requests
import traceback
import pickle
import numpy as np
# from collections import Counter
# from collections import defaultdict
from nlu.system_ner import SystemNER
from sentence_transformers import SentenceTransformer, util
from utils.redis_pool import master as redis
from robot_config.config import Config, get_sentence_embedding, bert_4_layer, time_spend
from sys_config import MODEL_HOST, USE_TF_SERVING, USE_MODEL_MANAGER, MODEL_MANAGER_HOST
# import time
import torch
sn = SystemNER()
text_ = "，杨先生发表演讲，杨唯美发表演讲杨唯美发表演讲"
a = sn.get_model_tags(text_, ["address"])
print(a)

@time_spend
def intent_test(text_embedding_sentence_bert,sentence_total,embddings_total,map_total, num):


    # print(embddings_total.tolist()[10:])

    # embddings_total.tolist()
    # print(text_embedding_sentence_bert.tolist())
    cosine_scores = util.pytorch_cos_sim(text_embedding_sentence_bert, embddings_total)
    pairs = []
    for j in range(len(cosine_scores[0] )- 1):
        pairs.append({'index': [0, j], 'score': cosine_scores[0][j],'sentence':sentence_total[j]})
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    # print(pairs)
    # temp = []
    # for pair in pairs[0:num]:
    #     i, j = pair['index']
    #     score_map[sentences_total[j]]=pair['score']
        # print(      "{:4} \t\t  {:32} \t\t {:32} \t\t Score: {:.4f}".format('       ', sentence_total[j], 'text', pair['score']))
        # temp.append(map_total[sentence_total[j]])
    # intent=Counter(temp).most_common(1)[0][0]
    # return {'name': map_total[sentence_total[pairs[0]['index'][1]]], 'similarity': pairs[0]['score'].item(), 'similarText': sentence_total[pairs[0]['index'][1]]}
    # return {'name': intent, 'similarity': pair['score'].item(), 'similarText': sentence_total[j]}
    return {'name': map_total[pairs[0]['sentence']], 'similarity': pairs[0]['score'].item(), 'similarText': pairs[0]['sentence']}

# def replace(data, replacement_words1, replacement_words2, test2_map):
#     for idx, row in data.iterrows():
#         test2_map[row['意图']].append(row['相似语句'])
#         for word1 in replacement_words1:
#             if word1 in row['相似语句']:
#                 for word2 in replacement_words1:
#                     if word1 != word2:
#                         new = row['相似语句'].replace(word1, word2)
#                         test2_map[row['意图']].append(new)
#         for word1 in replacement_words2:
#             if word1 in replacement_words2:
#                 for word2 in replacement_words2:
#                     if word1 != word2:
#                         new = row['相似语句'].replace(word1, word2)
#                         test2_map[row['意图']].append(new)
#     return test2_map

@time_spend
def changshu_(text_embedding_sentence_bert,sentence_total,embedding_sentence_bert,map_total):

    # for i in range(1, 10):
    return intent_test(text_embedding_sentence_bert,sentence_total,embedding_sentence_bert, map_total,num=1)
        # print('*' * 80)
# def sentence_bert_data_prepare():
#     replacement_words1 = ['地方', '位置', '地址']
#     replacement_words2 = ['没有时间', '没时间', '没空']
#     data = pd.read_excel('./nlu/测试集2.xlsx')
#     test2_map = defaultdict(list)
#     for idx, row in data.iterrows():
#         test2_map[row['意图']].append(row['相似语句'])
#     test2_map = replace(data, replacement_words1, replacement_words2, test2_map)
#     busy_sent = test2_map['没时间']
#     busy_sent_map = {x: '没时间' for x in busy_sent}
#     negative_add_car_sent = test2_map['否定']
#     negative_add_car_sent_map = {x: '否定' for x in negative_add_car_sent}
#     affirmative_add_car_sent = test2_map['肯定']
#     affirmative_add_car_sent_map = {x: '肯定' for x in affirmative_add_car_sent}
#     wrong_address = test2_map['地址有误']
#     wrong_address_map = {x: '地址有误' for x in wrong_address}
#     wrong_driver = test2_map['不是我在开']
#     wrong_driver_map = {x: '不是我在开' for x in wrong_driver}
#     sentences_total = busy_sent + negative_add_car_sent + affirmative_add_car_sent + wrong_address + wrong_driver
#     # sentences_total=busy_sent[:num]+negative_add_car_sent[:num]+affirmative_add_car_sent[:num]+wrong_address[:num]+wrong_driver[:num]
#     map_total = {**busy_sent_map,
#                  **negative_add_car_sent_map,
#                  **affirmative_add_car_sent_map,
#                  **wrong_address_map,
#                  **wrong_driver_map}
#
#     return sentences_total,map_total
def scale_to_float(i):
    i = float('%.2f' % float(i))
    if i <= 0.0:
        return 0.0
    elif i < 1.0:
        return i
    else:
        return 1.0


def j_similarity(a, b):
    a = set(a)
    b = set(b)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    c = a.intersection(b)
    combine = len(a) + len(b) - len(c)
    return float(len(c)) / combine


class NLU:
    def __init__(self, config: Config):
        self.config = config
        self.robot_id = self.config.robot_id
        self.version = self.config.version

        self.intent_model = None
        self.similarity_model = None
        # 修改判断模型的状态方案， 查询模型状态
        _ = "init nlu..."
        print(_), logging.info(_)
        # intent 包含 内置闲聊意图 default_chat，
        self.use_intent = True
        # self.intent_labels = None
        self.use_similarity = True
        self.sentence_total=self.config.texts_sentence
        # print(self.sentence_total[0])
        # print(self.sentence_total[1])
        # self.map_total=self.config.texts_intent_map

        self.map_total = {}
        for k, v in self.config.texts_intent_map.items():
            for v_ in v:
                self.map_total[v_] = k

        self.embedding_sentence_bert=self.config.embedding_sentence_bert
        # print(self.embedding_sentence_bert.tolist()[0])
        # print(self.embedding_sentence_bert.tolist()[1])
        try:
            res_ = requests.post(f"{MODEL_MANAGER_HOST}/api/model_query",
                                 json={"robotId": self.robot_id, "version": self.version}).json()
            print(res_)
            if int(res_["data"]["intent_model"]) > 0:
                self.use_intent = True
            else:
                self.use_intent = False
            if int(res_["data"]["similarity_model"]) > 0:
                self.use_similarity = True
            else:
                self.use_similarity = False
        except Exception as e:
            print(repr(e))
            self.use_intent = False
            self.use_similarity = False

        self.question_index = {}
        self.question_answers = {}
        self.similar_texts = {}
        self.embedding_word2vec = {}
        self.embedding_bert4 = {}

        self.id2title = {}
        self.id2label = {}
        self.has_category = False
        # for question in self.config.questions:
        #     label = 'qa' if len(question.get("categories", [])) < 1 else question["categories"][0]
        #     len_ = len(question['seg_texts'])
        #     self.id2title[question['id']] = question['question']
        #     self.id2label[question['id']] = label
        #     # 分词文本
        #     self.similar_texts[label] = self.similar_texts.get(label, []) + question['seg_texts']
        #     # 语义向量
        #     self.embedding_word2vec[label] = self.embedding_word2vec.get(label, []) + question["embedding_word2vec"]
        #     # 新增 embedding_bert4
        #     self.embedding_bert4[label] = self.embedding_bert4.get(label, []) + question["embedding_bert4"]
        #
        #     self.question_index[label] = self.question_index.get(label, []) + [question['id']] * len_
        #     # 答案存在可能不存在
        #     self.question_answers[label] = self.question_answers.get(label, []) + [question.get('answer', '')] * len_

        # 意图的 预处理数据
        # for intent in self.config.intents:
        #     label = intent["name"]
        #     len_ = len(intent['texts'])
        #     self.id2title[label] = label
        #     self.id2label[label] = label
        #     # 分词文本
        #     self.similar_texts[label] = self.similar_texts.get(label, []) + intent['seg_texts']
        #     # 语义向量
        #     self.embedding_word2vec[label] = self.embedding_word2vec.get(label, []) + intent["embedding_word2vec"]
        #     # 新增 embedding_bert4
        #     self.embedding_bert4[label] = self.embedding_bert4.get(label, []) + intent["embedding_bert4"]
        #
        #     self.question_index[label] = self.question_index.get(label, []) + [intent['name']] * len_
        #     # 答案存在可能不存在
        #     self.question_answers[label] = self.question_answers.get(label, []) + [intent.get('answer', '')] * len_

        # question_num = np.mean([len(t) for l, t in self.similar_texts.items()])
        # _ = f"question num: {question_num}"
        # print(_), logging.info(_)
        # if question_num < 100:
        #     print("intent clear...")
        #     self.use_intent = False

    @staticmethod
    def has_useful_text(text):
        a = re.sub(r"[_\s\·\~\！\@\#\￥\%\……\&\*\（\）\——\-\+\=\【\】\{\}\、\|\；\‘\’\：\“\”\《\》\？\，\。\、"
                   r"\`\~\!\#\$\%\^\&\*\(\)\_\[\]{\}\\\|\;\'\'\:\"\"\,\.\/\<\>\?\~\～]", "", text)
        return True if a == "" or len(a) > 20 else False

    @time_spend
    def get_intent_result(self, text, session_id=None, ):
        # 调用 意图识别模型接口
        try:
            payload_intent = {
                "modelName": "intent_model",
                "robotId": self.robot_id,
                "version": self.version,
                "text": text
            }
            # payload_similar = {"modelName": "similar_model",
            #                  "robotId": robot_id,
            #                  "version": new_version,
            #                  "text_as": ["今天天气", "今天天气"],
            #                  "text_bs": ["杭州疫情", "今天天气不错"]}
            url = f"{MODEL_MANAGER_HOST}/api/model_predict"
            res_ = requests.post(url=url, json=payload_intent).json()
            return res_["data"]
        except Exception as e:
            print(repr(e))
            return None

    @time_spend
    def get_similar_result_old(self, text_as, text_bs):
        # 调用 相似度模块
        try:
            # payload_intent = {
            #     "modelName": "intent_model",
            #     "robotId": self.robot_id,
            #     "version": self.version,
            #     "text": text
            # }
            payload_similar = {"modelName": "similar_model",
                               "robotId": self.robot_id,
                               "version": self.version,
                               "text_as": text_as,
                               "text_bs": text_bs}
            url = f"{MODEL_MANAGER_HOST}/api/model_predict"
            res_ = requests.post(url=url, json=payload_similar).json()
            return res_["data"]
        except Exception as e:
            print(repr(e))
            return None

    @time_spend
    def get_similar_result(self, text):
        # 调用 相似度模块
        _ = 'nlu:调用 相似度模块'
        print(_), logging.info(_)
        try:
            # payload_intent = {
            #     "modelName": "intent_model",
            #     "robotId": self.robot_id,
            #     "version": self.version,
            #     "text": text
            # }
            payload_similar = {"modelName": "sentence_bert",
                               "robotId": self.robot_id,
                               "version": self.version,
                               "text_as": text}
            url = f"{MODEL_MANAGER_HOST}/api/model_predict"
            res_ = requests.post(url=url, json=payload_similar).json()
            _ = f"nlu:拿到了res_，长度是：: {len(res_)}"
            print(_), logging.info(_)
            return res_["data"]
        except Exception as e:
            print(repr(e))
            return None
    @time_spend
    def get_slots_result(self, text, slots=None):
        # 调用 实体抽取 模块 slots 为 需要抽取的实体名称

        # 优先匹配自己的slot训练优化结果
        if slots is not None and len(slots):
            local_slots = []
            for slot in self.config.slots:
                if slot["name"] in slots:
                    flag = False
                    # 正则value抽取
                    res = re.search("(" + "|".join(slot["values"]) + ")", text)
                    if len(slot["values"]) and res:
                        local_slots.append({"name": slot["name"], "value": res.group()})
                        flag = True

                    if not flag:
                        # 正则pattern抽取
                        for p in slot["patterns"]:
                            res = re.search(p, text)
                            if res:
                                # local_slots.append({slot["name"]: res.group()})
                                local_slots.append({"name": slot["name"], "value": res.group()})
                                flag = True
                                break

                    # for v in slot["values"]:
                    #     if text.find(v):
                    #         local_slots.append({slot["name"]: v})
                    #         break
            # 模型可能会预测多个，在当前结构取第一个即可，同时不进行归一化操作
            # sys_slots = [{k: v[0]["origin"]} for _ in sn.get_ner_result(text, slots) for k, v in _.items()]
            return local_slots
        else:
            return []

    # @time_spend
    # def get_result_by_rule(self, text):
    #     """
    #     根据规则进行匹配
    #     """
    #     rule_result = []
    #     for intent in self.config.intents:
    #         patterns = intent["patterns"]
    #         texts = intent["texts"]
    #         texts = sorted(texts, key=lambda x: len(x), reverse=True)
    #         result = []
    #         for p in patterns:
    #             res = re.search(p, text)
    #             # 添加否定约束，前面不要存在否定词语修身，不然意图极性会变化
    #             if res and not re.search('[非别不没无勿]' + res.group(), text):
    #                 # s1 = float(len("".join([_ for _ in res.groups() if _]))) / len(res.group())
    #                 similarity_ = float(len(res.group())) / len(text)
    #                 # similarity_ = s1 * 0.3 + s2 * 0.7
    #                 # print(p, text, similarity_)
    #                 result.append({
    #                     "intent": intent["name"],
    #                     "probability": similarity_,
    #                     "similar_text": res.group(),
    #                 })
    #         try:
    #             res = re.search("|".join(texts), text)
    #         except Exception as e:
    #             print("rule error:", repr(e), intent["name"])
    #             res = None
    #         # 添加否定约束，前面不要存在否定词语修身，不然意图极性会变化
    #         if res and not re.search('[非别不没无勿]' + res.group(), text):
    #             # s1 = float(len("".join([_ for _ in res.groups() if _]))) / len(res.group())
    #             similarity_ = float(len(res.group())) / len(text)
    #             result.append({
    #                 "intent": intent["name"],
    #                 "probability": similarity_,
    #                 "similar_text": res.group(),
    #             })
    #         if len(result):
    #             result = sorted(result, key=lambda x: x["probability"], reverse=True)
    #             rule_result.append(result[0])
    #     return rule_result

    @time_spend
    def parser(self, text, session_id=None, slots=None):
        _ = '快捷键军军军军军军军军军军军军军1'
        print(_), logging.info(_)
        """
        计算 与 训练集 里面问题的相似度，取最大的值（NLU部分基本完成✌）
        """
        # rule_result = self.get_result_by_rule(text)
        rule_result = []
        # 若检测到纯标点符号的话，直接返回
        if self.has_useful_text(text):
            _ = f"no useful text: {text}"
            print(_), logging.info(_)
            return {"text": text, "intents": [], "questions": [], "slots": []}
        _ = '跳过实体抽取slots'
        print(_), logging.info(_)
        # 有实体抽取需求，优先实体抽取，并直接返回！！！
        # if slots is not None and len(slots):
        #     _ = f"优先实体抽取，并直接返回！: {text}"
        #     print(_), logging.info(_)
        #     result = self.get_slots_result(text, slots)
        #     # 抽取到了就返回，没有的化，还是走一遍意图
        #     model_result = sn.get_ner_result(text, slots)
        #     sys_slots = [{"name": k, "value": v[0]["origin"]} for _ in model_result for k, v in _.items()]
        #     result += sys_slots
        #     if result:
        #         text = sn.get_norm_text(text, model_result)
        #         return {"text": text, "intents": [], "questions": [], "slots": result}

        # _ = '快捷键军军军军军军军军军军军军军3'
        # print(_), logging.info(_)
        robot_id = self.robot_id
        version = self.version
        # 对话流  当前没有（去除）
        f_result = []

        # 调用意图识别 模块（http调用）
        labels = None
        _ = '跳过实体抽取use_intent'
        print(_), logging.info(_)
        # if self.use_intent:
        #     _ = "优先实体抽取，并直接返回！: {text}"
        #     print(_), logging.info(_)
        #     intent_result = self.get_intent_result(text, session_id=None, )
        #     print(f"intent_result: {intent_result}")
        #     try:
        #         labels = [_[0] for _ in intent_result if _[1] >= 0.2]
        #     except Exception as e:
        #         print(repr(e))
        #         labels = None

        # query 预处理；
        _ = '跳过query 预处理'
        print(_), logging.info(_)
        # text_a = list(jieba.cut(text))
        # _ = f"text normal:{text}, text seg:{' '.join(text_a)}"
        # print(_), logging.info(_)
        # text_a_embedding = get_sentence_embedding(text.lower(), log_time=True)
        # text_a_embedding_b = bert_4_layer.get_embedding(text.lower(), log_time=True)
        #
        # # 闲聊模块 当前模块没有
        # c_result = []

        labels = None
        # 矩阵计算相似度
        _ = '跳过矩阵计算相似度'
        print(_), logging.info(_)
        # if labels is None:
        #     print("no intent"), logging.info("no intent")
        #     similar_texts = [st for sts in self.similar_texts.values() for st in sts]
        #     question_answers = [qa for qas in self.question_answers.values() for qa in qas]
        #     embedding_word2vec = [ve for ves in self.embedding_word2vec.values() for ve in ves]
        #     embedding_bert4 = [ve for ves in self.embedding_bert4.values() for ve in ves]
        #     question_index = [qi for qis in self.question_index.values() for qi in qis]
        # else:
        #     similar_texts = [st for l in labels if l in self.similar_texts for st in self.similar_texts[l]]
        #     question_answers = [qa for l in labels if l in self.question_answers for qa in self.question_answers[l]]
        #     embedding_word2vec = [ve for l in labels if l in self.embedding_word2vec for ve in
        #                           self.embedding_word2vec[l]]
        #     embedding_bert4 = [ve for l in labels if l in self.embedding_bert4 for ve in self.embedding_bert4[l]]
        #     question_index = [qi for l in labels if l in self.question_index for qi in self.question_index[l]]
        #
        # if len(similar_texts) == 0:
        #     q_result = []
        # else:
        #     # 规则，如果字数较少，意图识别为闲聊是，添加一个
        #     _ = '规则，如果字数较少，意图识别为闲聊是，添加一个'
        #     print(_), logging.info(_)
        #     j_similarities = [j_similarity(text_a, _.split(" ")) for _ in similar_texts]
        #     # w_similarities = cosine_similarity(X=[text_a_embedding], Y=vec_embeddings)[0]
        #     w_similarities = (np.array([text_a_embedding]) * np.array(embedding_word2vec)).sum(axis=1)
        #     b_similarities = (np.array([text_a_embedding_b]) * np.array(embedding_bert4)).sum(axis=1)
        #     similarities = [i * 0.25 + j * 0.3 + k * 0.45 for i, j, k in
        #                     zip(j_similarities, w_similarities, b_similarities)]
        #
        #     q_result = {}
        #     for id, answer, sim_text, similarity in zip(question_index, question_answers, similar_texts, similarities):
        #         if id not in q_result or q_result[id]['probability'] < similarity:
        #             q_result[id] = {
        #                 "intent": self.id2label[id],
        #                 "q_id": id,
        #                 "answer": answer,
        #                 "probability": similarity,
        #                 "mode": "similarity",
        #                 "similar_text": sim_text.replace(" ", ""),
        #             }
        #
        #     q_result = list(q_result.values())

        self.use_similarity=False
        _ = '跳过对top 5 候选问题进行重排序'
        print(_), logging.info(_)
        # if len(q_result) and self.use_similarity:
        #     # 对top 5 候选问题进行重排序
        #     q_result = sorted(q_result, key=lambda x: x["probability"], reverse=True)
        #     sorted_top_k = 5
        #     # 调用相似度模块，优化 (目前流程是通的，还需优化 相似度模型效果，这块浮动较大，还是使用3层 roberta-wwm-3 进行)
        #     text_bs = [i["similar_text"].replace(" ", "") for i in q_result[:sorted_top_k]]
        #     text_as = [text] * len(text_bs)
        #     similarity_result = self.get_similar_result(text_as, text_bs)
        #     print(f"similarity_result:{similarity_result}")
        #     if similarity_result is not None:
        #         for index, i in enumerate(q_result[:sorted_top_k]):
        #             # i["probability"] = i["probability"] * 0.3 + float(ss_[index]) * 0.7
        #             i["probability"] = i["probability"] * 0.5 + float(similarity_result[index]) * 0.5
        #     q_result = sorted(q_result[:sorted_top_k], key=lambda x: x["probability"], reverse=True)
        _ = '进入NLU_parser-开始获取句向量'
        print(_), logging.info(_)
        text_embedding_sentence_bert = self.get_similar_result(text)
        text_embedding_sentence_bert = torch.tensor(text_embedding_sentence_bert,dtype=torch.float32)
        _ = '结束NLU_parser-开始获取句向量'
        print(_), logging.info(_)
        print('*'*80)
        print('*'*80)
        # print(text_embedding_sentence_bert)
        print('*'*80)
        print('*'*80)
        _ = '跳过模型 微调 问答相似度'
        print(_), logging.info(_)
        # 模型 微调 问答相似度
        # try:
        #     print('测试模块1')
        #     result_total = f_result + c_result + q_result
        #     print('测试模块2')
        #     result = sorted(result_total, key=lambda x: x["probability"], reverse=True)
        #     print('测试模块3')
        #     intents = []
        #     questions = []
        #     for i in result:
        #         print('测试模块有intent在result里')
        #         if i["intent"] == "qa":
        #             print('当意图是问题的时候')
        #             questions.append(
        #                 {"id": i["q_id"], "similarity": scale_to_float(i["probability"]), "answer": i["answer"],
        #                  "question": self.id2title[i["q_id"]], "similarText": i["similar_text"], })
        #         else:
        #             # 这边是 意图的识别结果 rule_result 为规则的处理结果
        #             print('这边是 意图的识别结果 rule_result 为规则的处理结果')
        #             for _ in rule_result:
        #                 if _["intent"] == i["intent"] and _["probability"] > i["probability"]:
        #                     i["probability"] = _["probability"]
        #                     break
        #             intents.append({"name": i["intent"], "similarity": scale_to_float(i["probability"]),
        #                             "similarText": i["similar_text"]})
        # except:
        #     traceback.print_exc()

        #######################################################
        #######################################################
        intents=[]
        # model_path=f"config_models/robot_{robot_id}_version_{version}_similarity_sentbert"
        # model_path='pretrained_models/distiluse-base-multilingual-cased-v2'
        intents.append(changshu_(text_embedding_sentence_bert,self.sentence_total,self.embedding_sentence_bert,self.map_total))
        # print("*"*80)
        # print(intents)
        # print("*"*80)
        #######################################################
        #######################################################
        print('*'*80)
        _ = {"text": text, "intents": intents, "questions": [], "slots": []}
        print(_), logging.info(_)
        print('*'*80)
        return {"text": text, "intents": intents, "questions": [], "slots": []}
        # return {"text": text, "intents": intents, "questions": questions, "slots": []}


if __name__ == "__main__":
    import pickle
    import time
    import pandas as pd


    robot_id_ = 155
    version_ = 28
    c = pickle.load(open(f"./config_models/robot_{robot_id_}_version_{version_}.model", "rb"))
    nlu = NLU(config=c)
    result_ = nlu.parser('报告厅在哪？')
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.parser('你是谁呀？')
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.parser('你是谁？')
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.parser('嗯好的？')
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.parser('好的？')
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.parser('是的')
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.parser('嗯是的')
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.parser('我没问题的')
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.parser('你好，车牌号位 浙A 一二三四5，颜色是黑色', slots=["car_num"])
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.parser('你好，车牌号位车牌号是婉J八巴七二三', slots=["car_num"])
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.get_slots_result('张三在文一西路停车，堵住路了，需要挪一下，车牌号是浙A34678？', ["name", "location", "车牌号"])
    print("1", json.dumps(str(result_), ensure_ascii=False))

    result_ = nlu.get_slots_result('余杭街道', ["address", "location", "车牌号"])
    print("1", json.dumps(str(result_), ensure_ascii=False))
