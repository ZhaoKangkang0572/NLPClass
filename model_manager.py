#!/usr/bin/python
# coding:utf8

"""
@author: Cong Yu
@time: 2020-03-06 15:53
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re
import json
import signal
import traceback
import multiprocessing
import logging.handlers
from flask_cors import CORS
from flask import Flask, request, current_app, send_from_directory, render_template
# from predict_intent_roberta import IntentModel
# from predict_similarity_bert_esim import BertSimilarityModel

from nlu.intent import BertIntent, TrainConfig, DataProcessor
from nlu.similarity import BertSimilarity, TrainConfig, DataProcessor
from nlu.sentence_bert_similarity import SentenceBertSimilarity

from utils.sftp_utils import SftpServer, HOST, USER, PASSWORD, PORT, ORIGIN_PATH
from utils.redis_pool import master as redis
from sentence_transformers import SentenceTransformer, util


def kill(pid):
    try:
        a = os.kill(pid, signal.SIGKILL)
        print('已杀死pid为%s的进程,　返回值是:%s' % (pid, a))

    except OSError:
        print('没有如此进程!!!')


# ftp 版本需要 提供下载模型的接口
"""
模型管理层（多进程版）
"""

if not os.path.exists("./log"):
    print("not exits...")
    os.makedirs("./log")

app = Flask(__name__)
CORS(app, supports_credentials=True)

logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
logger.setLevel(logging.NOTSET)
handler = logging.handlers.TimedRotatingFileHandler('./log/tf_service.log', when='D', interval=1)
handler.setFormatter(formatter)
logger.addHandler(handler)


class PredictSimilarityServer:

    def __init__(self):
        self.robot_version = {}

    def func(self, conn, robot_id, version):  # conn管道类型
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        import tensorflow as tf
        has_model = False
        key = f"{robot_id}-{version}"
        try:
            # robot_id = 1
            # version = 1
            config = TrainConfig(f"config_models/robot_{robot_id}_version_{version}_similarity/bert_config.json",
                                 ["0", "1"],
                                 max_seq_length=64)
            dp = DataProcessor(f"config_models/robot_{robot_id}_version_{version}_similarity/vocab.txt",
                               ["0", "1"],
                               max_seq_length=64)
            bsm = BertSimilarity(config, dp)
            bsm.load_model(f"config_models/robot_{robot_id}_version_{version}_similarity")
            # bsm = BertSimilarityModel(robotid=robot_id, version=version)
            a = bsm.predict_batch(["今日疫情"], ["今天天气怎么样"])
            has_model = True

            # sentence_bert_model = SentenceTransformer(f"config_models/robot_{robot_id}_version_{version}_similarity_sentbert")

            print(a)
        except:
            # traceback.print_exc()
            print("model not found...")
        while True:
            temp = conn.recv()
            flag_ = True
            try:
                json_data = json.loads(temp)
                if json_data["robot_id"] == robot_id and json_data["version"] == version:
                    if has_model:
                        res = bsm.predict_batch(json_data["text_as"], json_data["text_bs"])
                        res = [float(_) for _ in res]
                    else:
                        res = "not found model"
                elif json_data["robot_id"] == robot_id:
                    flag_ = False
                    res = "update model"
                else:
                    # 目前设置更新模型时，把之前的模型关闭掉
                    res = "sorry, robot_id not match"
            except Exception as e:
                print(repr(e))
                res = "sorry, error"
            # print(res)
            conn.send(res)  # 发送的数据
            flag = flag_ and self.robot_version.get(key, None) is not None and self.robot_version[key][3]
            print(f"进程状态：{flag}，子进程：{os.getpid()} ，接受数据：{json_data}，返回：{res}")

    def predict_batch(self, text_as, text_bs, robot_id, version):
        json_data = {
            "robot_id": robot_id,
            "version": version,
            "text_as": text_as,
            "text_bs": text_bs
        }
        json_data_string = json.dumps(json_data)
        key = f"{robot_id}-{version}"
        # 检测，所有死掉的进程，全部下线
        pops = []
        for k, v in self.robot_version.items():
            if not v[2].is_alive():
                pops.append(k)
        [self.robot_version.pop(_) for _ in pops]
        if self.robot_version.get(key, None):
            print("has model")
            [conn_a, conn_b, p, _] = self.robot_version[key]
            if p.is_alive():
                conn_b.send(json_data_string)
                try:
                    a = conn_b.recv()
                except:
                    a = "model is ..."
            else:
                a = "model is close"
            return a
        else:
            print(f"init model {robot_id}-{version}")
            conn_a, conn_b = multiprocessing.Pipe()  # 创建一个管道，两个口
            p = multiprocessing.Process(target=self.func, args=(conn_a, robot_id, version))
            p.daemon = True
            self.robot_version[key] = [conn_a, conn_b, p, True]
            p.start()
            self.robot_version[key] = [conn_a, conn_b, p, True]
            conn_b.send(json_data_string)
            a = conn_b.recv()
            # 其他版本 下线
            online_robot_versions = str(redis.get("online_robot_versions").decode()).split(";")
            pops = []
            for k, v in self.robot_version.items():
                if key != k and re.search(f"^{robot_id}-", k) and k.replace("-", ":") not in online_robot_versions:
                    # v[2].terminate()
                    # print("stop process")
                    # v[2].join()
                    _ = f"pid:{v[2].pid}, name:{v[2].name}"
                    print(_), logging.info(_)
                    kill(v[2].pid)
                    pops.append(k)
                    # pops.append(k)
            [self.robot_version.pop(_) for _ in pops]
            return a

    def predict_batch_sentence_bert(self, text_as, robot_id, version):
        json_data = {
            "robot_id": robot_id,
            "version": version,
            "text_as": text_as,
        }
        json_data_string = json.dumps(json_data)
        key = f"{robot_id}-{version}"
        # 检测，所有死掉的进程，全部下线
        pops = []
        for k, v in self.robot_version.items():
            if not v[2].is_alive():
                pops.append(k)
        [self.robot_version.pop(_) for _ in pops]
        if self.robot_version.get(key, None):
            print("has model")
            [conn_a, conn_b, p, _] = self.robot_version[key]
            if p.is_alive():
                conn_b.send(json_data_string)
                try:
                    a = conn_b.recv()
                except:
                    a = "model is ..."
            else:
                a = "model is close"
            return a
        else:
            print(f"init model sentence nert {robot_id}-{version}")
            conn_a, conn_b = multiprocessing.Pipe()  # 创建一个管道，两个口
            p = multiprocessing.Process(target=self.func, args=(conn_a, robot_id, version))
            p.daemon = True
            self.robot_version[key] = [conn_a, conn_b, p, True]
            p.start()
            self.robot_version[key] = [conn_a, conn_b, p, True]
            conn_b.send(json_data_string)
            a = conn_b.recv()
            # 其他版本 下线
            online_robot_versions = str(redis.get("online_robot_versions").decode()).split(";")
            pops = []
            for k, v in self.robot_version.items():
                if key != k and re.search(f"^{robot_id}-", k) and k.replace("-", ":") not in online_robot_versions:
                    # v[2].terminate()
                    # print("stop process")
                    # v[2].join()
                    _ = f"pid:{v[2].pid}, name:{v[2].name}"
                    print(_), logging.info(_)
                    kill(v[2].pid)
                    pops.append(k)
                    # pops.append(k)
            [self.robot_version.pop(_) for _ in pops]
            return a

    def delete_model(self, robot_id, version):
        pops = []
        for k, v in self.robot_version.items():
            key = f"{robot_id}-{version}"
            if key == k:
                _ = f"pid:{v[2].pid}, name:{v[2].name}"
                print(_), logging.info(_)
                kill(v[2].pid)
                pops.append(k)
                # pops.append(k)
                break
        [self.robot_version.pop(_) for _ in pops]
        return True


class PredictSentenceSimilarityServer:
    def __init__(self):
        self.robot_version = {}

    def func(self, conn, robot_id, version):  # conn管道类型
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        import tensorflow as tf
        has_model = False
        key = f"{robot_id}-{version}"
        try:
            # robot_id = 1
            # version = 1
            # config = TrainConfig(f"config_models/robot_{robot_id}_version_{version}_similarity/bert_config.json",
            #                      ["0", "1"],
            #                      max_seq_length=64)
            # dp = DataProcessor(f"config_models/robot_{robot_id}_version_{version}_similarity/vocab.txt",
            #                    ["0", "1"],
            #                    max_seq_length=64)
            print('bsm_sentence_bert_model初始化开始')
            print(robot_id)
            print(version)
            bsm_sentence_bert_model = SentenceBertSimilarity(
                f"config_models/robot_{robot_id}_version_{version}_similarity_sentbert/")
            print('bsm_sentence_bert_model初始化结束')
            # bsm.load_model(f"config_models/robot_{robot_id}_version_{version}_similarity")
            # # bsm = BertSimilarityModel(robotid=robot_id, version=version)
            # a = bsm.predict_batch(["今日疫情"], ["今天天气怎么样"])
            has_model = True
            #
            # sentence_bert_model = SentenceTransformer(f"config_models/robot_{robot_id}_version_{version}_similarity_sentbert")
            #
            # print(a)
        except:
            # traceback.print_exc()
            print("model not found...")
        while True:
            temp = conn.recv()
            flag_ = True
            try:
                json_data = json.loads(temp)
                if json_data["robot_id"] == robot_id and json_data["version"] == version:
                    print("是否有模型")
                    print(has_model)
                    print(json_data["text_as"])
                    if has_model:
                        res = bsm_sentence_bert_model.predict_embedding1(json_data["text_as"])
                        print('拿到了res，返回，长度是:',len(res))
                        # print(res)
                        # res = [float(_) for _ in res]

                    else:
                        res = "not found model"
                elif json_data["robot_id"] == robot_id:
                    flag_ = False
                    res = "update model"
                else:
                    # 目前设置更新模型时，把之前的模型关闭掉
                    res = "sorry, robot_id not match"
            except Exception as e:
                print(repr(e))
                res = "sorry, error"
            # print(res)
            conn.send(res)  # 发送的数据
            flag = flag_ and self.robot_version.get(key, None) is not None and self.robot_version[key][3]
            # print(f"进程状态：{flag}，子进程：{os.getpid()} ，接受数据：{json_data}，返回：{res}")

    def predict_batch_sentence_bert(self, text_as, robot_id, version):
        json_data = {
            "robot_id": robot_id,
            "version": version,
            "text_as": text_as,
        }
        _ = f"进入predict_batch_sentence_bert方法"
        print(_), logging.info(_)
        json_data_string = json.dumps(json_data)
        key = f"{robot_id}-{version}"
        # 检测，所有死掉的进程，全部下线
        pops = []
        for k, v in self.robot_version.items():
            if not v[2].is_alive():
                pops.append(k)
        [self.robot_version.pop(_) for _ in pops]
        if self.robot_version.get(key, None):
            _ = f"模型存在"
            print(_), logging.info(_)
            _ = f"模型版本{key}"
            print(_), logging.info(_)
            _ = f"json_data_string：{json_data_string}"
            print(_), logging.info(_)
            _ = f"self.robot_version[key]:：{self.robot_version[key]}"
            print(_), logging.info(_)

            [conn_a, conn_b, p, _] = self.robot_version[key]
            print('conn_b',conn_b)
            if p.is_alive():
                conn_b.send(json_data_string)
                try:
                    a = conn_b.recv()
                except:
                    a = "model is ..."
            else:
                a = "model is close"

            _ = f"拿到并且返回了a：{a}"
            print(_), logging.info(_)
            # print(len(a[0]))
            # print()
            return a
        else:
            print(f"init model sentence nert {robot_id}-{version}")
            conn_a, conn_b = multiprocessing.Pipe()  # 创建一个管道，两个口
            p = multiprocessing.Process(target=self.func, args=(conn_a, robot_id, version))
            p.daemon = True
            self.robot_version[key] = [conn_a, conn_b, p, True]
            p.start()
            self.robot_version[key] = [conn_a, conn_b, p, True]
            conn_b.send(json_data_string)
            a = conn_b.recv()
            # 其他版本 下线
            online_robot_versions = str(redis.get("online_robot_versions").decode()).split(";")
            pops = []
            for k, v in self.robot_version.items():
                if key != k and re.search(f"^{robot_id}-", k) and k.replace("-", ":") not in online_robot_versions:
                    # v[2].terminate()
                    # print("stop process")
                    # v[2].join()
                    _ = f"pid:{v[2].pid}, name:{v[2].name}"
                    print(_), logging.info(_)
                    kill(v[2].pid)
                    pops.append(k)
                    # pops.append(k)
            [self.robot_version.pop(_) for _ in pops]
            return a

    def delete_model(self, robot_id, version):
        pops = []
        for k, v in self.robot_version.items():
            key = f"{robot_id}-{version}"
            if key == k:
                _ = f"pid:{v[2].pid}, name:{v[2].name}"
                print(_), logging.info(_)
                kill(v[2].pid)
                pops.append(k)
                # pops.append(k)
                break
        [self.robot_version.pop(_) for _ in pops]
        return True


class PredictIntentServer:

    def __init__(self):
        self.robot_version = {}

    def func(self, conn, robot_id, version):  # conn管道类型
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        import tensorflow as tf
        has_model = False
        key = f"{robot_id}-{version}"
        try:
            # robot_id = 1
            # version = 1
            labels = [_.strip() for _ in
                      open(f"config_models/robot_{robot_id}_version_{version}_intent/label.txt").readlines()]
            config = TrainConfig(f"config_models/robot_{robot_id}_version_{version}_intent/bert_config.json", labels,
                                 max_seq_length=64)
            dp = DataProcessor(f"config_models/robot_{robot_id}_version_{version}_intent/vocab.txt",
                               labels,
                               max_seq_length=64)
            bsm = BertIntent(config, dp)
            bsm.load_model(f"config_models/robot_{robot_id}_version_{version}_intent")

            # bsm = IntentModel(robotid=robot_id, version=version)
            a = bsm.predict("今日疫情")
            has_model = True
            print(a)
        except:
            print("model not found...")
        while True:
            temp = conn.recv()
            print("rec ok1")
            flag_ = True
            try:
                json_data = json.loads(temp)
                if json_data["robot_id"] == robot_id and json_data["version"] == version:
                    if has_model:
                        res = bsm.predict(json_data["text"])
                        # res = [float(_) for _ in res]
                    else:
                        res = "not found model"
                elif json_data["robot_id"] == robot_id:
                    flag_ = False
                    res = "update model"
                else:
                    # 目前设置更新模型时，把之前的模型关闭掉
                    res = "sorry, robot_id not match"
            except Exception as e:
                print(repr(e))
                res = "sorry, error"
            # print(res)
            conn.send(res)  # 发送的数据
            print("rec ok2")
            flag = flag_ and self.robot_version.get(key, None) is not None and self.robot_version[key][3]
            print(f"进程状态：{flag}，子进程：{os.getpid()} ，接受数据：{json_data}，返回：{res}")

    def predict(self, text, robot_id, version):
        json_data = {
            "robot_id": robot_id,
            "version": version,
            "text": text
        }
        json_data_string = json.dumps(json_data)
        key = f"{robot_id}-{version}"
        # 检测，所有死掉的进程，全部下线
        pops = []
        for k, v in self.robot_version.items():
            if not v[2].is_alive():
                pops.append(k)
        [self.robot_version.pop(_) for _ in pops]
        if self.robot_version.get(key, None):
            print("has model")
            [conn_a, conn_b, p, _] = self.robot_version[key]
            if p.is_alive():
                conn_b.send(json_data_string)
                a = conn_b.recv()
            else:
                a = "model is close"
            return a
        else:
            print(f"init model {robot_id}-{version}")
            conn_a, conn_b = multiprocessing.Pipe()  # 创建一个管道，两个口
            print("ok1")
            p = multiprocessing.Process(target=self.func, args=(conn_a, robot_id, version))
            p.daemon = True
            self.robot_version[key] = [conn_a, conn_b, p, True]
            p.start()
            self.robot_version[key] = [conn_a, conn_b, p, True]
            print("ok2")
            conn_b.send(json_data_string)
            print("ok3")
            a = conn_b.recv()
            print("ok4")
            # 其他版本 下线
            online_robot_versions = str(redis.get("online_robot_versions").decode()).split(";")
            pops = []
            for k, v in self.robot_version.items():
                if key != k and re.search(f"^{robot_id}-", k) and k.replace("-", ":") not in online_robot_versions:
                    # v[2].terminate()
                    # print("stop process")
                    # v[2].join()
                    # 可能需要保留线上的版本，不要杀掉当前进程（相当于，一个线上，一个当前测试的可以保留）
                    _ = f"pid:{v[2].pid}, name:{v[2].name}"
                    print(_), logging.info(_)
                    kill(v[2].pid)
                    pops.append(k)
            [self.robot_version.pop(_) for _ in pops]
            return a

    def delete_model(self, robot_id, version):
        pops = []
        for k, v in self.robot_version.items():
            key = f"{robot_id}-{version}"
            if key == k:
                _ = f"pid:{v[2].pid}, name:{v[2].name}"
                print(_), logging.info(_)
                kill(v[2].pid)
                pops.append(k)
                # pops.append(k)
                break
        [self.robot_version.pop(_) for _ in pops]
        return True


# pss_model = PredictSimilarityServer()
pss_model = ''
print("pss_sentence_bert_model 开始创建")
pss_sentence_bert_model = PredictSentenceSimilarityServer()
print("pss_sentence_bert_model 结束创建")

pis_model = PredictIntentServer()
ctx = app.app_context()
ctx.push()


@app.route('/api/model_intent', methods=['GET', 'POST'])
def query_model_intent():
    # try:
    def form_or_json():
        if request.get_json(silent=True):
            return request.get_json(silent=True)
        else:
            if request.form:
                return request.form
            else:
                return request.args

    data = form_or_json()

    # print(json.loads(data))

    def fun():
        # try:
        res = {}
        for k, v in pis_model.robot_version.items():
            res[k] = v[2].is_alive()
        return {"code": 0, "msg": "ok", "data": res}

    if request.method == "POST" or request.method == "GET":
        result = fun()
        return json.dumps(result, ensure_ascii=False)

    else:
        result = {"code": -1, "msg": "请求异常！"}
        return json.dumps(result, ensure_ascii=False)


@app.route('/api/model_similar', methods=['GET', 'POST'])
def query_model_similar():
    # try:
    def form_or_json():
        if request.get_json(silent=True):
            return request.get_json(silent=True)
        else:
            if request.form:
                return request.form
            else:
                return request.args

    data = form_or_json()

    def fun():
        res = {}
        for k, v in pss_model.robot_version.items():
            res[k] = v[2].is_alive()
        return {"code": 0, "msg": "ok", "data": res}

    if request.method == "POST" or request.method == "GET":
        result = fun()
        return json.dumps(result, ensure_ascii=False)

    else:
        result = {"code": -1, "msg": "请求异常！"}
        return json.dumps(result, ensure_ascii=False)


@app.route('/api/model_query', methods=['GET', 'POST'])
def query_models():
    # try:
    def form_or_json():
        if request.get_json(silent=True):
            return request.get_json(silent=True)
        else:
            if request.form:
                return request.form
            else:
                return request.args

    data = form_or_json()

    # print(json.loads(data))

    def fun():
        # try:
        _ = f"api_model_query :根据pss_sentence_bert_model获取机器人版本号以及模型是否活着.."
        print(_), logging.info(_)
        _ = pss_sentence_bert_model.robot_version.get(f"{robot_id}-{version}", None)
        res = {
            "intent_model": _[2].is_alive() if _ else -1
        }
        _ = pss_sentence_bert_model.robot_version.get(f"{robot_id}-{version}", None)
        res["similarity_model"] = _[2].is_alive() if _ else -1
        return {"code": 0, "msg": "ok", "data": res}

    if request.method == "POST" or request.method == "GET":
        robot_id = str(data['robotId'])
        version = str(data['version'])
        _ = f"api_model_query {robot_id}-{version} .."
        print(_), logging.info(_)
        result = fun()
        return json.dumps(result, ensure_ascii=False)

    else:
        result = {"code": -1, "msg": "请求异常！"}
        return json.dumps(result, ensure_ascii=False)


@app.route('/api/delete_robot', methods=['GET', 'POST'])
def delete_robot():
    # try:
    def form_or_json():
        if request.get_json(silent=True):
            return request.get_json(silent=True)
        else:
            if request.form:
                return request.form
            else:
                return request.args

    data = form_or_json()

    def fun():
        robot_version = list(pis_model.robot_version.keys())
        for k in robot_version:
            _ = k.split("-")
            if _[0] == str(robot_id):
                pis_model.delete_model(_[0], _[1])

        robot_version = list(pss_model.robot_version.keys())
        for k in robot_version:
            _ = k.split("-")
            if _[0] == str(robot_id):
                pss_model.delete_model(_[0], _[1])
        return {"code": 0, "msg": "ok"}

    if request.method == "POST" or request.method == "GET":
        try:
            robot_id = str(data['robotId'])
            result = fun()
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            traceback.print_exc()
            result = {"code": -1, "msg": repr(e)}
            return json.dumps(result, ensure_ascii=False)
    else:
        result = {"code": -1, "msg": "请求异常！"}
        return json.dumps(result, ensure_ascii=False)


# @app.route('/api/model_predict', methods=['GET', 'POST'])
# def predict_model_old():
#     # try:
#     def form_or_json():
#         if request.get_json(silent=True):
#             return request.get_json(silent=True)
#         else:
#             if request.form:
#                 return request.form
#             else:
#                 return request.args
#
#     data = form_or_json()
#
#     def fun():
#         if model_name == "intent_model":
#             text = data.get("text", None)
#             if text is None:
#                 return {"code": -1, "msg": "参数错误"}
#             res = pis_model.predict(text, robot_id, version)
#             if not isinstance(res, str):
#                 return {"code": 0, "msg": "ok", "data": res}
#             else:
#                 return {"code": -1, "msg": "不可用"}
#         elif model_name == "similar_model":
#             text_as = data.get("text_as", None)
#             text_bs = data.get("text_bs", None)
#             if text_as is None or text_bs is None:
#                 return {"code": -1, "msg": "参数错误"}
#             res = pss_model.predict_batch(text_as, text_bs, robot_id, version)
#             if not isinstance(res, str):
#                 return {"code": 0, "msg": "ok", "data": res}
#             else:
#                 return {"code": -1, "msg": "不可用"}
#         else:
#             return {"code": -1, "msg": "模型类型错误"}
#
#     if request.method == "POST" or request.method == "GET":
#         robot_id = str(data['robotId'])
#         version = str(data['version'])
#         model_name = data['modelName']
#         result = fun()
#         return json.dumps(result, ensure_ascii=False)
#
#     else:
#         result = {"code": -1, "msg": "请求异常！"}
#         return json.dumps(result, ensure_ascii=False)

@app.route('/api/model_predict', methods=['GET', 'POST'])
def predict_model():
    # try:
    def form_or_json():
        if request.get_json(silent=True):
            return request.get_json(silent=True)
        else:
            if request.form:
                return request.form
            else:
                return request.args

    data = form_or_json()

    def fun():
        _ = '进入model_manager的fun()'
        print(_), logging.info(_)

        if model_name == "intent_model":
            _ = '进入intent_model'
            print(_), logging.info(_)
            text = data.get("text", None)
            if text is None:
                return {"code": -1, "msg": "参数错误"}
            res = pis_model.predict(text, robot_id, version)
            if not isinstance(res, str):
                return {"code": 0, "msg": "ok", "data": res}
            else:
                return {"code": -1, "msg": "不可用"}
        elif model_name == "similar_model":
            _ = '进入similar_model'
            print(_), logging.info(_)
            text_as = data.get("text_as", None)
            text_bs = data.get("text_bs", None)
            if text_as is None or text_bs is None:
                return {"code": -1, "msg": "参数错误"}
            # res = pss_model.predict_batch(text_as, text_bs, robot_id, version)
            res = pss_sentence_bert_model.predict_batch_sentence_bert(text_as, robot_id, version)
            if not isinstance(res, str):
                return {"code": 0, "msg": "ok", "data": res}
            else:
                return {"code": -1, "msg": "不可用"}
        elif model_name == 'sentence_bert':
            _ = '进来sentenceBert'
            print(_), logging.info(_)
            text_as = data.get("text_as", None)
            if text_as is None:
                return {"code": -1, "msg": "参数错误"}
            _ = f"文本是：: {text_as}"
            print(_), logging.info(_)
            _ = f"机器人ID：: {robot_id}"
            print(_), logging.info(_)
            _ = f"version：: {version}"
            print(_), logging.info(_)
            res = pss_sentence_bert_model.predict_batch_sentence_bert(text_as, robot_id, version)
            if not isinstance(res, str):
                return {"code": 0, "msg": "ok", "data": res}
            else:
                return {"code": -1, "msg": "不可用"}
        else:
            return {"code": -1, "msg": "模型类型错误"}

    if request.method == "POST" or request.method == "GET":
        robot_id = str(data['robotId'])
        version = str(data['version'])
        model_name = data['modelName']
        result = fun()
        return json.dumps(result, ensure_ascii=False)

    else:
        result = {"code": -1, "msg": "请求异常！"}
        return json.dumps(result, ensure_ascii=False)


@app.route('/api/download_model', methods=['GET', 'POST'])
def download_model():
    # try:
    def form_or_json():
        if request.get_json(silent=True):
            return request.get_json(silent=True)
        else:
            if request.form:
                return request.form
            else:
                return request.args

    data = form_or_json()

    # print(json.loads(data))

    def fun():
        try:
            sftp = SftpServer(HOST, USER, PASSWORD, PORT)
            sftp.get_file(f"./{ORIGIN_PATH}/config_models/robot_{robot_id}_version_{version}_label.txt",
                          f"./config_models/robot_{robot_id}_version_{version}_label.txt")
            sftp.get_file(f"./{ORIGIN_PATH}/config_models/robot_{robot_id}_version_{version}_v2.model",
                          f"./config_models/robot_{robot_id}_version_{version}_v2.model")

            # 原始 ckpt 模式
            sftp.get_dir(f"./{ORIGIN_PATH}/config_models/robot_{robot_id}_version_{version}",
                         f"./config_models/robot_{robot_id}_version_{version}")
            sftp.get_dir(f"./{ORIGIN_PATH}/config_models/robot_{robot_id}_version_{version}_bert_esim",
                         f"./config_models/robot_{robot_id}_version_{version}_bert_esim")
            sftp.close()
            return {"code": 0, "msg": "ok"}
        except:
            return {"code": -1, "msg": "下载失败"}

    if request.method == "POST" or request.method == "GET":
        robot_id = str(data['robotId'])
        version = str(data['version'])
        result = fun()
        return json.dumps(result, ensure_ascii=False)

    else:
        result = {"code": -1, "msg": "请求异常！"}
        return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    1
    app.run(host='0.0.0.0', port="8010", threaded=False, processes=1,debug=True)
    # app.run(host='0.0.0.0', port="8010", threaded=True, processes=1)
