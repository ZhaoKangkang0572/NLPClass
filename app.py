#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-11-04 11:12
"""
import os
import re
import json
import uuid
import time
import requests
import logging.handlers
import traceback
# from robot_config.config import Config
from datetime import datetime#, timedelta
from flask_cors import CORS
from flask import Flask, request# current_app, send_from_directory, render_template
from utils.mysql_pool import pool, CLUSTER, TABLE_NAME
# from utils.es_util import insert_data, get_data, get_data_by_id, es_dict, BRANCH, INDEX_NAME
from utils.mongo_util import collection
from utils.other import time_spend
from nlu.server_nlu import sn, redis, MODEL_MANAGER_HOST
from robot_config.robot_manager import RobotManager

if not os.path.exists("./log"):
    _ = "log not exits..."
    print(_), logging.info(_)
    os.makedirs("./log")

# flask 设置 静态资源路径与url访问路径
app = Flask(__name__, static_folder='static', static_url_path="/")
CORS(app, supports_credentials=True)

# 机器人管理
robot_manager = RobotManager()

logger = logging.getLogger()
# 时间 + 级别 +  地址/函数  + uuid + 信息
formatter = logging.Formatter('%(message)s')
logger.setLevel(logging.NOTSET)
handler = logging.handlers.TimedRotatingFileHandler('./log/service.log', when='D', interval=1)
handler.setFormatter(formatter)
logger.addHandler(handler)


#  获取一个分布式锁
def acquire_lock(lock_name, acquire_time=2, time_out=10):
    # 生成唯一id
    identifier = str(uuid.uuid4())
    # 客户端获取锁的结束时间
    end = time.time() + acquire_time
    # key
    lock_names = "lock_name:" + lock_name
    while time.time() < end:
        # setnx(key,value) 只有key不存在情况下，将key的值设置为value，若key存在则不做任何动作,返回True和False
        if redis.setnx(lock_names, identifier):
            # 设置键的过期时间，过期自动剔除，释放锁
            redis.expire(lock_names, time_out)
            return identifier
        # 当锁未被设置过期时间时，重新设置其过期时间
        elif redis.ttl(lock_names) == -1:
            redis.expire(lock_names, time_out)
        time.sleep(0.001)
    return False


# 锁的释放
def release_lock(lock_name, identifire):
    lock_names = "lock_name:" + lock_name
    pipe = redis.pipeline(True)
    while True:
        try:
            # 通过watch命令监视某个键，当该键未被其他客户端修改值时，事务成功执行。当事务运行过程中，发现该值被其他客户端更新了值，任务失败
            pipe.watch(lock_names)
            if pipe.get(lock_names) == identifire:  # 检查客户端是否仍然持有该锁
                # multi命令用于开启一个事务，它总是返回ok
                # multi执行之后， 客户端可以继续向服务器发送任意多条命令， 这些命令不会立即被执行， 而是被放到一个队列中， 当 EXEC 命令被调用时， 所有队列中的命令才会被执行
                pipe.multi()
                # 删除键，释放锁
                pipe.delete(lock_names)
                # execute命令负责触发并执行事务中的所有命令
                pipe.execute()
                return True
            pipe.unwatch()
            break
        except:
            # # 释放锁期间，有其他客户端改变了键值对，锁释放失败，进行循环
            pass
    return False


# ----------------------------  v2  ----------------------------
# 这个基本不用改 @1
@app.route('/api/algorithm/train', methods=['GET', 'POST'])
def algorithm_train():
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

    @time_spend
    def fun():
        # robot_id=1, version=1 默认，固定版本
        _ = f"机器编号：{robot_id}, 版本编号：{version}"
        print(_), logging.info(_)
        conn = pool.connection()  # 以后每次需要数据库连接就是用connection（）函数获取连接就好了
        cur = conn.cursor()
        identifier = False
        lock_name = f"lock_name_{robot_id}_{version}"
        try:
            # 加锁, 修改为1s ，拿不到就直接返回了
            identifier = acquire_lock(lock_name, acquire_time=1, time_out=10)
            if not identifier:
                return {"code": -2, "data": {}, "msg": f"{robot_id}, {version}并发锁异常！"}
            # step-1, 检测 机器、版本是否存在
            sql = f"SELECT robot_id,version_id,status " \
                  f"FROM {TABLE_NAME} " \
                  f"where robot_id='{robot_id}' and version_id='{version}' and DELETE_FLAG=0 and CLUSTER='{CLUSTER}';"
            cur.execute(sql)
            result_ = cur.fetchall()
            if len(result_):
                # print(result_[0])
                return {"code": -3, "data": {}, "msg": "机器和版本已存在！"}

            sql = f"SELECT robot_id,version_id,status " \
                  f"FROM {TABLE_NAME} " \
                  f"where DELETE_FLAG=0 and CLUSTER='{CLUSTER}';"
            cur.execute(sql)
            result_ = cur.fetchall()
            data = []
            for i in result_:
                data.append({
                    "robot_id": i[0],
                    "version": i[1],
                    "status": i[2],
                })
            robot_versions = {}
            for i in data:
                robot_versions[i["robot_id"]] = robot_versions.get(i["robot_id"], []) + [int(i["version"])]

            _ = f"当前机器版本：{robot_versions}"
            print(_), logging.info(_)
            if len(robot_versions) >= 15 and robot_id not in robot_versions:
                return {"code": -4, "data": {}, "msg": "机器人数量限制，目前不能超过15！"}

            # step-2, 检测，技能组件是否存在，不存在则提示，存在则插入

            # step-3，通过后，保存config model 文件，后面加载调用 （非常耗时的任务）
            config = {
                # 机器人ID
                "robotId": robot_id,
                # 版本ID
                "version": version,
                # 知识库问答
                "questions": questions,
                # 意图识别库
                "intents": intents,
                # 槽位抽取库
                "slots": slots,
            }

            # 修改为，数据库插入任务数据
            col = "robot_id,version_id,es_id,es_link,status," \
                  "CREATED_BY,CREATED_AT,UPDATED_BY,UPDATED_AT,DELETE_FLAG,CLUSTER"
            t1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            doc_ = {
                "robot": robot_id,
                "version": version,
                "json_data": json.dumps(config, ensure_ascii=False),
            }
            # 2020-12-05 改成mongodb进行读取
            # 返回es 插入的id
            # a_ = insert_data(doc_)
            # if a_ is None:
            #     return {"code": -7, "data": {}, "msg": "插入数据错误！"}
            # es_id = a_["_id"]
            # es_link = f'{es_dict[BRANCH]["url"]}/{INDEX_NAME}/_doc/{es_id}'
            # print(es_link)
            es_link = ""
            try:
                # 修改使用mongodb 进行存储训练机器人数据
                insert_one_result = collection.insert_one(config)
                es_id = str(insert_one_result.inserted_id)
            except Exception as e:
                print(repr(e)), logging.error(repr(e))
                return {"code": -7, "data": {}, "msg": "插入数据错误！"}
            # 2020-12-05 改成mongodb进行读取
            sql = f'insert into {TABLE_NAME}({col}) ' \
                  f'VALUES("{robot_id}", "{version}", "{es_id}", "{es_link}",  1, "实在科技", NOW(), ' \
                  f'"实在科技", NOW(), 0, "{CLUSTER}")'
            # print(sql), logging.info(sql)
            index = cur.execute(sql)
            conn.commit()
        except Exception as e:
            traceback.print_exc()
            return {"code": -1, "data": {}, "msg": repr(e)}
        finally:
            if identifier:
                release_lock(lock_name, identifier)
            else:
                return {"code": -2, "data": {}, "msg": f"{robot_id}, {version}并发锁异常！"}
            cur.close()
            conn.close()
        return {"code": 0, "msg": "ok！"}

    if request.method == "POST" or request.method == "GET":
        try:
            print("\n")
            _ = "/api/algorithm/train"
            print(_), logging.info(_)
            # 新增 训练的字段校验机制，不合规的提示失败
            robot_id = int(data['robotId'])
            version = int(data['version'])

            slots = data["slots"]
            if len(slots):
                # 校验格式
                try:
                    for slot in slots:
                        assert type(slot["name"]) is str
                        assert type(slot["patterns"]) is list
                        if len(slot["patterns"]):
                            for _ in slot["patterns"]:
                                assert type(_) is str
                        assert type(slot["values"]) is list
                        if len(slot["values"]):
                            for _ in slot["values"]:
                                assert type(_) is str or type(_) is int or type(_) is float
                except Exception as e:
                    result = {"code": -1, "data": {}, "msg": f"intents error: {repr(e)}"}
                    return json.dumps(result, ensure_ascii=False)

            intents = data["intents"]
            if len(intents):
                # 校验格式
                try:
                    for intent in intents:
                        assert type(intent["name"]) is str
                        assert type(intent["patterns"]) is list
                        if len(intent["patterns"]):
                            for _ in intent["patterns"]:
                                assert type(_) is str
                        assert type(intent["texts"]) is list
                        if len(intent["texts"]):
                            for _ in intent["texts"]:
                                assert type(_) is str
                except Exception as e:
                    result = {"code": -1, "data": {}, "msg": f"intents error: {repr(e)}"}
                    return json.dumps(result, ensure_ascii=False)

            questions = data['questions']
            if len(questions):
                # 校验格式
                try:
                    for question in questions:
                        assert type(question["id"]) is int or type(question["id"]) is str
                        assert type(question["question"]) is str and question["question"] != ""
                        assert type(question["similarQuestions"]) is list
                        if len(question["similarQuestions"]):
                            for _ in question["similarQuestions"]:
                                assert type(_) is str and _ != ""
                    # 验证不能有 重复问题 id
                    assert len(questions) == len(set([question["id"] for question in questions]))
                except Exception as e:
                    result = {"code": -1, "data": {}, "msg": f"questions error: {repr(e)}"}
                    return json.dumps(result, ensure_ascii=False)

            # 数据为空的话，报异常
            if len(slots) + len(questions) + len(intents) == 0:
                result = {"code": -1, "data": {}, "msg": f"data is null"}
                return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            traceback.print_exc()
            result = {"code": -1, "data": {}, "msg": repr(e)}
            return json.dumps(result, ensure_ascii=False)

        _ = f"questions:{questions[:3]}\n, intents:{intents[:3]}\n, slots:{slots[:3]}\n"
        print(_), logging.info(_)
        if str(robot_id) != "" and str(version) != "":
            # 校验是否存在 robot id 和 version 冲突
            try:
                res = fun()
                if res:
                    result = res
                else:
                    result = {"code": -1, "data": {}, "msg": "服务错误！"}
            except Exception as e:
                traceback.print_exc()
                result = {"code": -1, "data": {}, "msg": repr(e)}
                return json.dumps(result, ensure_ascii=False)
        else:
            result = {"code": -6, "data": {}, "msg": "机器或版本为空异常！"}
        return json.dumps(result, ensure_ascii=False)

    else:
        result = {"code": -1, "data": {}, "msg": "请求异常！"}
        return json.dumps(result, ensure_ascii=False)


# 这个改动较大 @2
# @app.route('/api/algorithm/predict', methods=['GET', 'POST'])
# def algorithm_predict_old():
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
#     if request.method == "POST" or request.method == "GET":
#         try:
#             print("\n")
#             _ = "/api/algorithm/predict"
#             print(_), logging.info(_)
#             robot_id = int(data['robotId'])
#             version = int(data['version'])
#             if str(robot_id) != "" and str(version) != "":
#                 session_id = data['sessionId']
#                 session_id = "_".join([str(robot_id), str(version), str(session_id)])
#                 question = data["question"]
#                 slots = data["slots"]
#                 assert type(slots) is list
#                 for slot in slots:
#                     assert type(slot) is str
#
#                 if question != "":
#                     _ = f"机器编号：{robot_id}, 版本编号：{version}, 会话编号：{session_id}, 文本内容：{question}"
#                     print(_), logging.info(_)
#                     nlu_i = robot_manager.load_config_model(robot_id=robot_id, version=version)
#                     if nlu_i is None:
#                         result = {"code": -1, "msg": "机器或版本不合法！", "data": {}}
#                     else:
#                         response = nlu_i.parser(question, slots=slots)
#                         print(response)
#                         if response:
#                             result = {"code": 0, "msg": "ok", "data": response}
#                         else:
#                             result = {"code": 0, "msg": "ok", "data": {}}
#                 else:
#                     result = {"code": -8, "data": {}, "msg": "当前只支持文本输入哦！"}
#             else:
#                 result = {"code": -6, "data": {}, "msg": "机器或版本为空异常！"}
#             return json.dumps(result, ensure_ascii=False)
#         except Exception as e:
#             traceback.print_exc()
#             result = {"code": -1, "data": {}, "msg": repr(e)}
#             return json.dumps(result, ensure_ascii=False)
#
#     else:
#         result = {"code": -1, "data": {}, "msg": "request method should be GET/POST ！"}
#         return json.dumps(result, ensure_ascii=False)

@app.route('/api/algorithm/predict', methods=['GET', 'POST'])
def algorithm_predict():
    # try:
    print('进入predict app下的1')
    def form_or_json():
        if request.get_json(silent=True):
            return request.get_json(silent=True)
        else:
            if request.form:
                return request.form
            else:
                return request.args
    _='进入predict app下的'
    print(_), logging.info(_)
    data = form_or_json()
    if request.method == "POST" or request.method == "GET":
        try:
            print("\n")
            _ = "/api/algorithm/predict"
            print(_), logging.info(_)
            robot_id = int(data['robotId'])
            version = int(data['version'])
            if str(robot_id) != "" and str(version) != "":
                session_id = data['sessionId']
                session_id = "_".join([str(robot_id), str(version), str(session_id)])
                question = data["question"]
                slots = data["slots"]
                assert type(slots) is list
                for slot in slots:
                    assert type(slot) is str

                if question != "":
                    _ = '问题不为空'
                    print(_), logging.info(_)
                    _ = f"机器编号：{robot_id}, 版本编号：{version}, 会话编号：{session_id}, 文本内容：{question}"
                    print(_), logging.info(_)

                    nlu_i = robot_manager.load_config_model(robot_id=robot_id, version=version)
                    if nlu_i is None:
                        result = {"code": -1, "msg": "机器或版本不合法！", "data": {}}
                    else:
                        _ = '进入predict app下的NLU预测'
                        print(_), logging.info(_)
                        response = nlu_i.parser(question, slots=slots)
                        _ = '拿到predict app下的预测结果'
                        print(_), logging.info(_)
                        if response:
                            result = {"code": 0, "msg": "ok", "data": response}
                        else:
                            result = {"code": 0, "msg": "ok", "data": {}}
                else:
                    result = {"code": -8, "data": {}, "msg": "当前只支持文本输入哦！"}
            else:
                result = {"code": -6, "data": {}, "msg": "机器或版本为空异常！"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            traceback.print_exc()
            result = {"code": -9, "data": {}, "msg": repr(e)}
            return json.dumps(result, ensure_ascii=False)

    else:

        result = {"code": -3, "data": {}, "msg": "request method should be GET/POST ！"}
        return json.dumps(result, ensure_ascii=False)
# 这个不用改 @3
@app.route('/api/algorithm/query', methods=['GET', 'POST'])
def algorithm_query():
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
        # 还需新增状态
        _ = "【查询模型状态】机器编号：{}, 版本编号：{}".format(robot_id, version)
        print(_), logging.info(_)

        conn = pool.connection()  # 以后每次需要数据库连接就是用connection（）函数获取连接就好了
        cur = conn.cursor()
        """
        状态码：(由于设计，队列也更改为训练中)
        0 - 训练完成
        1 - 在队列中
        2 - 在训练中
        3 - 训练失败
        """
        # 首先查询是否存在 机器和改版本
        sql = f"select robot_id,version_id,status,STATUS_MESSAGE,created_at,start_time,end_time,INTENT_RESULT,SIMILARITY_RESULT,IS_ONLINE " \
              f"from {TABLE_NAME} " \
              f"where robot_id='{robot_id}' and version_id='{version}' and DELETE_FLAG=0 and CLUSTER='{CLUSTER}';"
        cur.execute(sql)
        res_ = cur.fetchall()
        if len(res_):
            row = res_[0]
            # config_ = get_data(robot_id, row[1])
            # print(row[7])
            intent_res = {}
            if str(row[7]) in ["", "NULL", "null", "None", "none"]:
                pass
            else:
                intent_res = json.loads(str(row[7]).strip())
            similar_res = {}
            if str(row[8]) in ["", "NULL", "null", "None", "none"]:
                pass
            else:
                similar_res = json.loads(str(row[8]).strip())
            # 去除 intent，
            acc_res = round((similar_res.get("dev", 0.0) + similar_res.get("dev", 0.0)) / 2, 2)
            # config_ = get_data_by_id(row[7])
            # config_ = json.loads(config_)
            # train_type = config_.get("train_type", "deep")
            if str(row[2]) == "0":
                sql = f"select robot_id,version_id,status,STATUS_MESSAGE,created_at,start_time,end_time " \
                      f"from {TABLE_NAME} " \
                      f"where robot_id='{robot_id}'and DELETE_FLAG=0 and CLUSTER='{CLUSTER}';"
                cur.execute(sql)
                res__ = cur.fetchall()
                versions = [int(_[1]) for _ in res__ if str(_[2]) == "0"]
                # 获取最新的运行的状态
                versions = sorted(versions, key=lambda x: x, reverse=True)
                print(versions), logging.info(versions)

                if int(row[1]) in versions[:2]:
                    result_ = {"code": 0, "msg": "ok！",
                               "data": {"status": "running", "statusMessage": "正在运作中...",
                                        "result": acc_res,
                                        "version": int(row[1]),
                                        "createTime": str(row[4]).strip("None"),
                                        "startTime": str(row[5]).strip("None"),
                                        "endTime": str(row[6]).strip("None"),
                                        "isOnline": bool(row[9])
                                        }}
                else:
                    result_ = {"code": 0, "msg": "ok！",
                               "data": {"status": "running", "statusMessage": "训练成功，但已删除（老版本）...",
                                        "result": acc_res,
                                        "version": int(row[1]),
                                        "createTime": str(row[4]).strip("None"),
                                        "startTime": str(row[5]).strip("None"),
                                        "endTime": str(row[6]).strip("None"),
                                        "isOnline": bool(row[9])
                                        }}
            elif str(row[2]) == "1":
                # 捕捉队列的信息
                sql = f"select robot_id,version_id,status " \
                      f"from {TABLE_NAME} " \
                      f"where status=1 and DELETE_FLAG=0 and CLUSTER='{CLUSTER}' order by CREATED_AT"
                cur.execute(sql)
                robot_status = cur.fetchall()
                count = 0
                for row_ in robot_status:
                    count += 1
                    if str(row_[0]) == str(robot_id) and str(row_[1]) == str(version):
                        break
                result_ = {"code": 0, "msg": "ok！",
                           "data": {"status": "queue", "statusMessage": f"前面还有{count}个机器人在排队中...",
                                    "result": acc_res,
                                    "version": int(row[1]),
                                    "createTime": str(row[4]).strip("None"),
                                    "startTime": str(row[5]).strip("None"),
                                    "endTime": str(row[6]).strip("None"),
                                    "isOnline": bool(row[9])
                                    }}
            elif str(row[2]) == "2":
                result_ = {"code": 0, "msg": "ok！", "data": {"status": "training", "statusMessage": "正在训练中",
                                                             "result": acc_res,
                                                             "version": int(row[1]),
                                                             "createTime": str(row[4]).strip("None"),
                                                             "startTime": str(row[5]).strip("None"),
                                                             "endTime": str(row[6]).strip("None"),
                                                             "isOnline": bool(row[9])
                                                             }}
            elif str(row[2]) == "3":
                result_ = {"code": 0, "msg": "ok！",
                           "data": {"status": "error", "statusMessage": str(row[3]), "version": int(row[1]),
                                    "result": acc_res,
                                    "createTime": str(row[4]).strip("None"),
                                    "startTime": str(row[5]).strip("None"),
                                    "endTime": str(row[6]).strip("None"),
                                    "isOnline": bool(row[9])}}
            else:
                result_ = {"code": -10, "data": {}, "msg": "状态码异常，不在范围内！"}
        else:
            result_ = {"code": -9, "data": {}, "msg": "机器和版本不存在!"}
            # {"code": 0, "msg": "ok！", "data": {"status": "error", "statusMessage": "机器不存在"}}
        cur.close()
        conn.close()
        return result_

    if request.method == "POST" or request.method == "GET":
        try:
            print("\n")
            _ = "/api_v2/chat/model_query"
            print(_), logging.info(_)
            robot_id = int(data['robotId'])
            version = int(data['version'])
            if str(robot_id) != "" and str(version) != "":
                # 校验是否存在 robot id 和 version 冲突
                res = fun()
                if res:
                    result = res
                else:
                    result = {"code": -1, "data": {}, "msg": "服务错误！"}
            else:
                result = {"code": -6, "data": {}, "msg": "机器或版本为空异常！"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            traceback.print_exc()
            result = {"code": -1, "data": {}, "msg": repr(e)}
            return json.dumps(result, ensure_ascii=False)

    else:
        result = {"code": -1, "data": {}, "msg": "请求异常！"}
        return json.dumps(result, ensure_ascii=False)


# 新增机器人删除接口 @4
@app.route('/api/algorithm/release', methods=['GET', 'POST'])
def algorithm_release():
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
        _ = f"机器编号：{robot_id}，版本编号：{version}"
        print(_), logging.info(_)
        conn = pool.connection()  # 以后每次需要数据库连接就是用connection（）函数获取连接就好了
        cur = conn.cursor()
        try:
            # step-1, 检测 机器 是否存在
            sql = f"SELECT robot_id,version_id,status " \
                  f"FROM {TABLE_NAME} " \
                  f"where robot_id='{robot_id}' and version_id='{version}' and CLUSTER='{CLUSTER}' and DELETE_FLAG=0;"
            cur.execute(sql)
            result_ = cur.fetchall()
            if len(result_):
                if result_[0][2] == "1":
                    return {"code": -12, "msg": "机器人还在排队中...", "data": {}}
                elif result_[0][2] == "2":
                    return {"code": -12, "msg": "机器人正在训练中...", "data": {}}
                elif result_[0][2] == "3":
                    return {"code": -12, "msg": "机器人该版本训练失败了_-_", "data": {}}
                # 修改数据库
                t1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 发布上线了
                try:
                    res_ = requests.post(f"{MODEL_MANAGER_HOST}/api/model_query",
                                         json={"robotId": robot_id, "version": version}).json()
                    if int(res_["data"]["intent_model"]) > 0:
                        # 查询模型存在，即跳过；
                        print("intent:", res_["data"]["intent_model"])
                        pass
                    else:
                        # 调用下接口，若是deep模式，会自动提起模型，或者fast模型，返回-1
                        _ = f"push model {robot_id}-{version} online.."
                        print(_), logging.info(_)
                        # payload_intent = {"modelName": "intent_model",
                        #                   "robotId": robot_id,
                        #                   "version": version,
                        #                   "text": "你好？", }
                        # payload_similar = {"modelName": "similar_model",
                        #                    "robotId": robot_id,
                        #                    "version": version,
                        #                    "text_as": ["今天天气", "今天天气"],
                        #                    "text_bs": ["杭州疫情", "今天天气不错"]}
                        payload_similar = {"modelName": "sentence_bert",
                                                       "robotId": robot_id,
                                                       "version": version,
                                                       "text_as": "好的"}
                        url = f"{MODEL_MANAGER_HOST}/api/model_predict"
                        # requests.post(url=url, json=payload_intent).json()
                        requests.post(url=url, json=payload_similar).json()
                        pass
                except Exception as e:
                    _ = repr(e)
                    print(_), logging.info(_)
                    return {"code": -12, "msg": "机器人该版本发布失败了_-_", "data": {}}

                # 当前版本置为1，其他版本置为0
                sql1 = f"UPDATE {TABLE_NAME} SET IS_ONLINE=1,UPDATED_AT=NOW() " \
                       f"where robot_id='{robot_id}' and version_id='{version}' " \
                       f"and CLUSTER='{CLUSTER}' and DELETE_FLAG=0;"

                sql2 = f"UPDATE {TABLE_NAME} SET IS_ONLINE=0,UPDATED_AT=NOW() " \
                       f"where robot_id='{robot_id}' and version_id !='{version}' " \
                       f"and CLUSTER='{CLUSTER}' and DELETE_FLAG=0;"
                index1 = cur.execute(sql1)
                index2 = cur.execute(sql2)
                print(sql1 + sql2), logging.info(sql1 + sql2)
                conn.commit()
                print(index1, index2), logging.info(index1 + index2)
                # 可以更新 到当前机器人所有的发布版本信息到redis，减少重复查询数据库操作
                sql = f"SELECT robot_id,version_id " \
                      f"FROM {TABLE_NAME} " \
                      f"where IS_ONLINE=1 and CLUSTER='{CLUSTER}' and DELETE_FLAG=0;"
                cur.execute(sql)
                result_ = cur.fetchall()
                if len(result_):
                    online_robot_versions = []
                    for row in result_:
                        online_robot_versions.append(row[0] + ":" + row[1])
                    _ = ";".join(online_robot_versions)
                    print(_), logging.info(_)
                    redis.set("online_robot_versions", ";".join(online_robot_versions))
                    print(redis.get("online_robot_versions"))
                return {"code": 0, "msg": "机器发布成功！", "data": {}}
            else:
                return {"code": -11, "data": {}, "msg": "机器不存在或已删除！"}
        except Exception as e:
            traceback.print_exc()
            _ = repr(e)
            print(_), logging.info(_)
            return {"code": -1, "data": {}, "msg": _}
        finally:
            cur.close()
            conn.close()

    if request.method == "POST" or request.method == "GET":
        try:
            print("\n")
            _ = "/api/algorithm/release"
            print(_), logging.info(_)
            robot_id = int(data['robotId'])
            version = int(data['version'])
            if str(robot_id) != "" and str(version) != "":
                # 校验是否存在 robot id 和 version 冲突
                res = fun()
                if res:
                    result = res
                else:
                    result = {"code": -1, "data": {}, "msg": "服务错误！"}
            else:
                result = {"code": -6, "data": {}, "msg": "机器ID空异常！"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            _ = repr(e)
            print(_), logging.info(_)
            result = {"code": -1, "data": {}, "msg": _}
            return json.dumps(result, ensure_ascii=False)
    else:
        result = {"code": -1, "data": {}, "msg": "request method should be GET/POST ！"}
        return json.dumps(result, ensure_ascii=False)


# 新增机器人删除接口 @5
@app.route('/api/algorithm/delete', methods=['GET', 'POST'])
def algorithm_delete():
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
        _ = f"机器编号：{robot_id}"
        print(_), logging.info(_)
        conn = pool.connection()  # 以后每次需要数据库连接就是用connection（）函数获取连接就好了
        cur = conn.cursor()
        try:
            # step-1, 检测 机器 是否存在
            sql = f"SELECT robot_id,version_id,status " \
                  f"FROM {TABLE_NAME} " \
                  f"where robot_id='{robot_id}' and CLUSTER='{CLUSTER}' and DELETE_FLAG=0;"
            cur.execute(sql)
            result_ = cur.fetchall()
            if len(result_):
                sql = f"SELECT robot_id,version_id,status " \
                      f"FROM {TABLE_NAME} " \
                      f"where robot_id='{robot_id}' and status=2 and DELETE_FLAG=0 and CLUSTER='{CLUSTER}';"
                cur.execute(sql)
                result_ = cur.fetchall()
                if len(result_):
                    return {"code": -12, "msg": "机器人正在训练中...", "data": {}}
                t1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sql = f"UPDATE {TABLE_NAME} SET DELETE_FLAG=1, IS_ONLINE=0, UPDATED_AT=NOW() " \
                      f"where robot_id='{robot_id}' and CLUSTER='{CLUSTER}' and DELETE_FLAG=0;"
                print(sql), logging.info(sql)
                index = cur.execute(sql)
                conn.commit()
                print(index), logging.info(index)
                # 模型管理哪里执行下 下线操作

                try:
                    res = requests.post(f"{MODEL_MANAGER_HOST}/api/delete_robot", json={"robotId": robot_id}).json()
                    _ = f"{robot_id} : {res}"
                    print(_), logging.info(_)
                except:
                    pass

                # 同时需要更新下redis
                online_robot_versions = redis.get("online_robot_versions")
                if online_robot_versions:
                    online_robot_versions = str(online_robot_versions.decode())
                    now_online_robot_versions = []
                    for rv in online_robot_versions.split(";"):
                        r = rv.split(":")[0]
                        if r == str(robot_id):
                            continue
                        else:
                            now_online_robot_versions.append(rv)
                    # 重置当前 上线版本
                    redis.set("online_robot_versions", ";".join(now_online_robot_versions))
                    _ = "now online robot_version:" + str(redis.get("online_robot_versions").decode())
                    print(_), logging.info(_)

                delete_robot = redis.get("delete_robot")
                if delete_robot:
                    delete_robot = str(delete_robot.decode())
                    delete_robot += ";" + str(robot_id)
                    redis.set("delete_robot", delete_robot)
                    redis.expire("delete_robot", 60)
                else:
                    delete_robot = str(robot_id)
                    redis.set("delete_robot", delete_robot)
                    redis.expire("delete_robot", 60)
                print(redis.get("delete_robot"))
                return {"code": 0, "msg": "机器删除成功！", "data": {}}
            else:
                delete_robot = redis.get("delete_robot")
                if delete_robot:
                    delete_robot = str(delete_robot.decode())
                    delete_robot += ";" + str(robot_id)
                    redis.set("delete_robot", delete_robot)
                    redis.expire("delete_robot", 60)
                else:
                    delete_robot = str(robot_id)
                    redis.set("delete_robot", delete_robot)
                    redis.expire("delete_robot", 60)
                print(redis.get("delete_robot"))
                return {"code": -11, "data": {}, "msg": "机器不存在或已删除！"}
        except Exception as e:
            traceback.print_exc()
            _ = repr(e)
            print(_), logging.info(_)
            return {"code": -1, "data": {}, "msg": _}
            pass
        finally:
            cur.close()
            conn.close()

    if request.method == "POST" or request.method == "GET":
        try:
            print("\n")
            _ = "/api_v2/chat/robot_model_delete"
            print(_), logging.info(_)
            robot_id = int(data['robotId'])
            if str(robot_id) != "":
                # 校验是否存在 robot id 和 version 冲突
                res = fun()
                if res:
                    result = res
                else:
                    result = {"code": -1, "data": {}, "msg": "服务错误！"}
            else:
                result = {"code": -6, "data": {}, "msg": "机器ID空异常！"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            _ = repr(e)
            print(_), logging.info(_)
            result = {"code": -1, "data": {}, "msg": _}
            return json.dumps(result, ensure_ascii=False)
    else:
        result = {"code": -1, "data": {}, "msg": "request method should be GET/POST ！"}
        return json.dumps(result, ensure_ascii=False)


# 这个不用改 @6
@app.route('/api/algorithm/slots', methods=['GET', 'POST'])
def algorithm_slots():
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

    # 不需要提交数据
    # print(json.loads(data))

    def fun():
        slot_ = {
            "code": 0,
            "msg": "ok！",
            "data": sn.get_system_ner_names()
        }
        return slot_

    if request.method == "POST" or request.method == "GET":
        try:
            print("\n")
            _ = "/api/algorithm/slots"
            print(_), logging.info(_)
            res = fun()
            if res:
                result = res
            else:
                result = {"code": -1, "data": {}, "msg": "服务错误！"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            traceback.print_exc()
            result = {"code": -1, "data": {}, "msg": repr(e)}
            return json.dumps(result, ensure_ascii=False)

    else:
        result = {"code": -1, "data": {}, "msg": "请求异常！"}
        return json.dumps(result, ensure_ascii=False)


# ----------------------------  v2  ----------------------------


# ----------------- 静态页面 -----------------
@app.route('/<page>.html', methods=['GET', 'POST'])
def html(page):
    # 直接返回静态文件
    # return render_template("submit.html")
    return app.send_static_file("examples/" + page + ".html")


# ----------------- 静态页面 -----------------


if __name__ == '__main__':
    # app.run(debug=True)
    # port = int(os.environ.get("PORT", "8002")) , threaded=True, processes=2
    # app.run(host='0.0.0.0', port="8008", threaded=False, processes=1)
    app.run(host='0.0.0.0', port="8008", threaded=True, processes=1, debug=True)
