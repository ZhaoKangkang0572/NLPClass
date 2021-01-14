#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2020-10-23 16:53
"""
import os
import json
import time
import pickle
import traceback
from datetime import datetime
import threading
import shutil
import requests
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from concurrent.futures import ThreadPoolExecutor

from robot_config.config import Config
from utils.other import time_spend
from utils.redis_pool import master as redis
from utils.mysql_pool import pool, CLUSTER, TABLE_NAME
from utils.mongo_util import collection
from utils.sftp_utils import SftpServer, HOST, USER, PASSWORD, PORT, ORIGIN_PATH
# from utils.es_util import get_data_by_id
from bson import ObjectId

from sys_config import MODEL_HOST, USE_TF_SERVING, USE_MODEL_MANAGER, MODEL_MANAGER_HOST, PYTHON_BIN_PATH

__PATH__ = os.getcwd()

if not os.path.exists("./log"):
    print("not exits..."), logging.info("not exits...")
    os.makedirs("./log")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='log/job_train.log',
                    filemode='w')

executor = ThreadPoolExecutor(1)


def do_thread(id, command):
    os.system(command)


@time_spend
def long_task_train(config):
    conn = pool.connection()  # 以后每次需要数据库连接就是用connection（）函数获取连接就好了
    cur = conn.cursor()
    try:
        # step 1 更新数据库状态
        robot_id = config["robotId"]
        version_id = config["version"]
        # t1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = f'UPDATE {TABLE_NAME} SET status=2,UPDATED_AT=NOW(),START_TIME=NOW() ' \
              f'WHERE robot_id="{robot_id}" and version_id="{version_id}" and DELETE_FLAG=0 and CLUSTER="{CLUSTER}";'
        print(sql), logging.info(sql)
        index = cur.execute(sql)
        conn.commit()
        print(index), logging.info(index)

        def do_update_mysql():
            # t1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sql_ = f'UPDATE {TABLE_NAME} SET TRAINING_TIME=NOW(),UPDATED_AT=NOW() ' \
                   f'WHERE robot_id="{robot_id}" and version_id="{version_id}" and DELETE_FLAG=0 and CLUSTER="{CLUSTER}";'
            # print(sql), logging.info(sql)
            index = cur.execute(sql_)
            conn.commit()



        # step 3 训练模型(训练 意图)
        # _ = "start train intent ..."
        # print(_), logging.info(_)
        # command_0 = f'CUDA_VISIBLE_DEVICES=0 {PYTHON_BIN_PATH} -m train_intent -r {robot_id} -v {version_id}'
        # thread_0 = threading.Thread(target=do_thread, args=('1', command_0))
        # thread_0.start()
        # while True:
        #     if not thread_0.is_alive():
        #         break
        #     do_update_mysql()
        #     time.sleep(20)

        @time_spend
        def do_robot_config1():
            # 预处理机器人配置文件，方便加载近内存
            try:
                print("开始第一次载入pickle")
                c = Config(config)
                print("开始第一次dumppickle")
                c.dumps()
            except:
                traceback.print_exc()

        _ = "start dumps robot config ..."
        print(_), logging.info(_)
        thread_ = threading.Thread(target=do_robot_config1, )
        thread_.start()
        while True:
            if not thread_.is_alive():
                break
            do_update_mysql()
            time.sleep(10)

        # step 4 训练模型(训练 相似度)
        _ = "start train similarity ..."
        print(_), logging.info(_)
        # os.system("source activate baili")
        # command_1 = f'CUDA_VISIBLE_DEVICES=0 {PYTHON_BIN_PATH} -m train_similarity -r {robot_id} -v {version_id}'
        # # command_1 = f'CUDA_VISIBLE_DEVICES=0 conda activate baili && {PYTHON_BIN_PATH} -m train_similarity -r {robot_id} -v {version_id}'
        # thread_1 = threading.Thread(target=do_thread, args=('1', command_1))
        # thread_1.start()
        # while True:
        #     if not thread_1.is_alive():
        #         break
        #     do_update_mysql()
        #     time.sleep(20)

        _ = "start train sentence bert ..."
        print(_), logging.info(_)
        command_2 = f'CUDA_VISIBLE_DEVICES=0 {PYTHON_BIN_PATH} -m train_similarity_sentenceBERT -r {robot_id} -v {version_id}'
        print(command_2)
        thread_2 = threading.Thread(target=do_thread, args=('1', command_2))
        thread_2.start()
        while True:
            if not thread_2.is_alive():
                break
            do_update_mysql()
            time.sleep(20)

        # step 2 执行任务(训练配置预处理)
        @time_spend
        def do_robot_config2():
            # 预处理机器人配置文件，方便加载近内存
            c = Config(config)
            c.process_sentence_bert()
            c.dumps()

        _ = "start dumps robot config ..."
        print(_), logging.info(_)
        thread_ = threading.Thread(target=do_robot_config2, )
        thread_.start()
        while True:
            if not thread_.is_alive():
                break
            do_update_mysql()
            time.sleep(10)






        # 删除老版本的缓存
        redis.delete(f"robot_{robot_id}_version_{version_id}_v2")

        # step 5 上传配置与模型 到sftp服务器
        _ = "start upload sftp ..."
        print(_), logging.info(_)

        @time_spend
        def upload_to_sftp():
            # 由于是隔离的，需要添加下 ftp 服务，上传、下载
            sftp = SftpServer(HOST, USER, PASSWORD, PORT)
            sftp.put_file(f'config_models/robot_{robot_id}_version_{version_id}.model',
                          f"./{ORIGIN_PATH}/config_models/robot_{robot_id}_version_{version_id}.model")

            # 上传训练好的模型文件

            if os.path.exists(f'config_models/robot_{robot_id}_version_{version_id}_intent'):
                _ = "上传意图模型"
                print(_), logging.info(_)
                os.chdir(__PATH__)
                sftp.put_dir(f'config_models/robot_{robot_id}_version_{version_id}_intent',
                             f"./{ORIGIN_PATH}/config_models/robot_{robot_id}_version_{version_id}_intent")
                _ = "上传模型配置完成, 上传相似度模型"
                print(_), logging.info(_)
            if os.path.exists(f'config_models/robot_{robot_id}_version_{version_id}_similarity'):
                os.chdir(__PATH__)
                sftp.put_dir(f'config_models/robot_{robot_id}_version_{version_id}_similarity',
                             f"./{ORIGIN_PATH}/config_models/robot_{robot_id}_version_{version_id}_similarity")
                _ = "上传相似度模型完成"
                print(_), logging.info(_)
                os.chdir(__PATH__)

            if os.path.exists(f'config_models/robot_{robot_id}_version_{version_id}_similarity_sentbert'):
                os.chdir(__PATH__)
                sftp.put_dir(f'config_models/robot_{robot_id}_version_{version_id}_similarity_sentbert',
                             f"./{ORIGIN_PATH}/config_models/robot_{robot_id}_version_{version_id}_similarity_sentbert")
                _ = "上传senenceBert相似度模型完成"
                print(_), logging.info(_)
                os.chdir(__PATH__)
            sftp.close()

        thread_3 = threading.Thread(target=upload_to_sftp, )
        thread_3.start()
        while True:
            if not thread_3.is_alive():
                break
            do_update_mysql()
            time.sleep(20)

        # step 3 更新数据库状态 （上传之后，避免数据库状态 更新是， 模型已上传完成）
        robot_id = config["robotId"]
        version_id = config["version"]
        t1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = f'UPDATE {TABLE_NAME} SET status=0,UPDATED_AT=NOW(),END_TIME=NOW() ' \
              f'WHERE robot_id="{robot_id}" and version_id="{version_id}" and DELETE_FLAG=0 and CLUSTER="{CLUSTER}";'
        print(sql), logging.info(sql)
        index = cur.execute(sql)
        conn.commit()
        print(index), logging.info(index)
        _ = "训练完成，更新到数据库"
        print(_), logging.info(_)

        # 首先查询 当前机器的 训练好的版本情况
        sql = f"SELECT robot_id,version_id,status FROM {TABLE_NAME} " \
              f"where status=0 and DELETE_FLAG=0 and CLUSTER='{CLUSTER}';"
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

        _ = robot_versions
        print(_), logging.info(_)
        for robot_id, versions in robot_versions.items():
            versions = list(set(versions))
            versions = sorted(versions, key=lambda x: int(x), reverse=True)
            # print(versions)
            top2versions = versions[:2]
            for version_id in versions:
                if version_id in top2versions:
                    pass
                else:
                    if os.path.exists(f"./config_models/robot_{robot_id}_version_{version_id}.model"):
                        _ = f"remove old version, robot: {robot_id}, version: {version_id}"
                        print(_), logging.info(_)
                        try:
                            shutil.rmtree(f"./config_models/robot_{robot_id}_version_{version_id}_intent")
                        except:
                            pass
                        try:
                            shutil.rmtree(f"./config_models/robot_{robot_id}_version_{version_id}_similarity")
                        except:
                            pass
                        try:
                            shutil.rmtree(f"./config_models/robot_{robot_id}_version_{version_id}_similarity_sentbert")
                        except:
                            pass
                        try:
                            os.remove(f"./config_models/robot_{robot_id}_version_{version_id}.model")
                        except:
                            pass
                    else:
                        pass
        cur.close()
        conn.close()

    except Exception as e:
        # 发生异常，更新数据库状态，置为 3
        traceback.print_exc()
        _ = "error: {}".format(repr(e))
        print(_), logging.info(_)
        robot_id = config["robotId"]
        version_id = config["version"]
        t1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = f'UPDATE {TABLE_NAME} SET status=3,STATUS_MESSAGE="{_}",UPDATED_AT=NOW(),END_TIME=NOW() ' \
              f'WHERE robot_id="{robot_id}" and version_id="{version_id}" and DELETE_FLAG=0 and CLUSTER="{CLUSTER}";'
        print(sql), logging.info(sql)
        index = cur.execute(sql)
        conn.commit()
        print(index), logging.info(index)
        traceback.print_exc()
        cur.close()
        conn.close()


def interval_sql_train():
    """
    由于频繁访问数据库，最好还是用一下数据库连接池！
    状态码：

    0 - 训练完成
    1 - 在队列中
    2 - 在训练中
    3 - 训练失败
    """
    delete_robot = redis.get("delete_robot")
    if delete_robot:
        delete_robot = str(delete_robot.decode())
        robots = delete_robot.split(";")
        for robot in robots:
            # 执行下线操作
            try:
                res = requests.post(f"{MODEL_MANAGER_HOST}/api/delete_robot", json={"robotId": robot}).json()
                _ = f"{robot} : {res}"
                print(_), logging.info(_)
            except:
                pass
            # 下面命令会删除 模型文件
            command = f"rm -r ./config_models/robot_{robot}_version*"
            os.system(command)
            command = f"rm ./config_models/robot_{robot}_version*"
            os.system(command)

        # 不删除，因为可能多点，其他节点也需要操作，可能进行重复，不影响
        # redis.delete("delete_robot")
    else:
        pass
    conn = pool.connection()  # 以后每次需要数据库连接就是用connection（）函数获取连接就好了
    cur = conn.cursor()
    # 首先查询是否存在训练中的任务
    sql = f"select robot_id,version_id,TRAINING_TIME from {TABLE_NAME} " \
          f"where status=2 and DELETE_FLAG=0 and CLUSTER='{CLUSTER}';"
    cur.execute(sql)
    result = cur.fetchall()
    if len(result):
        row = result[0]
        _ = "robot:{}, version:{} is in training....".format(row[0], row[1])
        print(_), logging.info(_)
        # 再加个判断失败的功能，一直监听训练是否无故卡死等状态
        # datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")

        sql = f'UPDATE {TABLE_NAME} SET status=3,UPDATED_AT=NOW() ' \
              f'WHERE TRAINING_TIME<SUBDATE(now(),interval 3 minute) and status=2 ' \
              f'and DELETE_FLAG=0 and CLUSTER="{CLUSTER}";'
        print(sql), logging.info(sql)
        index = cur.execute(sql)
        conn.commit()
        print(index), logging.info(index)

    else:
        sql = f"select robot_id,version_id,es_id,status from {TABLE_NAME} " \
              f"where status=1 and DELETE_FLAG=0 and CLUSTER='{CLUSTER}' order by CREATED_AT"
        cur.execute(sql)
        result = cur.fetchall()
        if len(result):
            row = result[0]
            _ = "robot:{}, version:{} is starting....".format(row[0], row[1])
            print(_), logging.info(_)
            try:
                # 2020-12-05 改成mongodb进行读取
                # 采用 es 获取 训练数据
                mongo_result = collection.find({"_id": ObjectId(row[2])})
                print(row[2])
                # res = get_data_by_id(row[2])
                # config = json.loads(res)
                config = mongo_result[0]
                # 2020-12-05 改成mongodb进行读取     1	1	1	Gg08WXUB-deteIE8h-bN	192.168.1.245:19200/algorithm_train_data_xs/_doc/Gg08WXUB-deteIE8h-bN	1	1	{"dev": 0.9743589743589745, "train": 0.9935064935064936}	{"dev": 1, "train": 1}		2020-11-03 16:20:01	2020-11-03 16:16:30	2020-11-03 16:20:21	实在科技	2020-10-24 14:12:18	实在科技	2020-11-13 15:00:30	0	1
                # 后台执行任务ObjectId("5fcaf28b8f762624170ca9a3")
                executor.submit(long_task_train, config)
            except Exception as e:
                _ = "error: {}".format(repr(e))
                print(_), logging.info(_)
                traceback.print_exc()
                robot_id = row[0]
                version_id = row[1]
                # t1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sql = f'UPDATE {TABLE_NAME} SET status=3,STATUS_MESSAGE="{_}",UPDATED_AT=NOW(),END_TIME=NOW() ' \
                      f'WHERE robot_id="{robot_id}" and version_id="{version_id}" ' \
                      f'and DELETE_FLAG=0 and CLUSTER="{CLUSTER}";'
                print(sql), logging.info(sql)
                index = cur.execute(sql)
                conn.commit()
        else:
            _ = "no task..."
            print(_), logging.info(_)
    cur.close()
    conn.close()


def my_listener(event):
    if event.exception:
        _ = '任务出错了！！！！！！'
        print(_), logging.info(_)
    else:
        # 任务正常，就不打印日志了
        # traceback.print_exc()
        pass


if __name__ == "__main__":

    # interval_sql_train()

    scheduler = BlockingScheduler()
    scheduler.add_job(func=interval_sql_train, trigger='interval', seconds=10,
                      id='interval_sql_train')  # args=('循环任务',)
    scheduler.add_listener(my_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    scheduler._logger = logging
    scheduler.start()
