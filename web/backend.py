'''
Author: Kaizyn
Date: 2023-04-24 16:56:13
LastEditTime: 2023-05-07 13:20:21
'''
import os
import time
import random
import logging

import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='./', static_folder='/', static_url_path='/')
# app = Flask(__name__, template_folder='./')
LOGGER = logging.getLogger(__name__)
# LOGGER.setLevel(logging.DEBUG)
LOGGER.setLevel(logging.INFO)

FILE_PATH = os.path.abspath(__file__)
WEB_PATH = os.path.dirname(FILE_PATH)
PROJ_PATH = os.path.dirname(WEB_PATH)

def TopKSort(count, start, end, res, k):
    randIndex = random.randint(start, end - 1)  # 随机挑一个下标作为中间值开始找
    count[start], count[randIndex] = count[randIndex], count[start] # 先把这个随机找到的中间元素放到开头
    
    midVal = count[start][1] # 选中的中间值
    index = start + 1
    for i in range(start + 1, end):
        if count[i][1] > midVal: # 把所有大于中间值的放到左边
            count[index], count[i] = count[i], count[index]
            index += 1
    count[start], count[index - 1] = count[index - 1], count[start] # 中间元素归位

    if k < index - start: # 随机找到的top大元素比k个多，继续从前面top大里面找k个
        TopKSort(count, start, index, res, k)
    elif k > index - start: # 随机找到的比k个少
        # for i in range(start, index): # 先把top大元素的key存入结果
            # res.append(count[i][0])
        TopKSort(count, index, end, res, k - (index - start)) # 继续往后找
    else: # 随机找到的等于k个
        # for i in range(start, index): # 把topk元素的key存入结果
            # res.append(count[i][0])
        return


class Recommender():
    K = 10
    D = 128
    U = 630
    epsilon = 0.5
    beta = 0.5

    def __init__(self):
        LOGGER.debug('Recommender.__init__()')
        self.outfits_posi_feat: list = None # U * D
        self.outfits_nega_feat: list = None # U * D
        self.user_codes: list        = None # U * D
        self.outfits: list           = None # N * [[3 * file_name], feat(D)]
        self.lambda_u                = None # D
        self.lambda_i                = None # D
        self.clicks_posi_feat: list  = []   # * D
        self.clicks_nega_feat: list  = []   # * D
        self.click_user_code         = np.random.rand(self.D)
        self.load_data()
        self.N = len(self.outfits)

    def load_data(self):
        time_start = time.time()
        LOGGER.info('load_data(): start')
        features_path = os.path.join(PROJ_PATH, 'results', 'features.npz')
        outfits_path = os.path.join(PROJ_PATH, 'results', 'outfits.npz')
        with open(features_path, "rb") as f:
            features_data = pickle.load(f)
        with open(outfits_path, "rb") as f:
            outfits_data = pickle.load(f)
        self.outfits_posi_feat = features_data.get('big_outfits_posi_feat', None)
        self.outfits_nega_feat = features_data.get('big_outfits_nega_feat', None)
        self.user_codes        = outfits_data.get('user_codes', None)
        self.outfits           = outfits_data.get('outfits', None)
        self.lambda_u          = outfits_data.get('lambda_u', None)
        self.lambda_i          = outfits_data.get('lambda_i', None)

        self.outfits_posi_feat = [outfit.numpy() for outfit in self.outfits_posi_feat]
        self.outfits_nega_feat = [outfit.numpy() for outfit in self.outfits_nega_feat]
        for i in range(len(self.outfits)):
            self.outfits[i][0] = [f'images/291x291/{file}' for file in self.outfits[i][0]]
        LOGGER.debug(f'outfits_posi_feat: {type(self.outfits_posi_feat[0])} {self.outfits_posi_feat[0]}')
        LOGGER.debug(f'outfits_nega_feat: {type(self.outfits_nega_feat[0])} {self.outfits_nega_feat[0]}')
        LOGGER.debug(f'user_codes: {type(self.user_codes[0])} {self.user_codes[0]}')
        LOGGER.debug(f'outfits: {type(self.outfits[0])} {self.outfits[0]}')
        LOGGER.debug(f'lambda_u: {type(self.lambda_u)} {self.lambda_u}')
        LOGGER.debug(f'lambda_i: {type(self.lambda_i)} {self.lambda_i}')
        LOGGER.info(f'load_data(): end, cost time: {time.time() - time_start} s.')
    
    def user_user_embedding(self, big_outfits, small_outfit):
        big_num_users = len(big_outfits)
        user_code = np.zeros(self.D)
        for u in range(big_num_users):
            sim = (small_outfit * self.lambda_i * big_outfits[u]).sum() / self.D
            user_code += sim * self.user_codes[u]
        user_code /= self.U
        return user_code

    def user_item_embedding(self, outfit):
        return outfit * self.lambda_u

    def get_user_embedding(self):
        time_points = [time.time()]
        # click_posi_feat = np.random.rand(self.D)
        # click_nega_feat = np.random.rand(self.D)
        click_posi_feat = np.ones(self.D)
        click_nega_feat = np.ones(self.D)
        if len(self.clicks_posi_feat) > 0:
            click_posi_feat = sum(self.clicks_posi_feat) / len(self.clicks_posi_feat)
        if len(self.clicks_nega_feat) > 0:
            click_nega_feat = sum(self.clicks_nega_feat) / len(self.clicks_nega_feat)
        time_points.append(time.time())
        LOGGER.info(f'user_embedding() click embedding time cost: {time_points[-1] - time_points[-2]}s.')
        code_user_posi  = self.user_user_embedding(self.outfits_posi_feat, click_posi_feat)
        code_user_nega1 = self.user_user_embedding(self.outfits_posi_feat, click_nega_feat)
        code_user_nega2 = self.user_user_embedding(self.outfits_nega_feat, click_posi_feat)
        code_user = (code_user_posi - self.epsilon * (code_user_nega1 + code_user_nega2)) / (1 + 2 * self.epsilon)
        time_points.append(time.time())
        LOGGER.info(f'user_embedding() user-user embedding time cost: {time_points[-1] - time_points[-2]}s.')
        code_item_posi = self.user_item_embedding(click_posi_feat)
        code_item_nega = self.user_item_embedding(click_nega_feat)
        code_item = (code_item_posi - code_item_nega) / 2
        time_points.append(time.time())
        LOGGER.info(f'user_embedding() user-item embedding time cost: {time_points[-1] - time_points[-2]}s.')
        user_code = np.sign(self.beta * code_user + (1 - self.beta) * code_item)
        time_points.append(time.time())
        LOGGER.info(f'user_embedding() total time cost: {time_points[-1] - time_points[0]}s.')
        return user_code
    
    def click(self, action: str, outfit: int):
        if action == 'yes':
            self.clicks_posi_feat.append(self.outfits[outfit][-1])
        elif action == 'no':
            self.clicks_nega_feat.append(self.outfits[outfit][-1])
        else:
            pass
        self.click_user_code = self.get_user_embedding()
        LOGGER.info(f'click({action}, {outfit}): user_code = {self.click_user_code}')
    
    def recommend(self):
        time_points = [time.time()]
        ratings = [(self.click_user_code * self.lambda_u * outfit[-1]).sum() for outfit in self.outfits]
        time_points.append(time.time())
        LOGGER.info(f'recommend() calculate ratings time cost: {time_points[-1] - time_points[-2]}s.')
        '''
        recommend_id = []
        rating_id = list(zip(ratings, range(self.N)))
        TopKSort(rating_id, 0, self.N, recommend_id, self.K * 10)  # 迭代函数求前k个
        time_points.append(time.time())
        LOGGER.info(f'recommend() rank ratings time cost: {time_points[-1] - time_points[-2]}s.')
        recommend_id = [i for r, i in rating_id[:self.K * 10]]
        recommend_id = random.sample(recommend_id, self.K)
        '''
        rank_outfits = [x for _, x in sorted(zip(ratings, range(len(ratings))), reverse=True)]
        time_points.append(time.time())
        LOGGER.info(f'recommend() rank ratings time cost: {time_points[-1] - time_points[-2]}s.')
        recommend_id = [rank_outfits[id] for id in random.sample(range(100), self.K)]
        recommend_outfits = [[*self.outfits[id][0], id] for id in recommend_id]
        time_points.append(time.time())
        LOGGER.info(f'recommend() total time cost: {time_points[-1] - time_points[0]}s.')
        LOGGER.debug(f'recommend(): recommend_id = {recommend_id}')
        return recommend_outfits


RECOMMENDER: Recommender = None

@app.route('/')
def index():
    global RECOMMENDER
    # 在此处生成两个时尚套装的图片地址，并将它们传递到前端渲染
    recommend_outfits = RECOMMENDER.recommend()
    return render_template('index.html', outfits=recommend_outfits)

@app.route('/', methods=['POST'])
def handle_vote():
    global RECOMMENDER
    # 处理用户投票信息的代码
    action = request.form['action']
    outfit = int(request.form['outfit'])
    RECOMMENDER.click(action, outfit)

    return 'OK'

if __name__ == '__main__':
    # global RECOMMENDER
    logging.debug('__main__')
    RECOMMENDER = Recommender()
    app.run(debug=True)
