'''
Author: Kaizyn
Date: 2023-04-24 16:56:13
LastEditTime: 2023-04-30 16:56:02
'''
import os

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='./', static_folder='/', static_url_path='/')
# app = Flask(__name__, template_folder='./')

FILE_PATH = os.path.abspath(__file__)
PROJ_PATH = os.path.dirname(FILE_PATH)
D = 128
user_embedding = np.zeros(D)
outfits = []
recommend_outfits = []

def load_data():
    pass

def recommend():
    '''
    similarity = []
    for i, outfit in enumerate(outfits):
        similarity.append()
    outfits = [x for _, x in sorted(zip(similarity, outfits))]
    return recommend_outfits
    '''
    return \
    [
        [
            "images/thumbs/clothes.jpg",
            "images/thumbs/pants.jpg",
            "images/thumbs/shoes.jpg",
        ],
        [
            "images/thumbs/clothes.jpg",
            "images/thumbs/01.jpg",
            "images/thumbs/02.jpg",
        ],
    ]

@app.route('/')
def index():
    # 在此处生成两个时尚套装的图片地址，并将它们传递到前端渲染
    recommend_outfits = recommend()
    return render_template('index.html', outfits=recommend_outfits)

@app.route('/', methods=['POST'])
def handle_vote():
    # 处理用户投票信息的代码
    action = request.form['action']
    outfit = request.form['outfit']
    # print(f'action: {action}, outfit: {outfit}')
    # print(request.form)
    if action == 'yes':
        # 用户喜欢这个套装
        print('yes')
    elif action == 'no':
        # 用户不喜欢这个套装
        print('no')

    return 'OK'

if __name__ == '__main__':
    load_data()
    app.run(debug=True)
