from flask import Flask, render_template, request
import base64
from scipy.misc import imsave, imread, imresize
import numpy as np
import pdb
import sys
import os
from load_DL_data import *
app = Flask(__name__)

global model, graph
model,graph = init()

@app.route('/draw_canvas')
def index():
   return render_template('canvas.html')

@app.route('/predict', methods=['POST', 'GET'])
def precict_data():
   print("under predict")
   enc_img = request.get_data()
   # pdb.Pdb(stdout=sys.__stdout__).set_trace()
   enc_img = enc_img[22:]
   with open('canvas_img.png', 'wb') as img :
         img.write(base64.b64decode(enc_img))

   # img_name = 'canvas_img.png'  # I assume you have a way of picking unique filenames
   dec_img = imread('canvas_img.png', mode='L')
   dec_img_swap = np.invert(dec_img)
   dec_img_swap = imresize(dec_img_swap, (28,28))
   dec_img_swap = dec_img_swap.reshape(1, 784)

   with graph.as_default():
       out = model.predict(dec_img_swap)
       response = np.array_str(np.argmax(out,axis=1))
       return response

if __name__ == '__main__':
   app.run(host='0.0.0.0',debug = True)