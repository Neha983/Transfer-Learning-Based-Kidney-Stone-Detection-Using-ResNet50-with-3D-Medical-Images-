from flask import Flask, request, jsonify, render_template


import os
import numpy as np # linear algebra
import pandas as pd 
import cv2
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.io import imread
from tensorflow.keras.models import load_model

# resnet50
from werkzeug.utils import secure_filename



app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("kidney_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    IMAGE_SIZE = [224, 224]
    img_path = request.files['myfile']
    file_name = secure_filename(img_path.filename)

    
    img_path.save(os.path.join(app.config['UPLOAD_FOLDER'],file_name))
    
    file_path = 'upload/'+file_name

    # Testing the a image with sample data

    pic=[]
    img = cv2.imread(str(file_path))
    img = cv2.resize(img, (28,28))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=np.array(img)
    img = img/255
    #label = to_categorical(0, num_classes=2)
    pic.append(img)

    #pic_labels.append(pneu)
    pic1 = np.array(pic)
    a=model.predict(pic1)
    ar=a.argmax()
    labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
    print("-----------------------------")
    print(ar)
    print('--------------------------------')
    s=labels[ar]
    os.remove(file_path)
    return render_template('index.html', prediction_text='Kidney Stone Prediction Results: {}'.format(s))

if __name__ == "__main__":
    app.run(debug=True)