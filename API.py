from flask import Flask, jsonify, request, Response,render_template
import pickle
import requests
import json
import numpy as np
from numpy import savez_compressed,load
from werkzeug.utils import secure_filename
import backend
from backend import preprocess_image, extract_faces, add_face, remove_face, identify_face, identify_all, facedetect
import cv2
import os

def get_pickle():
    faces=pickle.load(open("faces.p","rb")) 
    return faces 
# Flask API Stuff
app = Flask(__name__)

#root
@app.route('/')
def index():
    return render_template("index.html")

#face uploader (manual)
@app.route('/faces')
def upload_face():
    return render_template('tester.html')

#face parser
@app.route('/faces/add_face',methods=["POST"])
def yeet_face():
    if request.method=="POST":
        label=request.json['label']
        facelist=np.array(request.json['image'])
        
    
#face parse tester
@app.route('/faces/import_face',methods=['POST'])
def import_face():
    global library
    label=request.form['label']
    f=request.files.getlist('face')
    facelist=[]
    for i in f:
        i.save(secure_filename(i.filename))
        img = preprocess_image(str(i.filename))
        face=extract_faces(img)
        facelist.append(face)
        print(facelist[0].shape)
        os.remove(i.filename)
    if len(library)==0:
        library = backend.add_person(facelist, label)
    else:
        library = backend.add_person(facelist, label, library)
    # return(label)
    savez_compressed('library.npz', library[0],library[1],library[2])
    print(library[3])
    with open("svm.pkl", 'wb') as file:
        pickle.dump(library[3], file) 


    return("File uploaded successfully!")
    

if __name__ == '__main__':
    #dealing with storage
    data = load("library.npz", allow_pickle=True)
    library=[data['arr_0'], data['arr_1'], data['arr_2']]
    with open("svm.pkl", 'rb') as file:
        svm = pickle.load(file)
    library.append(svm)
    app.run(debug=True)
    

#load pickle file
faces=pickle.load(open("faces.p","rb"))



