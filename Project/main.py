import flask
from flask import Flask, render_template, Response,redirect,request, jsonify
from numpy import dtype
from camera import VideoCamera
from flask_pymongo import PyMongo
import os
from beautify.Moisturizer import *
from beautify.Facewash import *
from beautify.Sunscreen import *
import bson
from bson.objectid import ObjectId

from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

app = Flask(__name__)



app.config["MONGO_URI"] = "mongodb+srv://admindb:admindatabase@cluster0-vlwic.mongodb.net/myntra"
mongo = PyMongo(app)
db_operations = mongo.db.products

CART=[]

@app.route('/')
def indexx():
    return render_template('home.html')

@app.route('/tryon/<file_path>',methods = ['POST', 'GET'])
def tryon(file_path):
	file_path = file_path.replace(',','/')
	os.system('python tryOn.py ' + file_path)
	return redirect('http://127.0.0.1:5000/',code=302, Response=None)

@app.route('/tryall',methods = ['POST', 'GET'])
def tryall():
        CART = request.form['mydata'].replace(',', '/')
        os.system('python test.py ' + CART)
        render_template('checkout.html', message='')

##*************** EMOTION CODE *******

def emotionFun():
    # starting video streaming
    cv2.namedWindow('your_face')
    camera = cv2.VideoCapture(0)

    # parameters for loading data and images
    detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
    "neutral"]
    while True:
        frame = camera.read()[1]
        #reading the frame
        frame = imutils.resize(frame,width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
        else: continue

    
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    # draw the label + probability bar on the canvas
                # emoji_face = feelings_faces[np.argmax(preds)]

                    
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                (0, 0, 255), 2)


        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


@app.route('/emotion')
def emotion_idx():
    return render_template('emotion.html')

@app.route('/runemotion')
def emotion_fun():
    emotionFun()
    return redirect("http://127.0.0.1:5000/")

##*************** EMOTION CODE ENDS *******

@app.route('/read')
def read():
    users = db_operations.find()
    output = [{'Label' : user['Label'] } for user in users]
    return jsonify(output)

@app.route('/form', methods =["GET", "POST"])
def formData():
    if request.method == "POST":
        filter = {}
        filter['Label'] = request.form.get("label")
        skin = request.form.getlist("skin")
        for x in skin:
            if x == "Combination":
                filter['Combination'] = 1
            if x == "Dry":
                filter['Dry'] = 1
            if x == "Normal":
                filter['Normal'] = 1
            if x == "Oily":
                filter['Oily'] = 1
            if x == "Sensitive":
                filter['Sensitive'] = 1
        price_min = int(request.form['price-min'])
        price_max = int(request.form['price-max'])
        filter['price'] = { "$lte" : price_max, "$gte" : price_min}
        products = db_operations.find(filter)
        output = [
            {'Label' : product['Label'] ,
            'Img' : product['img'],
            'Brand' : product['brand'] ,
            'Name':product['name'],
            'Price':product['price'],
            'Rating':product['rating'],
            'Combination':product['Combination'],
            'Dry':product['Dry'],
            'Normal':product['Normal'],
            'Oily':product['Oily'],
            'Sensitive':product['Sensitive'],
            'Ingredients':product['ingredients'],
            '_id' : product['_id']
            }
            for product in products]
        return render_template("formoutput.html",output = output)
    return render_template('form.html')

@app.route("/details/<dbid>")
def insert_one(dbid):
    product=db_operations.find_one({'_id':bson.ObjectId(oid=str(dbid))})
    output = [
            {'Label' : product['Label'] ,
            'Img' : product['img'],
            'Brand' : product['brand'] ,
            'Name':product['name'],
            'Price':product['price'],
            'MRP':product['mrp'],
            'Rating':product['rating'],
            'Combination':product['Combination'],
            'Dry':product['Dry'],
            'Normal':product['Normal'],
            'Oily':product['Oily'],
            '_id':product['_id'],
            'Sensitive':product['Sensitive'],
            'Ingredients':product['ingredients']}]
    return render_template("details.html",output = output)

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/details/Face Moisturizer')
def applyMoisturizer_fun():
    applyMoisturizer()
    return redirect("http://127.0.0.1:5000/form")

@app.route('/details/Sunscreen')
def applySunscreen_fun():
    applySunscreen()
    return redirect("http://127.0.0.1:5000/form")

@app.route('/details/Face Wash And Cleanser')
def applyFacewash_fun():
    applyFacewash()
    return redirect("http://127.0.0.1:5000/form")

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/cart/<file_path>",methods = ['POST', 'GET'])
def cart(file_path):
    global CART
    file_path = file_path.replace(',','/')
    print("ADDED", file_path)
    CART.append(file_path)
    return render_template("checkout.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()