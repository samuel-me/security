from flask import Flask,url_for, redirect,Response,render_template
import numpy as np 
#from wtforms import FileField,SubmitField
#from tensorflow.keras.models import load_model
import pandas as pd 
#import joblib
#from flask_wtf import FlaskForm
from matplotlib.image import imread
import datetime
import cv2
import time
import os,pickle
import face_recognition

app = Flask(__name__)
app.config["SECRET_KEY"] = "MEEEEEE"
#flower_scaler= joblib.load("iris_scd.pkl")
with open("a.pkl", 'rb') as b:
    encodings = pickle.load(b)


codes = list(encodings.values())

codes = list(encodings.values())

def predict(imgpath):
    try:
        img = encode(imgpath)

        code = face_recognition.api.compare_faces(codes,img,tolerance = 0.4)
        dist = face_recognition.face_distance(codes,img)
   
        m = dist.argmin()
        if code[m]:
            name= list(encodings.keys())[m]
            name = name.replace('_','')
            name = name.lower()
        else:
            name= 'Unknown'

        time = datetime.datetime.now()
    except:
        time = 'nil'
        name = 'nil'
    return name,time

def encode(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    code = face_recognition.face_encodings(rgb_img)[0]
    return code


#main_path = 'C:\\Users\\user 1\\Desktop\\Keras\\TF_2_Notebooks_and_Data\\SSLR _ FACE SECURITY\\'
#import cv2
cascade = cv2.CascadeClassifier("DATA/haarcascades/haarcascade_frontalface_default.xml")
def multiple(rects):
    rect = []
    for i in rects:
        rect.append([i])
    return rect

def MyRec(rgb,x,y,w,h,v=20,color=(200,0,0),thikness =1):
    """To draw stylish rectangle around the objects"""
    cv2.line(rgb, (x,y),(x+v,y), color, thikness)
    cv2.line(rgb, (x,y),(x,y+v), color, thikness)

    cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)

    cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
    cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)

    cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)

data_table = []



def check(table):

    #pass
    print("Name\tTime in\tTime out")
    #for i in range(len(table)):
        #print("hello")
        #print(table["Name"].iloc[i],"\t",table["Sign_in"].iloc[i],"\t",table["Sign_Out"].iloc[i])

 
import pandas as pd
def Make_Attendance():
    columns =["Name","Sign_in","Sign_Out","Day","Month","Year","seconds_spent","Minutes_spent","Hours_spent"]
    now = datetime.datetime.now()
    save_date = str(now.day)+"_"+str(now.month)+"_"+str(now.year)
    
    
    first = pd.DataFrame( data_table, columns = ["Label","Time"])

    second_data = []
    for name in first['Label'].unique():
        if name== 'nil' or name =='Unknown':
            continue


        lanre = first[first['Label'] == name]

        
        begin =lanre['Time'].iloc[0]
        #print(begin,name)
        end = lanre['Time'].iloc[len(lanre)-1]

        time2= end.strftime('%I:%M %p')
        time1= begin.strftime('%I:%M %p')
        day,month,day2,time,year = begin.ctime().split(' ')
        day  =day + " "+ day2
        total_time = end-begin

        sec = total_time.total_seconds()

        minutes = sec/60
        hours = minutes/60

        second_data.append([name,time1,time2,day,month,year,sec,minutes,hours])

    second = pd.DataFrame(second_data,columns=columns)
#     second.head()
    second.to_csv("Attendance"+save_date+".csv")
    return second


data_table = []
def detect_face(img,path):
   
    face_img = img.copy()
    
    gray = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)   
    
    rects = face_recognition.face_locations(gray)

    if len(rects) <1:
        label = 'nil'
        time = 'nil'

        pass

    elif len(rects) >1:
        rec = multiple(rects)
        for i in range(len(rec)):
            
            (y1,x2,y2,x1) = rec[i][0]
            #cv2.rectangle(face_img, (x1,y1), (x2,y2), (255,0,0), 3) 
            rol_color = img[y1:y2,x1:x2]


            size = cv2.resize(rol_color,(224,224))            
            label,time = predict(size)
            data_table.append([label,time])
            
        
            if label == 'Unknown' or label =='nil':
                
                cv2.rectangle(face_img, (x1,y1), (x2,y2), (0,0,255), 3)
        
                cv2.putText(face_img,'Unknown',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            #print(imread(mine)/255)
            else:
                cv2.rectangle(face_img, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.putText(face_img,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        
    elif len(rects) == 1:
        for (y1,x2,y2,x1) in rects:
            cv2.rectangle(face_img, (x1,y1), (x2,y2), (255,0,0), 10) 
        rol_color = img[y1:y2,x1:x2]

        size = cv2.resize(rol_color,(224,224))
        
        label,time = predict(size)
        data_table.append([label,time])              
                
        if label == 'Unknown' or label =='nil':
            cv2.rectangle(face_img, (x1,y1), (x2,y2), (255,0,0), 10)
        
            cv2.putText(face_img,'Unknown',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
           
        else:
            cv2.rectangle(face_img, (x1,y1), (x2,y2), (255,0,0), 10)
            cv2.putText(face_img,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    

    Make_Attendance()
    return (face_img)
    
    
    
    
def extract_video(video):
    # Same command function as streaming, its just now we pass in the file path, nice!
    cap = cv2.VideoCapture(0)
    if cap.isOpened()== False: 
        print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")

    while cap.isOpened():

        ret, frame = cap.read()

        # If we got frames, show them.
        if ret == True:

            frame = detect_face(frame,video)
            #
            #cv2.imshow('frame',frame)
            rec,uffer = cv2.imencode(".jpg",frame)
            frame = uffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


        #     if cv2.waitKey(25) & 0xFF == ord('q'):
            
        #         break

            
        # # Or automatically break this whole loop if the video is over.
        # else:
        #     break

            




@app.route('/video')
def video():
    
    file= ''
    print(f"extracting: {file}")
    return Response(extract_video(file), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def table():

    now = datetime.datetime.now()
    save_date = str(now.day)+"_"+str(now.month)+"_"+str(now.year)  
    data = pd.read_csv("Attendance"+save_date+ ".csv", index_col = 'Unnamed: 0')  
    return render_template("table.html",tables = [data.to_html()],titles = [""])

if __name__ == '__main__':
    app.run(debug= False, host = '0.0.0.0')
