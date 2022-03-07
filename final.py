#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import os
from keras.preprocessing.image import load_img
from keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt
from flask import Flask , request , render_template
from PIL import Image
from io import BytesIO  
from fpdf import FPDF
import pymongo

Client = pymongo.MongoClient()

mydb = Client['SKIN_DISEASE']

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('logo.png', 10, 8, 33)
        
        # Arial bold 15
        self.set_font('Arial', 'B', 20)
        # Move to the right
        self.cell(70,80,'',0,0,0)
        # Title
        self.cell(80, 10, 'Skin Disease Classification', 0, ln=0, align='C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')





app = Flask(__name__)


modeld = load_model("model.h5")

@app.route('/')
def home():
    return render_template('check.html')




@app.route('/gettypeofdisease',methods=['POST'])
def gettypeofdisease():
    msge = ''
    name = request.form['name']
    age = request.form['age']
    phoneNumber = request.form['phno']
    Email = request.form['E-mail']
    file_pic = request.files['file']
    mycol = mydb['User']
    path = f"static/temp/image.jpg"
    i=Image.open(file_pic)
    newsize = (200, 200)
    i= i.resize(newsize)
    i.save(path,'png')
    url=f"static/userinfo"
    #image_data = np.array(img) 
    path1 = "C:\\project\\static\\temp\\"
    img = cv2.imread(path1+'image.jpg',)
    print(img)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    filterSize =(5, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filterSize)


    Masked_image = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT,kernel)

   
    cleared_image = cv2.inpaint(img,Masked_image,3,cv2.INPAINT_NS)

#%%

    plt.imshow(cleared_image)
    #cv2.imwrite(os.path.join(path , 'Cleared_image.jpg'), cleared_image)
   
#
    #img = image.resize((200, 200))
    # plt.imshow(img)
    # plt.show()
    X = image.img_to_array(cleared_image)
    X = np.expand_dims(X,axis=0)
    #print(images)
    val = modeld.predict(X)

     # save FPDF() class into
# a variable pdf
    if val == 1:
      msge = 'pigmented benign keratosis'
      pdf = PDF()
      pdf.alias_nb_pages()
      pdf.add_page()
      pdf.line(10, 50, 200, 50)
      pdf.set_font('Times', 'i', 20)
      pdf.cell(0,30,'',0,1,0)
      pdf.cell(41,0,'',0,0,0)
      pdf.cell(0,15,'Name:'+name,0,1,0)
      pdf.cell(46,0,'',0,0,0)
      pdf.cell(0,15,'Age:'+age,0,1,1)
      pdf.cell(39,0,'',0,0,0)
      pdf.cell(0,15,'E-mail:'+Email,0,1,1)
      pdf.cell(17,0,'',0,0,0)
      pdf.cell(0,15,'PhoneNumber:'+phoneNumber,0,1,0)
      pdf.cell(19,0,'',0,0,0)
      pdf.cell(0,15,'DiseaseName:'+msge,0,1,0)
      pdf.output('C:\project\\static\\userinfo\\%s.pdf'%phoneNumber, 'F')
   
      mydict = { "name":name, "age":age, "phoneNumber":phoneNumber, "E-mail":Email,"Disease_name":msge }
      x = mycol.insert_one(mydict)
      print(x) 
    else:
      msge = 'Vascular lesian'
      pdf = PDF()
      pdf.alias_nb_pages()
      pdf.add_page()
      pdf.line(10, 50, 200, 50)
      pdf.set_font('Times', 'i', 20)
      pdf.cell(0,30,'',0,1,0)
      pdf.cell(41,0,'',0,0,0)
      pdf.cell(0,15,'Name:'+name,0,1,0)
      pdf.cell(46,0,'',0,0,0)
      pdf.cell(0,15,'Age:'+age,0,1,1)
      pdf.cell(39,0,'',0,0,0)
      pdf.cell(0,15,'E-mail:'+Email,0,1,1)
      pdf.cell(17,0,'',0,0,0)
      pdf.cell(0,15,'PhoneNumber:'+phoneNumber,0,1,0)
      pdf.cell(19,0,'',0,0,0)
      pdf.cell(0,15,'DiseaseName:'+msge,0,1,0)
      pdf.output('C:\project\\static\\userinfo\\%s.pdf'%phoneNumber, 'F')
      mydict = { "name":name, "age":age, "phoneNumber":phoneNumber, "E-mail":Email, "Disease_name":msge }
      x = mycol.insert_one(mydict)
      print(x)

    return render_template('second.html', msge = msge,name=name,age=age,phno=phoneNumber,email=Email)



@app.route('/downloadpdf',methods=['POST'])
def downloadpdf():
    phno = request.form['postId']
    print(phno)
    url=f"static/userinfo"
    print(url)
    dirs = os.listdir(url)
    return render_template("ex1.html",files = dirs,url=url,phone=phno)
    



app.run()

# %%
