#%%writefile app.py
import streamlit as st
#from tensorflow.keras.applications.imagenet_utils import decode_predictions
#import cv2
from PIL import Image, ImageOps
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf
#from main import getPrediction
import os
from sklearn.preprocessing import LabelEncoder

 
@st.cache_data
@st.cache_resource
def load_model():
  
  #model = load_model("model/efficientnetbo.h5")
  model=tf.keras.models.load_model("model/efficientnetbo.h5")
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Image Classification
         """
         )
 
file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
 
#def upload_predict(upload_image, model):
    
        #size = (180,180)    
        #image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        #image = np.asarray(image)
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        
        #img_reshape = img_resize[np.newaxis,...]
    
        #prediction =   model.predict(img_reshape)
        #pred_class  =  decode_predictions(prediction,top=1)
 #       out  =        getPrediction(upload_image)
  #      pred_class =  out[0]
  #      pred_prob =    out[1]
 #       return pred_class,pred_prob



def getPrediction(filename):
 '''   
    classes =  ['Tomato_Target_Spot',
 'Tomato_YellowLeaf_Curl_Virus',
 'Maize_Healthy',
 'Maize_Cercospora_Leaf_Spot (CLS)',
 'Tomato_Late_Blight',
 'Tomato_Mosaic_Virus',
 'Tomato_Early_Blight',
 'Maize_Leaf_Blight (MLB)',
 'Maize_Fall_Army_Worm (FAW)',
 'Tomato_Spider_Mite',
 'Cassava_Healthy',
 'Tomato_Bacterial_Spot',
 'Bean_Rust (BR)',
 'Tomato_Leaf_Mold',
 'Maize_Lethal_Necrosis (MLN)',
 'Cassava_Brown_Streak_Disease (CBSD)',
 'Bean_Angular_Leaf_Spot (ALS)',
 'Tomato_Septoria_Leaf_Spot',
 'Tomato_Healthy',
 'Maize_Streak_Virus (MSV)',
 'Cassava_Mosaic_Disease (CMD)',
 'Bean_Healthy']
'''


    class_names = {
    '0':'Tomato_Target_Spot', '1':'Tomato_YellowLeaf_Curl_Virus',
     '2':'Maize_Healthy','3':'Maize_Cercospora_Leaf_Spot (CLS)',
     '4':'Tomato_Late_Blight','5':'Tomato_Mosaic_Virus',
      '6':'Tomato_Early_Blight','7':'Maize_Leaf_Blight (MLB)',
      '8':'Maize_Fall_Army_Worm (FAW)','9':'Tomato_Spider_Mite',
     '10':'Cassava_Healthy','11':'Tomato_Bacterial_Spot',
     '12':'Bean_Rust (BR)','13':'Tomato_Leaf_Mold',
     '14':'Maize_Lethal_Necrosis (MLN)','15':'Cassava_Brown_Streak_Disease (CBSD)',
     '16':'Bean_Angular_Leaf_Spot (ALS)','17':'Tomato_Septoria_Leaf_Spot',
     '18':'Tomato_Healthy','19':'Maize_Streak_Virus (MSV)',
     '20':'Cassava_Mosaic_Disease (CMD)','21':'Bean_Healthy'
  }



    #le = LabelEncoder()
    #le.fit(classes)
    #le.inverse_transform([2])
    
    
    #Load model
    #my_model=load_model("model/efficientnetbo.h5")
    my_model=load_model()
    
    SIZE = 224 #Resize to same size as training images
    #img_path = 'static/images/'+filename
    img = np.asarray(Image.open(filename).resize((SIZE,SIZE)) ,dtype='float64')
    img = img.astype(np.float64)
    #img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    predictions = my_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    #pred_class = le.inverse_transform([np.argmax(pred)])[0]
    predicted_class = class_names.get(str(predictions.argmax()))
    predicted_prob = str(np.max(prediction))
    print(f"Plant Disease is: {predicted_class} with probability:{predicted_prob}")
    return predicted_class,predicted_prob







if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    #predictions = upload_predict(image, model)
    #image_class = str(predictions[0][0][1])
    #score=np.round(predictions[0][0][2]) 
    
    out  =        getPrediction(file)
    pred_class =  out[0]
    pred_prob =    np.round(float(out[1]),3)
    st.write("The image is classified as",pred_class)
    st.write("The similarity score is approximately",pred_prob)
    print("The image is classified as ",pred_class, "with a similarity score of",pred_prob)