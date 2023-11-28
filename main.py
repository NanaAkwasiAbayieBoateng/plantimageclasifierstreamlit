
"""

The 22 classes of various plant diseases  :
 'Tomato_Target_Spot',
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
 'Bean_Healthy'




"""



import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st


@st.cache_data
@st.cache_resource
def load_model():
  
  #model = load_model("model/efficientnetbo.h5")
  model=tf.keras.models.load_model("model/efficientnetbo.h5")
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
  

def getPrediction(filename):
    
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



    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    
    
    #Load model
    #my_model=load_model("model/efficientnetbo.h5")
    my_model=load_model()
    
    SIZE = 224 #Resize to same size as training images
    #img_path = 'static/images/'+filename
    img = np.asarray(Image.open(filename).resize((SIZE,SIZE)))
    
    #img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = my_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    pred_prob = str(np.max(pred))
    print(f"Plant Disease is: {pred_class} with probability:{pred_prob}")
    return pred_class,pred_prob


label =getPrediction('tomato_healthy.jpg')

