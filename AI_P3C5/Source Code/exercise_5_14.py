# Exercise 5.14: Streamlit Webcam Image Classification dengan VGG19
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

# Impor model yang dibutuhkan, termasuk VGG19
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3,
    vgg19
)
from keras import backend as K

# Gunakan cache_resource agar webcam tidak dibuka berulang kali
@st.cache_resource
def get_cap():
    # Gunakan 0 untuk webcam, atau masukkan path file video (seperti Exercise 5.13)
    return cv2.VideoCapture(0)

# Gunakan cache_resource agar model hanya di-load satu kali ke memori
@st.cache_resource
def load_keras_model(model_name):
    K.clear_session()
    if model_name == "VGG16":
        return vgg16.VGG16(weights='imagenet')
    elif model_name == "RESNET50":
        return resnet50.ResNet50(weights='imagenet')
    elif model_name == "MOBILENET":
        return mobilenet.MobileNet(weights='imagenet')
    elif model_name == "INCEPTION_V3":
        return inception_v3.InceptionV3(weights='imagenet')
    elif model_name == "VGG19":
        return vgg19.VGG19(weights='imagenet')

# Fungsi prediksi tunggal untuk semua model
def predict_frame(cam_frame, image_size, model, model_name):
    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    # Preprocess input sesuai model yang dipilih
    if model_name == "VGG16":
        processed_image = vgg16.preprocess_input(image_batch.copy())
    elif model_name == "RESNET50":
        processed_image = resnet50.preprocess_input(image_batch.copy())
    elif model_name == "MOBILENET":
        processed_image = mobilenet.preprocess_input(image_batch.copy())
    elif model_name == "INCEPTION_V3":
        processed_image = inception_v3.preprocess_input(image_batch.copy())
    elif model_name == "VGG19":
        processed_image = vgg19.preprocess_input(image_batch.copy())
        
    predictions = model.predict(processed_image)
    label = decode_predictions(predictions)
    
    # Menampilkan teks hasil prediksi pada frame video
    cv2.putText(cam_frame, f"{model_name}: {label[0][0][1]}, {label[0][0][2]:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return cam_frame


# Konfigurasi UI Streamlit
st.title("Webcam Image Classification")
st.sidebar.markdown("# Image Classification")

# Menambahkan VGG19 ke dalam daftar pilihan
option = st.sidebar.selectbox(
    'Select a Deep Learning Model:',
    ["VGG16", "RESNET50", "MOBILENET", "INCEPTION_V3", "VGG19"], index=0)

st.sidebar.write('You selected:', option)

# Load model dan tentukan ukuran gambar berdasarkan pilihan
model = load_keras_model(option)
image_size = 299 if option == "INCEPTION_V3" else 224

cap = get_cap()
frameST = st.empty()

# Loop untuk membaca frame dari kamera/video secara terus-menerus
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Kamera atau video tidak dapat diakses / sudah berakhir.")
        break
        
    # Proses prediksi dan tampilkan gambar
    frame = predict_frame(frame, image_size, model, option)
    frameST.image(frame, channels="BGR")