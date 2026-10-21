# Exercise 5.13
import streamlit as st
import cv2
import numpy as np

@st.cache_resource
def get_cap():
    # Mengganti 0 dengan nama file video
    return cv2.VideoCapture(r'E:\Tugas\Semester 5\AI\MasterHUb\AI_P3C5\Output\rocket-launch.mkv')

cap = get_cap()

frameST = st.empty()
contrast = st.sidebar.slider('Contrast', 0, 100, 50) / 50.0
brightness = st.sidebar.slider('Brightness', 0, 100, 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Done processing !!!")
        cv2.waitKey(3000)
        cap.release()
        break
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[:,:,2] = np.clip(contrast * frame[:,:,2] + brightness, 0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    
    frameST.image(frame, channels="BGR")