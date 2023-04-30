import streamlit as st
import numpy as np
import cv2 as cv2
import numpy as np 
from  PIL import Image, ImageEnhance
img = cv2.imread("test3.jpg")
#Create two columns with different width
col1, col2 = st.columns( [0.8, 0.2])
with col1:               # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your image here...</p>', unsafe_allow_html=True)


st.sidebar.markdown('<p class="font">Road Extraction using CNN in presence of Shadow</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
     st.write("""
        Intially image is taken and shadow is minimized to avoid occusion.\n  \nThis app was created by Suruthi Mayavel!
     """)

#Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov
#Shadow removal


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    print(uploaded_file.name)
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">With Shadow</p>',unsafe_allow_html=True)
        st.image(image,width=300) 
    with col2:
        st.markdown('<p style="text-align: center;">After Minimizing shadow</p>',unsafe_allow_html=True)
        mat=cv2.imread(uploaded_file.name)
        print(mat)
        shad = shadow_remove(mat)
        cv2.imwrite('img/after_shadow_remove1.jpg', shad)
        im2=Image.open('img/after_shadow_remove1.jpg')
        st.image(im2,width=300)