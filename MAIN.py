# Modules
import streamlit as st
from pyrebase import pyrebase
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from keras.utils import load_img, img_to_array
from keras import preprocessing
from streamlit_option_menu import option_menu
import cv2
from keras.models import load_model
import streamlit as st


# To store Configuration key in database
firebaseconfig = {
    'apiKey': "AIzaSyBjwkNm-FLWgH5Rge6dxby_dB5ySKNG6yk",
    'authDomain': "applediseasedetection1.firebaseapp.com",
    'databaseURL': "https://applediseasedetection1-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "applediseasedetection1",
    'storageBucket': "applediseasedetection1.appspot.com",
    'messagingSenderId': "229891354768",
    'appId': "1:229891354768:web:5c1e0d28931d250f10e815",
    'measurementId': "G-PBVPDY303C"
}


# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseconfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()
# Authentication
with st.sidebar:
    choice = option_menu(
        menu_title='Main Menu',
        options=['Login', 'Signup'],
        icons=['door-open', 'app-indicator'],
        orientation='horizontal'
    )

st.title('Apple Leaf Disease Detection')
st.sidebar.title('User Authentication')
st.image('welcome.png')

# User Input

email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password', type='password')

# Sign Up Block


def Auth():
    user = auth.sign_in_with_email_and_password(email, password)
    return user["registered"]


if choice == 'Sign Up':
    handle = st.sidebar.text_input('Please enter your Username')
    submit = st.sidebar.button('Create My Account')

    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account has been created successfully')

        # Sign in
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("Name").set(handle)
        db.child(user['localId']).child("ID").set(user['localId'])
        st.title('Welcome ' + handle)
        st.info('Login via dropdown Login Option')

# Login Block
if choice == 'Login':
    # print("logout")
    # login = st.sidebar.button('Login')
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email, password)

        st.write(
            '<style>div.row-widget.stRadio > div{flex-direction:column;}</style>', unsafe_allow_html=True)

        bio = st.radio('Pages', ['Home', 'Detection'])

# Home Page
        if bio == 'Home':
            st.header('Apple Leaf Disease Detection')
            st.image("welcome 2.jpg")
            st.write('''Welcome to the Apple Leaf Disease Detection system - the fast and easy way to identify and manage diseases in your apple leaves. This project named "Apple Leaf Disease Detection using CNN" is a minor project from BEI076.
      The model is trained on a large dataset of annotated apple leaf images, and is capable of detecting several common diseases such as apple scab, cedar apple rust, and black rot.''')
        # Detection Page

        elif bio == 'Detection':
            st.title('Detect Image')
            plant_image = st.file_uploader(
                "Choose an image...", type=['jpg', 'png', 'jpeg'])
            submit = st.button('Predict')
        # On predict button click
            if submit:

                if plant_image is not None:
                    file_bytes = np.asarray(
                        bytearray(plant_image.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)

                    model = load_model(
                        r'C:\Users\hp\Desktop\Minor project\trained_model_inception.h5')
                    CLASS_NAMES = ['apple_scab', 'black_rot',
                                   'cedar_apple_rust', 'healthy']
                    # Displaying the image
                    st.image(opencv_image, channels="BGR")
                    st.write(opencv_image.shape)
                    # Resizing the image
                    opencv_image = cv2.resize(opencv_image, (224, 224))
                    # Convert image to 4 Dimension
                    opencv_image.shape = (1, 224, 224, 3)
                    # Make Prediction
                    Y_pred = model.predict(opencv_image)
                    result = CLASS_NAMES[np.argmax(Y_pred)]
                    st.title("The image uploaded is:  {} ".format(result))
