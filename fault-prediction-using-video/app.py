import logging
from logging import error, info
from venv import create
import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
import tempfile
from keras.layers import *
from keras.models import Sequential
import file_utils
import time
import math

logging.basicConfig(level=logging.INFO)


# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["Non Accident","Not Tesla Fault","Tesla Fault"]   


st.header("Fault prediction using video evidence")




def load_lrcn_model():
    # model = create_LRCN_model()
    # model.load_weights('/Users/vsolanki/workspace/Webtool/fault-prediction-using-video/model/LRCN_model___Date_Time_2022_09_14__17_23_36___Loss_0.8800032734870911___Accuracy_0.6571428775787354.h5')
    model = load_model('model/LRCN_model___Date_Time_2022_09_14__17_23_36___Loss_0.8800032734870911___Accuracy_0.6571428775787354.h5')
    return model

def predict(file_path):
    
    start_time= time.time()

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(file_path)
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read() 

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    inference_time = time.time() - start_time
    
    # Display the predicted action along with the prediction confidence.

    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    
    st.info(f'Prediction: {predicted_class_name}')
    st.info(f"Confidence: {'{:.0%}'.format(round(predicted_labels_probabilities[predicted_label],2))}")
    st.info(f'Inference time: {str(round(inference_time, 2))} Sec')


    # Release the VideoCapture object. 
    video_reader.release()


#Load the model
model =load_lrcn_model()

#Pick the file
video_file = st.file_uploader("Choose a video file")
if video_file is not None:

    video_bytes = video_file.read()
    st.video(video_bytes)

    file_name= ''
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(video_bytes)
        file_name=tfile.name

    with st.spinner("Please wait.."):
        predict(file_name)

    info(f"[INFO] Temp file delete is: {file_name}")

    is_deleted= file_utils.delete_file(file_name)
    info(f"[INFO] File deleted? : {is_deleted}")




    

    



