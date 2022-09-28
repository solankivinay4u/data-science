# Import the required libraries.
import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from moviepy.editor import *
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from train import LRCN_model,CLASSES_LIST,SEQUENCE_LENGTH
from preprocess import IMAGE_HEIGHT , IMAGE_WIDTH


# from train import LRCN_model_training_history

# def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
#     '''
#     This function will plot the metrics passed to it in a graph.
#     Args:
#         model_training_history: A history object containing a record of training and validation 
#                                 loss values and metrics values at successive epochs
#         metric_name_1:          The name of the first metric that needs to be plotted in the graph.
#         metric_name_2:          The name of the second metric that needs to be plotted in the graph.
#         plot_name:              The title of the graph.
#     '''
    
#     # Get metric values using metric names as identifiers.
#     metric_value_1 = model_training_history.history[metric_name_1]
#     metric_value_2 = model_training_history.history[metric_name_2]
    
#     # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
#     epochs = range(len(metric_value_1))

#     # Plot the Graph.
#     plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
#     plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)

#     # Add title to the plot.
#     plt.title(str(plot_name))

#     # Add legend to the plot.
#     plt.legend()

# # Visualize the training and validation loss metrices.
# plot_metric(LRCN_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')  

# # Visualize the training and validation accuracy metrices.
#plot_metric(LRCN_model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

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
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
    # Release the VideoCapture object. 
    video_reader.release()

# Construct tihe nput youtube video path
input_video_file_path = r"C:\Users\earya\ekta_repo\aug22\vedio_classifier\Lstm\Lstm\test_benchmark\3566323_(Rear) 2020-03-07_01-39-52-back.mp4"

# Perform Single Prediction on the Test Video.
predict_single_action(input_video_file_path, SEQUENCE_LENGTH)

# Display the input video.
VideoFileClip(input_video_file_path, audio=False, target_resolution=(300,None)).ipython_display()    