import streamlit as st
from py2 import return_result
import random
import time
import matplotlib.pyplot as plt
import numpy as np

import cv2
from keras.models import load_model
import math
from PIL import Image
from matplotlib.animation import FuncAnimation


st.title("Emotional Detection")
col1, col2, col3 = st.columns(3)
model = load_model('model.h5')
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# Checkbox in the right column
run = col3.checkbox('Run')

# Placeholder for the image in the right column
FRAME_WINDOW = col3.image([])

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if run:
    camera = cv2.VideoCapture(0)  # Use 0 to indicate the built-in webcam.
    col3_placeholder = col3.empty()
    while run:
        ret, frame = camera.read()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Loop over the face detections
        for (x, y, w, h) in faces:
            # Extract face ROI
            faceROI = gray[y:y + h, x:x + w]

            # Resize to be compatible with the model
            faceROI = cv2.resize(faceROI, (48, 48))

            # Scale pixel values to [0, 1]
            faceROI = faceROI / 255.0

            # Add batch dimension
            faceROI = np.expand_dims(faceROI, axis=0)

            # Add channel dimension
            faceROI = np.expand_dims(faceROI, axis=-1)

            # Make prediction on the face
            prediction = model.predict(faceROI)

            # Get the label with highest probability
            pred_label = LABELS[np.argmax(prediction)]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put predicted label near the face
            cv2.putText(frame, pred_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Put the labels under camera
            col3_placeholder.write(pred_label)
        # Display the resulting frame
        cv2.imshow('Face Expression Recognition', frame)
        FRAME_WINDOW.image(frame)

else:
    col3.write('Stopped')


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    if st.button("Reset"):
        st.session_state.messages = []

form1 = st.form(key='form1')


store = {}
tea = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        markdown_content = message["content"]
        st.markdown(markdown_content, help=message.get("result", None))
        if message.get("emotional_probabilities"):
            tea.append(message.get("emotional_probabilities"))
st.sidebar.title("Final")
if st.sidebar.button("Show"):
    mean_emotions = {}
    for emotion in tea[0].keys():
        mean_emotion = [entry[emotion]*id for id,entry in zip(range(1,len(tea)+1),tea)]
        mean_emotion = np.array(mean_emotion)
        mean_emotion = np.sum(mean_emotion / np.sum(np.arange(1,len(tea)+1)))
        print(mean_emotion)
        # mean_emotion = mean_emotion / sum(range(1,len(tea)))
        mean_emotions[emotion] = mean_emotion


    fig, ax = plt.subplots()
    ax.bar(mean_emotions.keys(), mean_emotions.values())
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Probability")
    ax.set_title("Final Result")
    for emotion, probability in mean_emotions.items():
        ax.text(emotion, probability, f"{probability:.2f}", ha='center', va='bottom')
    st.sidebar.pyplot(fig)
    max_emotion = max(mean_emotions, key=mean_emotions.get)
    st.sidebar.write(f"Final Result: {max_emotion} ({mean_emotions[max_emotion]})")

if chat_box := st.chat_input("Your Sentence!"):
    if chat_box:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": chat_box})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(chat_box)

        # Get emotional response
        assistant_response = return_result(chat_box)

        # Check joy probability
        joy_probability = 0.0
        sadness_probability = 0.0
        anger_probability = 0.0
        love_probability = 0.0
        surprise_probability = 0.0
        fear_probability = 0.0

        # Extract emotion probabilities from response
        for line in assistant_response.split('\n'):
            if "joy" in line:
                joy_probability = float(line.split(':')[1].strip()[:-1])
            elif "sadness" in line:
                sadness_probability = float(line.split(':')[1].strip()[:-1])
            elif "anger" in line:
                anger_probability = float(line.split(':')[1].strip()[:-1])
            elif "love" in line:
                love_probability = float(line.split(':')[1].strip()[:-1])
            elif "surprise" in line:
                surprise_probability = float(line.split(':')[1].strip()[:-1])
            elif "fear" in line:
                fear_probability = float(line.split(':')[1].strip()[:-1])
        emotional_probabilities = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "love": 0.0,
            "surprise": 0.0,
            "fear": 0.0
        }

        # Extract emotion probabilities from response
        for line in assistant_response.split('\n'):
            for emotion in emotional_probabilities.keys():
                if emotion in line:
                    emotional_probabilities[emotion] = float(line.split(':')[1].strip()[:-1])
        # Determine response based on emotion probabilities
        if joy_probability > 45:
            response_text = "Glad to hear that!"
        elif sadness_probability > 45:
            response_text = "I'm here for you. Is there something you'd like to talk about?"
        elif anger_probability > 45:
            response_text = "I'm sorry to hear that. How can I assist you in resolving the issue?"
        elif love_probability > 45:
            response_text = "That's wonderful! Love is a beautiful feeling."
        elif surprise_probability > 45:
            response_text = "Wow! What's the story behind this surprise?"
        elif fear_probability > 45:
            response_text = "It's okay to feel fear sometimes. How can I help you overcome it?"
        else:
            response_text = "I'm here to assist you. What else would you like to know or discuss?"
        rr = return_result(chat_box)
        ans = rr.split('\n')
        result = ""
        for i in ans:
            result += i
        content_order = len(st.session_state.messages) // 2


        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate typing with a delay
            for char in response_text:
                full_response += char
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response, help = result)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text, "result": result , "content_order": content_order, "emotional_probabilities": emotional_probabilities})
