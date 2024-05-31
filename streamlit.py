import streamlit as st
from py1 import return_result , return_result2
import random
from translate import translate
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model
from modelT import answer_question, tokenizer, transformer
import torch
import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model2 = AutoModelForCausalLM.from_pretrained('./finetune')
tokenizer2 = AutoTokenizer.from_pretrained('./finetune')
if "language" not in st.session_state:
    st.session_state.language = "English"  # Default language
# Language selection
st.sidebar.title("Language Selection/Lựa chọn ngôn ngữ")
language = st.sidebar.selectbox("Choose your language/Chọn ngôn ngữ của bạn:", ["English", "Vietnamese"])

if st.session_state.language != language:
    st.session_state.language = language
    st.session_state.messages = []
st.title(translate("title"))
col1, col2, col3 = st.columns(3)
model = load_model('model.h5')
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Checkbox in the right column
run = col3.checkbox(translate('run'))

FRAME_WINDOW = col3.image([])

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_camera():
    return cv2.VideoCapture(0)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        faceROI = gray[y:y + h, x:x + w]
        faceROI = cv2.resize(faceROI, (56, 56)) / 255.0
        faceROI = np.expand_dims(faceROI, axis=(0, -1))
        prediction = model.predict(faceROI)
        pred_label = LABELS[np.argmax(prediction)]
        translated_label = translate(pred_label)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, pred_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        col3_placeholder.write(translated_label)
    return frame

if run:
    camera = load_camera()
    col3_placeholder = col3.empty()
    while run:
        ret, frame = camera.read()
        frame = process_frame(frame)
        FRAME_WINDOW.image(frame)
else:
    col3.write(translate('stopped'))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    if st.button("Reset"):
        st.session_state.messages = []

form1 = st.form(key='form1')

store = {}
tea = []
print(tea)
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        markdown_content = message["content"]
        st.markdown(markdown_content, help=message.get("result", None))
        if message.get("emotional_probabilities"):
            tea.append(message.get("emotional_probabilities"))

st.sidebar.title(translate("final_result"))
if st.sidebar.button(translate("show")):
    mean_emotions = {}
    merged_dict = {key: [d[key] for d in tea] for key in tea[0]}
    for emotion in tea[0].keys():
        mean_emotion = [entry[emotion] * id for id, entry in zip(range(1, len(tea) + 1), tea)]
        mean_emotion = np.array(mean_emotion)
        mean_emotion = np.sum(mean_emotion / np.sum(np.arange(1, len(tea) + 1)))
        mean_emotions[emotion] = mean_emotion

    fig, ax = plt.subplots()
    translated_emotions = [translate(emotion) for emotion in mean_emotions.keys()]
    ax.bar(translated_emotions, mean_emotions.values())
    ax.set_xlabel(translate("emotion"))
    ax.set_ylabel(translate("probability"))
    ax.set_title(translate("final_result"))
    for emotion, probability in mean_emotions.items():
        ax.text(translate(emotion), probability, f"{probability:.2f}", ha='center', va='bottom')
    st.sidebar.pyplot(fig)
    max_emotion = max(mean_emotions, key=mean_emotions.get)
    st.sidebar.write(f"{translate('final_result')}: {translate(max_emotion)} ({mean_emotions[max_emotion]})")

def generate_response(chat_box, language):
    if language == "Vietnamese":
        inputs = tokenizer2("Bạn là một trợ lý tư vấn tâm lý nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn cho người dùng: " + chat_box, return_tensors="pt")
        output = model2.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                temperature=0.7,
                repetition_penalty=1.05, )

        decoded_output = tokenizer2.decode(output[0], skip_special_tokens=True)
        start = decoded_output.find('dùng:')
        final_response = decoded_output[start+5:]
        assistant_response2 = return_result(chat_box)
        print(assistant_response2)

    else:
        result = answer_question(chat_box, transformer, tokenizer)
        response_text = tokenizer.decode(result[0].numpy().tolist()[2:])
        final_response = response_text
        assistant_response2 = return_result2(chat_box)

    return final_response, assistant_response2

def extract_emotional_probabilities(assistant_response2,language):
    if language == "English":
        emotional_probabilities = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "love": 0.0,
            "surprise": 0.0,
            "fear": 0.0
        }
    else:
        emotional_probabilities = {
            "vui mừng": 0.0,
            "buồn bã": 0.0,
            "tức giận": 0.0,
            "yêu thương": 0.0,
            "bất ngờ": 0.0,
            "sợ hãi": 0.0
        }
    for line in assistant_response2.split('\n'):
        for emotion in emotional_probabilities.keys():
            if emotion in line:
                    emotional_probabilities[emotion] = float(line.split(':')[1].strip()[:-1])
    return emotional_probabilities

if chat_box := st.chat_input(translate("your_sentence")):
    st.session_state.messages.append({"role": "user", "content": chat_box})

    final_response, assistant_response2 = generate_response(chat_box, st.session_state.language)
    emotional_probabilities = extract_emotional_probabilities(assistant_response2,st.session_state.language)
    print(emotional_probabilities)
    with st.chat_message("user"):
            st.markdown(chat_box)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for char in final_response:
            full_response += char
            time.sleep(0.02)
            message_placeholder.markdown(full_response + " ", help=str(assistant_response2))
        st.markdown(translate("You want to meet a doctor?"))
        if st.button(translate("Click here!")):
            pass

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_response,
        "result": assistant_response2,
        "content_order": len(st.session_state.messages) // 2,
        "emotional_probabilities": emotional_probabilities
    })
