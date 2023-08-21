import streamlit as st
from py1 import return_result
st.title("Emotional Detection")
st.write("cre: hoangtrhien")

form1 = st.form(key='form1')


chat_box = form1.text_input("Enter your sentence")
b1 = form1.form_submit_button(label="send")
if b1:
    rr = return_result(chat_box)
    ans = rr.split('\n')
    for i in ans:
        st.write(i)