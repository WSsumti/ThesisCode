import streamlit as st
import pandas as pd
import numpy as np
import torch as torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from model_demo import VisoBERT, PhoBERT_LSTM


def merge_text():
    input_text = st.sidebar.text_area(label='Text input', height=10, placeholder='Input Text here!')
    data = {
        'Text': input_text
    }
    merge_df = pd.DataFrame(data=data, index=[0])
    return merge_df

def chosen_model():
    model = st.sidebar.selectbox('Select our best model: ', ('PhoBERT-LSTM with RL and AL', 'ViSoBERT with RL and AL'))
    if model == 'PhoBERT-LSTM with RL and AL':
        model_type = 'phobert'
    else:
        model_type = 'viso'
    return model_type

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

st.write("""
# Emotion Recognition from Social Media Text App
This app which is trained with social media data, will classify writers's emotion from their text!
""")
st.sidebar.header('User Input') 
uploaded_file = st.sidebar.file_uploader("Upload your input CSV, XLSX file", type=["csv", "xlsx"], help="Your CSV/XLSX file should include column 'Text'")

file_type = ''

if uploaded_file is not None:
    if 'xlsx' in uploaded_file.name:
        file_type = 'xlsx'
    else:
        file_type = 'csv'

st.subheader('This is your input data from the file uploader: ')
text_df = pd.DataFrame(data={'Text':''}, index=[0])
if file_type == 'xlsx':
    text_df = pd.read_excel(uploaded_file, sheet_name=None)['Sheet1']
    st.dataframe(text_df, width=20, use_container_width=20)
elif file_type == 'csv':
    text_df = pd.read_csv(uploaded_file)
    st.dataframe(text_df, width=20, use_container_width=20)
else:
    st.dataframe(None, width=20, use_container_width=20)

merge_df = merge_text()
merge_df= merge_df[merge_df['Text'] != '']
st.subheader('This is your input data from the text input: ')
st.dataframe(merge_df, width=20, use_container_width=20)

model = chosen_model()

st.subheader('This is the merged data from both input option: ')
df = pd.concat([text_df, merge_df], ignore_index=True)
df = df[df['Text']!='']
st.dataframe(df, width=20, use_container_width=20)

# Model part:
viso_model = VisoBERT(7).to(device)
viso_model.load_state_dict(torch.load('thesis_model/visobert_ALRL_fold_1.pth'))
pb_model = PhoBERT_LSTM(7).to(device)
pb_model.load_state_dict(torch.load('thesis_model/phobert_fold_rl_al_12.pth'))
# Tokenizer part:
viso_tokenizer = AutoTokenizer.from_pretrained('uitnlp/visobert')
pb_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

softmax = nn.Softmax()

if model == 'phobert':
    if df['Text'] is not None:
        data = df['Text'].values.tolist()
        outputs = []
        probs = []
        for t in data:
            encoded = pb_tokenizer.encode_plus(
                t,
                max_length=50,
                truncation=True,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
            )
            inp = encoded['input_ids'].to(device)
            att = encoded['attention_mask'].to(device)
            output = pb_model(inp, att)
            _, y_pred = torch.max(output, dim=1)
            output = softmax(output)
            probs.append(output.cpu().detach().numpy()[0])
            outputs.append(y_pred.cpu().detach().numpy()[0])

        class_labels = {
            0: 'Enjoyment',
            1: 'Disgust',
            2: 'Sadness',
            3: 'Anger',
            4: 'Surprise',
            5: 'Fear',
            6: 'Other'
        }

        probs_df = pd.DataFrame(probs, columns=[class_labels[i] for i in range(len(class_labels))])

        st.subheader('Probability: ')
        st.table(probs_df)
        out_df = pd.DataFrame({
            'Class': outputs,
            'Emotion': [class_labels[idx] for idx in outputs]
        })
        st.dataframe(out_df)
else:
    if df['Text'] is not None:
        data = df['Text'].values.tolist()
        outputs = []
        probs = []
        for t in data:
            encoded = viso_tokenizer.encode_plus(
                t,
                max_length=50,
                truncation=True,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
            )
            inp = encoded['input_ids'].to(device)
            att = encoded['attention_mask'].to(device)
            output = viso_model(inp, att)
            _, y_pred = torch.max(output, dim=1)
            output = softmax(output)
            probs.append(output.cpu().detach().numpy()[0])
            outputs.append(y_pred.cpu().detach().numpy()[0])

        class_labels = {
            0: 'Enjoyment',
            1: 'Disgust',
            2: 'Sadness',
            3: 'Anger',
            4: 'Surprise',
            5: 'Fear',
            6: 'Other'
        }

        probs_df = pd.DataFrame(probs, columns=[class_labels[i] for i in range(len(class_labels))])

        st.subheader('Probability: ')
        st.table(probs_df)
        out_df = pd.DataFrame({
            'Class': outputs,
            'Emotion': [class_labels[idx] for idx in outputs]
        })
        st.dataframe(out_df)
