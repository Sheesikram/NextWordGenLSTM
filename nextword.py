import streamlit as st
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
import re
import time


st.image("download.jpg", use_column_width=True)

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("LSTM Next Word Predictor")

# Input: Document text for training
document = st.text_area(
    "Enter the document text for training:",
    value="""The Impact of Technology on Modern Society Technology has become an integral part of human civilization, shaping how we live, work, and communicate. From the early days of the wheel and fire to the modern advancements in artificial intelligence and quantum computing, technology has continuously evolved, transforming societies and industries. This essay explores the profound impact of technology on modern society, discussing its benefits, challenges, and future implications.
The Evolution of Technology

Human civilization has always been driven by innovation. The discovery of fire and the invention of the wheel marked the beginning of technological advancements. The Industrial Revolution introduced mechanization, significantly boosting productivity and transforming economies. The 20th and 21st centuries have seen rapid technological progress, with developments in computing, telecommunications, and medical sciences revolutionizing daily life.

Technology in Communication

One of the most significant impacts of technology is in communication. The invention of the telephone allowed people to communicate over long distances, and the rise of the internet has taken connectivity to a whole new level. Social media platforms, video conferencing, and instant messaging have made it possible for people to stay connected regardless of geographical barriers. The rise of artificial intelligence has further improved communication through chatbots, virtual assistants, and real-time language translation.

Technology in Education

The education sector has undergone a major transformation due to technology. Online learning platforms, digital textbooks, and virtual classrooms have made education more accessible. Students can now learn from anywhere in the world, and teachers can use interactive tools to enhance their lessons. Technologies such as artificial intelligence and virtual reality are making learning more immersive and effective."""
    , height=300)

# Input: Initial text for prediction
initial_text = st.text_input("Enter initial text for prediction:", value="The Evolution")
num_tokens = st.slider("Number of words to predict:", 1, 10, 5)

### Preprocess the document, build vocabulary, and prepare training sequences ###

# Tokenize document and build vocabulary
tokens = word_tokenize(document.lower())
vocab = {"<unk>": 0}
countofTokens = Counter(tokens)
for token in countofTokens.keys():
    if token not in vocab:
        vocab[token] = len(vocab)

def split_into_sentences(text):
    pattern = r'(?<=[.!?])\s+'
    return re.split(pattern, text)

input_sentences = split_into_sentences(document)

def text_to_index(sentence, vocab):
    return [vocab[token] if token in vocab else vocab["<unk>"] for token in sentence]

input_numerical_sentences = []
for sent in input_sentences:
    tokens_sent = word_tokenize(sent.lower())
    input_numerical_sentences.append(text_to_index(tokens_sent, vocab))

training_seq = []
for sentence in input_numerical_sentences:
    for i in range(1, len(sentence)):
        training_seq.append(sentence[:i+1])

max_len = max([len(seq) for seq in training_seq])
padded_training_seq = []
for seq in training_seq:
    padded_training_seq.append([0] * (max_len - len(seq)) + seq)

padded_training_seq = torch.tensor(padded_training_seq, dtype=torch.long)
X = padded_training_seq[:, :-1]
y = padded_training_seq[:, -1]

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

### Define the LSTM Model ###
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)
    def forward(self, x):
        embedded = self.embedding(x)
        _, (final_hidden_state, _) = self.lstm(embedded)
        output = self.fc(final_hidden_state.squeeze(0))
        return output

model = LSTMModel(len(vocab))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 50
learning_rate = 0.001
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def pridiction(model, vocab, text):
    tokenized_text = word_tokenize(text.lower())
    numerical_text = text_to_index(tokenized_text, vocab)
    padded_text = torch.tensor([0] * ((max_len - 1) - len(numerical_text)) + numerical_text, dtype=torch.long).unsqueeze(0)
    output = model(padded_text.to(device))
    _, index = torch.max(output, dim=1)
    return text + " " + list(vocab.keys())[index]

### Streamlit Button: Train and Predict ###
if st.button("Train Model and Predict"):
    with st.spinner("Training the model..."):
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criteria(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    st.success("Training completed!")
    
    output_text = initial_text
    predictions = []
    for i in range(num_tokens):
        output_text = pridiction(model, vocab, output_text)
        predictions.append(output_text)
        time.sleep(0.5)
    
    st.write("### Prediction Output:")
    for pred in predictions:
        st.write(pred)
