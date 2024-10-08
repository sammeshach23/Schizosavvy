import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
import numpy as np
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the question dataset
questionnaire = [
    "I hear voices that other people do not hear.",
    "I believe that I am pervasive and on top of the world.",
    "It is difficult for me to converse or interact with others.",
    "I am confused or disorganized in my thoughts.",
    "I feel I am unable to experience emotions as intensely as others do.",
    "I feel my bodily movements appear to be slowed down or sped up.",
    "I feel my thoughts are being inserted or taken away by outside forces.",
    "I feel uncomfortable or paranoid when I am around others and try to avoid social situations.",
    "I am preoccupied with strange beliefs and thoughts.",
    "I see, feel, smell, or taste things that other people think aren't there."
]

options = ["Choose your response", "Never", "Rarely", "Sometimes", "Often", "Always"]

def prepare_data_for_lstm(answers):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(answers)
    sequences = tokenizer.texts_to_sequences(answers)
    X = pad_sequences(sequences, maxlen=100)

    # Example labels - Replace with actual schizophrenia stage data.
    y = np.array([0 if i % 3 == 0 else 1 if i % 3 == 1 else 2 for i in range(len(answers))])

    return X, y, tokenizer

def build_lstm_model(X, y):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(LSTM(units=128))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    try:
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
        model.save('lstm_model.h5')
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")

    return model

def predict_stage(lstm_model, tokenizer):
    X, _, _ = prepare_data_for_lstm(st.session_state.answers)
    try:
        predictions = lstm_model.predict(X)
        avg_pred = np.mean(predictions)

        if avg_pred < 0.3:
            stage = "Stage 1"
        elif avg_pred < 0.7:
            stage = "Stage 2"
        else:
            stage = "Stage 3"

        st.write(f"Based on your responses, the chatbot predicts your schizophrenia stage as: **{stage}**.")
        st.write("Please consult a mental health professional for further assistance.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

def provide_therapeutic_response(answer):
    responses = {
        "Never": "It's great to hear that you're not experiencing these symptoms.",
        "Rarely": "It's important to monitor any changes in how you feel. Take time for self-care.",
        "Sometimes": "It's okay to feel this way sometimes. Remember to reach out to loved ones or professionals if needed.",
        "Often": "You're not alone in feeling this way. It might help to talk with a mental health professional.",
        "Always": "These feelings seem to be affecting you frequently. Please consider reaching out for support from a mental health professional."
    }
    return responses.get(answer, "")

def handle_additional_questions(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        st.error(f"An error occurred while handling additional questions: {e}")
        return "Sorry, I encountered an error while processing your question."

def get_conversational_chain():
    prompt_template = """
    You are a chatbot designed to provide therapeutic responses for individuals with schizophrenia.
    Answer in a supportive, empathetic manner based on the context provided.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def main():
    st.set_page_config(page_title="Schizosavvy", page_icon="🤖", layout="wide")

    st.sidebar.title("About the App")
    st.sidebar.info("""
    **Schizosavvy** is an interactive chatbot designed to help individuals monitor their cognitive and emotional states through a series of questions.
    The chatbot uses an LSTM model to predict the stage of schizophrenia based on user responses. Additionally, it answers any follow-up questions in an empathetic manner.

    **Features:**
    - Cognitive and emotional state monitoring.
    - Schizophrenia stage prediction (Stage 1, Stage 2, Stage 3).
    - Interactive therapeutic conversation.
    - Additional support for user questions.

    Please remember that this chatbot is not a replacement for professional mental health advice.
    """)

    st.markdown("""
    <style>
    .stApp {
        font-family: Arial, sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Schizosavvy")
    st.subheader("An empathetic chatbot for cognitive and emotional support.")

    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'therapeutic_response' not in st.session_state:
        st.session_state.therapeutic_response = ""

    if st.session_state.current_question < len(questionnaire):
        question = questionnaire[st.session_state.current_question]
        st.write(f"**Question {st.session_state.current_question + 1}:** {question}")
        answer = st.radio("Choose your response:", options[1:], key=f"question_{st.session_state.current_question}")

        if answer:
            st.session_state.therapeutic_response = provide_therapeutic_response(answer)
            st.write(f"**Response:** {st.session_state.therapeutic_response}")

            if st.button("Next", key=f"next_{st.session_state.current_question}"):
                st.session_state.answers.append(answer)
                st.session_state.current_question += 1
                st.session_state.therapeutic_response = ""
                st.rerun()
        else:
            st.write("Please select an option to proceed.")
    else:
        if os.path.exists('lstm_model.h5'):
            lstm_model = load_model('lstm_model.h5')
        else:
            X, y, tokenizer = prepare_data_for_lstm(st.session_state.answers)
            lstm_model = build_lstm_model(X, y)

        predict_stage(lstm_model, tokenizer)

        user_question = st.text_input("Do you have any additional questions?")
        if user_question:
            response = handle_additional_questions(user_question)
            st.write(f"**Chatbot's response:** {response}")

if __name__ == "__main__":
    main()
