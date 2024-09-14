import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the question dataset (example questions for schizophrenia)
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

# Options for the answers, with a prompt as the first option
options = ["Never", "Rarely", "Sometimes", "Often", "Always"]

# Function to prepare data for a classifier and generate labels
def prepare_data_for_classifier(answers):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(answers).toarray()
    
    # Example labels - Replace with actual schizophrenia stage data.
    # 0: Stage 1, 1: Stage 2, 2: Stage 3
    y = np.array([0 if i % 3 == 0 else 1 if i % 3 == 1 else 2 for i in range(len(answers))])
    
    return X, y, vectorizer

# Build and train a Logistic Regression model
def build_classifier_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Function to predict the stage of schizophrenia as Stage 1, Stage 2, or Stage 3
def predict_stage(classifier_model, vectorizer):
    X = vectorizer.transform(st.session_state.answers).toarray()
    predictions = classifier_model.predict(X)
    avg_pred = np.mean(predictions)  # Average prediction across questions
    
    # Predict stage based on cognitive level
    if avg_pred < 1:
        stage = "Stage 1"
    elif avg_pred < 2:
        stage = "Stage 2"
    else:
        stage = "Stage 3"
    
    st.write(f"Based on your responses, the chatbot predicts your schizophrenia stage as: **{stage}**.")
    st.write("Please consult a mental health professional for further assistance.")

# Function to provide therapeutic response based on the selected answer
def provide_therapeutic_response(answer):
    if answer == "Never":
        response = "It's great to hear that you're not experiencing these symptoms."
    elif answer == "Rarely":
        response = "It's important to monitor any changes in how you feel. Take time for self-care."
    elif answer == "Sometimes":
        response = "It's okay to feel this way sometimes. Remember to reach out to loved ones or professionals if needed."
    elif answer == "Often":
        response = "You're not alone in feeling this way. It might help to talk with a mental health professional."
    elif answer == "Always":
        response = "These feelings seem to be affecting you frequently. Please consider reaching out for support from a mental health professional."
    else:
        response = ""  # No response if the prompt is selected
    
    return response

# Function to handle additional questions from the user
def handle_additional_questions(user_question, classifier_model, vectorizer):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to create a conversational AI chain with prompt engineering
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

# Streamlit app
def main():
    st.set_page_config(page_title="Schizosavvy", page_icon="🤖", layout="wide")

    # Sidebar for app information
    st.sidebar.title("About the App")
    st.sidebar.info("""
    **Schizosavvy** is an interactive chatbot designed to help individuals monitor their cognitive and emotional states through a series of questions. 
    The chatbot uses a classifier model to predict the stage of schizophrenia based on user responses. Additionally, it answers any follow-up questions in an empathetic manner.
    
    **Features:**
    - Cognitive and emotional state monitoring.
    - Schizophrenia stage prediction (Stage 1, Stage 2, Stage 3).
    - Interactive therapeutic conversation.
    - Additional support for user questions.

    Please remember that this chatbot is not a replacement for professional mental health advice.
    """)

    st.title("🧠 Schizosavvy")
    st.subheader("An empathetic chatbot for cognitive and emotional support.")

    # Initialize session state
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'therapeutic_response' not in st.session_state:
        st.session_state.therapeutic_response = ""

    # Display questions one by one
    if st.session_state.current_question < len(questionnaire):
        # Ask the current question
        question = questionnaire[st.session_state.current_question]
        st.write(f"**Question {st.session_state.current_question + 1}:** {question}")
        answer = st.radio("Choose your response:", options, key=f"question_{st.session_state.current_question}")

        if answer:
            # Provide therapeutic response after selecting an answer
            st.session_state.therapeutic_response = provide_therapeutic_response(answer)
            st.write(f"**Response:** {st.session_state.therapeutic_response}")

            # Button to submit the answer and move to the next question
            if st.button("Next", key=f"next_{st.session_state.current_question}"):
                st.session_state.answers.append(answer)
                st.session_state.current_question += 1
                st.session_state.therapeutic_response = ""  # Clear therapeutic response for the next question
                st.experimental_rerun()  # Rerun to update the question
        else:
            st.write("Please select an option to proceed.")

    else:
        # Build and use the classifier model after all questions are answered
        X, y, vectorizer = prepare_data_for_classifier(st.session_state.answers)
        classifier_model = build_classifier_model(X, y)
        predict_stage(classifier_model, vectorizer)

        # Handle additional user questions
        user_question = st.text_input("Do you have any additional questions?")
        if user_question:
            response = handle_additional_questions(user_question, classifier_model, vectorizer)
            st.write(f"**Chatbot Response:** {response}")

if __name__ == "__main__":
    main()
