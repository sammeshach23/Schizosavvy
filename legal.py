def main():
    st.set_page_config(page_title="Schizosavvy", page_icon="🤖", layout="wide")

    # Sidebar for app information
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

    # Add a soothing background color
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
        answer = st.radio("Choose your response:", options[1:], key=f"question_{st.session_state.current_question}")

        if answer:
            # Provide therapeutic response after selecting an answer
            st.session_state.therapeutic_response = provide_therapeutic_response(answer)
            st.write(f"**Response:** {st.session_state.therapeutic_response}")

            # Button to submit the answer and move to the next question
            if st.button("Next", key=f"next_{st.session_state.current_question}"):
                st.session_state.answers.append(answer)
                st.session_state.current_question += 1
                st.session_state.therapeutic_response = ""  # Clear therapeutic response for the next question
                st.experimental_rerun()  # Rerun to update the question (if this is still valid)
        else:
            st.write("Please select an option to proceed.")

    else:
        # Build and use the LSTM model after all questions are answered
        X, y, tokenizer = prepare_data_for_lstm(st.session_state.answers)
        lstm_model = build_lstm_model(X, y)  # Now passing both X and y
        predict_stage(lstm_model, tokenizer)

        # Handle additional user questions
        user_question = st.text_input("Do you have any additional questions?")
        if user_question:
            response = handle_additional_questions(user_question, lstm_model, tokenizer)
            st.write(f"**Chatbot Response:** {response}")

if __name__ == "__main__":
    main()
