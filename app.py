import streamlit as st
import pickle
from PIL import Image

def main():
    # Front-end elements of the web page with enhanced styling
    html_temp = """ 
    <style>
        body {
            background-color: #e6f7ff;
            font-family: 'Arial', sans-serif;
        }
        .main-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);
            margin: 20px;
        }
        .header {
            background: linear-gradient(to right, #ffcc00, #ff6600);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 28px;
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .stTextInput, .stSelectbox, .stNumberInput, .stTextArea {
            font-size: 18px;
            color: #333;
        }
    </style>
    <div class="header">Streamlit Loan Prediction ML App</div>
    <div class="main-container">
    """
    # Load the saved model and TF-IDF vectorizer
    try:
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading model or TF-IDF: {e}")

    # Display the front-end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Display image
    image = Image.open("fake.jpg")
    st.image(image, width=800)
     
    # Input text box
    user_input = st.text_area("Enter the news text")

    if st.button("Submit"):
        if user_input.strip():  # Check if input is not empty
            # Transform the input using the TF-IDF vectorizer
            input_tfidf = tfidf.transform([user_input])

            # Make prediction
            prediction = model.predict(input_tfidf)

            # Display result
            if prediction[0] == '0':
                st.success("Not Fake News")
            else:
                st.error("Fake News")
        else:
            st.warning("Please enter some text before submitting!")

if __name__ == "__main__":
    main()
