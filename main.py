import streamlit as st
from preprocessing import preprocess, model, vectorizer

# Streamlit app UI
st.title("Sentiment Analysis App")
st.write("Enter a sentence and get its sentiment prediction.")

# Text input
user_input = st.text_area("Enter your sentence here:")

# Prediction button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess the user input
        processed_text = preprocess(user_input)

        # Vectorize the preprocessed text
        input_vector = vectorizer.transform([processed_text])

        # Predict sentiment
        prediction = model.predict(input_vector)[0]

        # Map numeric prediction to label
        sentiment_label = "Positive" if prediction == 1 else "Negative"

        # Output
        st.success(f"Predicted Sentiment: **{sentiment_label}**")