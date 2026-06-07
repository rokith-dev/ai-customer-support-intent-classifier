import pickle


# Load trained model
with open("models/intent_classifier.pkl", "rb") as file:
    model = pickle.load(file)


# Load TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


user_input = input("Enter customer issue: ")


clean_text = user_input.lower()


text_vector = vectorizer.transform([clean_text])


prediction = model.predict(text_vector)


print("\nPredicted Category:")
print(prediction[0])