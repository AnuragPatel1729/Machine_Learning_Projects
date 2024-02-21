import os
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

class TextPreprocessor:
    def __init__(self):
        self.stopwords_set = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.cv = None

        current_directory = os.path.dirname(os.path.abspath(__file__))
        path_to_pkl = os.path.join(current_directory, 'count_vectorizer.pkl')

        # Load CountVectorizer during object creation
        with open(path_to_pkl, 'rb') as file:
            self.cv = pickle.load(file)

    def remove_html_tags(self, text):
        clean_text = re.sub('<.*?>', '', text)
        return clean_text

    def lower_casing(self, text):
        return text.lower()

    def remove_special_char(self, text):
        return ''.join(char if char.isalnum() else ' ' for char in text)

    def remove_space_in_between(self, text):
        return ' '.join(text.split())

    def remove_stopwords(self, text):
        return [word for word in text.split() if word not in self.stopwords_set]

    def review_lemmatize(self, words):
        return [self.lemmatizer.lemmatize(word) for word in words]

    def text_join(self, stem_list):
        return ' '.join(stem_list)

    def preprocess_text(self, text):
        text = self.remove_html_tags(text)
        text = self.lower_casing(text)
        text = self.remove_special_char(text)
        text = self.remove_space_in_between(text)
        text = self.remove_stopwords(text)
        text = self.review_lemmatize(text)
        text = self.text_join(text)
        text_vectorized = self.cv.transform([text]).toarray()
        return text_vectorized

# Example usage:
if __name__ == "__main__":
    text_preprocessor = TextPreprocessor()
    text = "Your input text here"
    processed_text = text_preprocessor.preprocess_text(text)
    print(processed_text)
