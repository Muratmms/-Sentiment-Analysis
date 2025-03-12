import tensorflow as tf
import numpy as np
from django.shortcuts import render
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.stem import WordNetLemmatizer
# Modeli yükle
model = tf.keras.models.load_model('sentiment_analysis/reviews_model.h5')

# Kullanıcı yorumlarını temizleme
nltk.download('stopwords')
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    # Lemmatization işlemi
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

def predict_sentiment(request):
    if request.method == 'POST':
        comment = request.POST.get('comment')  # Formdan gelen yorum
        clean_comment = clean_text(comment)

        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts([clean_comment])  # Tek bir yorum için tokenization yapıyoruz
        sequence = tokenizer.texts_to_sequences([clean_comment])
        padded = pad_sequences(sequence, maxlen=100)

        # Model ile tahmin yap
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction, axis=1)  # 0: Negatif, 1: Pozitif

        if predicted_class == 1:
            result = "Pozitif"
        else:
            result = "Negatif"

        return render(request, 'sentiment_analysis/result.html', {'result': result, 'comment': comment})
    return render(request, 'sentiment_analysis/index.html')

