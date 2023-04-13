import pickle
import nltk
import numpy

ignore_list = [".", ",", "?", "/", "'s", "'m"]
stop_words = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
x = open("chtbot_vect.pkl", "rb")
vectorizer = pickle.load(x)
x.close()
x = open("chtbot_mdl.pkl", "rb")
model = pickle.load(x)
x.close()

while True:
    print("Hi!! \nHow can I help you?")
    inp = input()
    if inp == "Exit" or inp == "Leave":
        break
    tokens = nltk.word_tokenize(inp)
    tokens = [token for token in tokens if token not in ignore_list]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    sent = [" ".join(tokens)]
    sent = numpy.array(sent)

    ques = vectorizer.transform(sent)
    ques = ques.toarray()

    fin = model.predict(ques)
    print("Here are some links to", fin[0], " help you...")
    