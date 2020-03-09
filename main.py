from nltk.stem.lancaster import LancasterStemmer
import nltk
import numpy as np
import tensorflow
import random
import json
import pickle
import tflearn
import os

from flask import Flask, render_template, request

nltk.download('punkt')
# to stem the words
stemmer = LancasterStemmer()

###############################
app = Flask(__name__)
@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    user_input = request.form['user_input']
    # chat(user_input)
    bot_response = chat(user_input)
    return render_template('index.html', user_input=user_input, bot_response=bot_response)

#################################


# read json file
with open('C:/Users/Kuldeep/Documents/GitHub/desktop-tutorial/intents.json') as file:
    data = json.load(file)

# print(data["intents"])
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []

    # to know what intent/tag it is a part of
    docs_y = []

    # stemming: to get the root of the word
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # tokenize : get all the words in a pattern
            wrds = nltk.word_tokenize(pattern)
            # returns a list of all the tokenized words
            words.extend(wrds)

            # add pattern of words to the docs
            docs_x.append(wrds)

            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    # stem words and remove duplicate
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # creating one hot encoding (bag of words)
    training = []
    output = []

    # one hot encoding starts from 0 to the length of the intents
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # changing into arrays so that we can feed them into our model
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# reseting the previous setting
tensorflow.reset_default_graph()

# the model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# tain our model
model = tflearn.DNN(net)


if os.path.exists("model.tflearn.meta"):
    model.load("model.tflearn")
    print("Model exists")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    print("Model does not exists, so trained a new one")

# try:
#     model.load("model.tflearn")
# except:
#     model.fit(training, output, n_epoch=100, batch_size=8, show_metric=True)
#     model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# takes in a sentence and gives back a response


def chat(user_input):
    bot_response = ""
    try:
        print("Start talking with the bot !! Press q to quit ")
        print("You :"+user_input)
        while True:
            #inp=input("You: ")
            if user_input.lower() == "q":
                print("You quit the chat")
                return "You quit the chat"
                #return render_template('quit.html')
                # break

            results = model.predict([bag_of_words(user_input, words)])
            arr_result = results[0]
            print(arr_result)

            # gives index of the greatest number
            results_index = np.argmax(results)
            print('The result_index is '+str(results_index))

            # gives the relevant tag
            tag = labels[results_index]
            print('The tag is ' + tag)
            print('labels are'+str(labels))

            if arr_result[results_index] > 0.6:
                print('arr_result[results_index]>0.6 ' +
                      str(arr_result[results_index]))
                # get a random response from the json file
                print("above for loop")
                for tg in data["intents"]:
                    print(tg)
                    if tg['tag'] == tag:
                        response = tg['responses']
                        print('the array of response is' + str(response))
                        rand_response = random.choice(response)

                        bot_response = str(rand_response)
                        print("Bot: "+bot_response)
                        #return render_template('index.html', user_input=user_input, bot_response=bot_response)
                        return bot_response
                    

            else:
                bot_response = "Sorry, I do not understand"
                print("Bot: "+bot_response)
                #return render_template('index.html', user_input=user_input, bot_response=bot_response)
                return bot_response

    except TypeError:
        print("A TypeError exception occurred")
        #return render_template('quit.html')
        return bot_response


if __name__ == '__main__':
    app.run(debug=True, port=5002)
