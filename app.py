#import libraries
import emoji
import numpy as np
from flask import Flask, request, jsonify, render_template

#Initialize the flask App
from tensorflow import keras

app = Flask(__name__)
reconstructed_model = keras.models.load_model("./emoji_predictor_vanilla_emb")

emoji_dictionary = {"0": "\u2764\uFE0F",
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                   }

embeddings = {}
with open('vanilla_glove_100D_vocab50k.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        embeddings[word] = coeffs

def getOutputEmbeddings(X):
    embedding_matrix_output = np.zeros((1, 10, 100))

    for jx in range(len(X[0])):
        embedding_matrix_output[0][jx] = embeddings[X[0][jx].lower()]

    return embedding_matrix_output

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text = [x for x in request.form.values()]
    list_words = text[0].split()
    final_features = np.array(list_words)
    final_features = final_features.reshape((1, -1))
    # print(final_features.shape)
    # print(final_features)
    final_feature = getOutputEmbeddings(final_features)
    prediction = reconstructed_model.predict_classes(final_feature)
    # print(emoji.emojize(emoji_dictionary[str(prediction[0])]))
    return render_template('index.html', prediction_text='{} :{}'.format(text[0], emoji.emojize(emoji_dictionary[str(prediction[0])])))

if __name__ == "__main__":
    app.run(debug=True)