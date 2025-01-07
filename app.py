# render_template allows us to connect our HTML to our application
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from helperFunctions import *

app = Flask(__name__)
 # creating the Flask app and save it 

# model = load_model('seq2seq_translation_model.h5')
model = load_model('model2.h5')

# How it works?
# 
# The decorator first defines the relationship between the function and the route. 
# The function returns the landing page 
# and route shows the location where the landing page has to be displayed.
# 
#  the decorator
# ‘Routes’ modules are objects which configure the
#  webpages which receives inputs and displays the returned outputs
@app.route('/') # the home page
def home():
    return render_template('index.html') # returns the html template for my landing page

# the next decorator returns the predictions
#
# By default, the routes decorator only receives 
# input, ‘GET’ requests.
# In order to return the predicted words, 
# we have to define a new method 
# in the decorator route called ‘POST’
@app.route('/translate', methods=['POST'])
def translate():       
        data = request.get_json()  # Get JSON data sent in the request
       
        if not data or 'input_text' not in data:
            return jsonify({'error': 'Invalid request, input_text is required'}), 400
        
        input_text = data.get('input_text', '')  # Get the input_text field
        print(f"Received input: {input_text}") 
        if not input_text.strip():
            return jsonify({'error': 'No input provided'}), 400

        # Step 1: Clean the input sentence
        cleanText = clean_sentence(input_text)
        # Step 2 : Converting to sequences and padding them &
        clean_text = clean_sentence(input_text)
        # Step 3 : Get the translation
        translation = translate_sentence(model, clean_text)

        return jsonify({'translated_text': translation}), 200

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=8000, debug=True) # 0.0.0.0 so it runs on my local host