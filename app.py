from flask import Flask, render_template, request

app = Flask(__name__)
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

def extract_and_respond(document, question):
    inputs = tokenizer(question, document, return_tensors='pt')
    outputs = model(**inputs)
    start_idx = torch.argmax(outputs['start_logits'])
    end_idx = torch.argmax(outputs['end_logits']) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx]))
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    if request.method == 'POST':
        document = request.form['document']
        question = request.form['question']
        response = extract_and_respond(document, question)
        return render_template('index.html', document=document, question=question, response=response)

if __name__ == '__main__':
    app.run(debug=True)
