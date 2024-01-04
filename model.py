from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

document = " My name is Sudhin"
user_query = "What is the Mentioned name"
def respond_to_query(document, user_query):
    result = qa_pipeline(context=document, question=user_query)
    answer = result['answer']
    return f"User Query: {user_query}\nAnswer: {answer}"

response = respond_to_query(document, user_query)
print(response)