from sklearn.metrics import f1_score
from tqdm import tqdm


labeled_dataset = [
    {"document": "Hugging Face is a company that specializes in Natural Language Processing models.", "question": "What is Natural language", "answer": "Natural Language Processing models"},
]

def evaluate_model(model, dataset):
    predicted_answers = []
    ground_truth_answers = []

    for data_point in tqdm(dataset, desc="Evaluating"):
        document = data_point["document"]
        question = data_point["question"]
        ground_truth_answer = data_point["answer"]

        predicted_answer = extract_and_respond(document, question)
        predicted_answers.append(predicted_answer)
        ground_truth_answers.append(ground_truth_answer)
    f1 = f1_score(ground_truth_answers, predicted_answers, average='micro')

    return f1

f1_score_result = evaluate_model(model, labeled_dataset)
print("F1 Score:", f1_score_result)