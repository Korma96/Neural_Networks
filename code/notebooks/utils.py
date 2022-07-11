import pandas as pd
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from scipy import spatial
import json

    
def parse_data(df: pd.DataFrame) -> tuple:
    questions = df['Question'].tolist()
    answers = df['Sentence'].tolist()
    labels = df['Label'].tolist()
    return questions, answers, [float(l) for l in labels]

def get_data_loader(model: SentenceTransformer, train_batch_size: int,
                    questions: list, answers: list, labels: list, shuffle: bool = True) -> DataLoader:
    examples = [InputExample(texts=[q, a], label=float(l)) 
                for q, a, l in zip(questions, answers, labels)]
    dataset = SentencesDataset(examples, model)
    return DataLoader(dataset, shuffle=shuffle, batch_size=train_batch_size)

def get_data_loader(model: SentenceTransformer, train_batch_size: int,
                    df: pd.DataFrame, shuffle: bool = True) -> DataLoader:
    questions, answers, labels = parse_data(df)
    examples = [InputExample(texts=[q, a], label=float(l)) 
                for q, a, l in zip(questions, answers, labels)]
    dataset = SentencesDataset(examples, model)
    return DataLoader(dataset, shuffle=shuffle, batch_size=train_batch_size)

def write_json(path: str, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def cosine_similarity(first_vector, second_vector):
  return 1 - spatial.distance.cosine(first_vector, second_vector)

def predict_similarity_cosine(model, questions, answers):
    questions_encoded, answers_encoded = model.encode(questions), model.encode(answers)
    predicted_similarity = []
    for question_encoded, answer_encoded in zip(questions_encoded, answers_encoded):
        predicted_similarity.append(cosine_similarity(question_encoded, answer_encoded))
    return predicted_similarity

def predict_similarity_dot(model, questions, answers):
    questions_encoded, answers_encoded = model.encode(questions), model.encode(answers)
    predicted_similarity = []
    for question_encoded, answer_encoded in zip(questions_encoded, answers_encoded):
        predicted_similarity.append(np.dot(question_encoded, answer_encoded))
    return predicted_similarity

def predict_similarity(model, questions, answers, similarity):
    if similarity == 'cosine':
        return predict_similarity_cosine(model, questions, answers)
    elif similarity == 'dot':
        return predict_similarity_dot(model, questions, answers)
    else:
        raise Exception('Not supported similarity')

def get_classification_report(labels, threshold, predicted_similarity, output_dict=True):
    predicted_labels = [1.0 if p > threshold else 0.0 for p in predicted_similarity]
    return classification_report(labels, predicted_labels, output_dict=output_dict, target_names=['0', '1'])

def find_optimum_values(model, questions, answers, labels, similarity='cosine'):
    predicted_similarity = predict_similarity(model, questions, answers, similarity)

    classification_reports = []
    thresholds = [l / 100.0 for l in list(range(0, 100, 1))]
    for threshold in thresholds:
        classification_reports.append(get_classification_report(labels, threshold, predicted_similarity))

    return get_maximum_point(thresholds, classification_reports, 'macro avg', 'f1-score')

def get_macro_f1_for_threshold(model, questions, answers, labels, threshold, similarity='cosine'):
    predicted_similarity = predict_similarity(model, questions, answers, similarity)
    return get_classification_report(labels, threshold, predicted_similarity)['macro avg']['f1-score']

def get_measure(classification_reports: list, first_key: str, second_key: str) -> list:
    return [class_report[first_key][second_key] for class_report in classification_reports]

def get_maximum_point(thresholds, classification_reports, first_key, second_key):
    x = thresholds
    y = get_measure(classification_reports, first_key, second_key)
    xmax = x[np.argmax(y)]
    ymax = max(y)
    return xmax, ymax
    
def annot_max(x,y, ha="right", va="top", ax=None):
    xmax = x[np.argmax(y)]
    ymax = max(y)
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)