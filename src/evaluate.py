import torch
from model import IntentClassifier, load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os


class ModelEvaluator:
    def __init__(self, model: IntentClassifier, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict_batch(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        predictions = []
        confidences = []

        for text in texts:
            pred_class, confidence = self.model.predict(text)
            predictions.append(pred_class)
            confidences.append(confidence)

        return predictions, confidences

    def evaluate_dataset(self,
                         texts: List[str],
                         true_labels: List[int],
                         label_names: List[str],
                         output_dir: str = '../models/evaluation'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        predictions, confidences = self.predict_batch(texts)

        report = classification_report(true_labels, predictions,
                                       target_names=label_names,
                                       output_dict=True)

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))

        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names,
                    yticklabels=label_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        accuracy = sum(p == t for p, t in zip(
            predictions, true_labels)) / len(true_labels)
        avg_confidence = sum(confidences) / len(confidences)

        metrics = {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        return metrics


def load_test_data(data_path: str) -> Tuple[List[str], List[int], List[str]]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('../models/intent_mapping.json', 'r') as f:
        intent_to_idx = json.load(f)

    test_data = data['test']

    test_texts = []
    test_intents = []
    for item in test_data:
        if isinstance(item, dict):
            text = item.get('text', '')
            intent = item.get('intent', '')
        else:
            text, intent = item[0], item[1]

        if intent != 'oos':
            test_texts.append(text)
            test_intents.append(intent)

    test_labels = [intent_to_idx[intent] for intent in test_intents]

    idx_to_intent = {idx: intent for intent, idx in intent_to_idx.items()}
    label_names = [idx_to_intent[i] for i in range(len(intent_to_idx))]

    print(f"Loaded {len(test_texts)} test samples")
    return test_texts, test_labels, label_names


if __name__ == "__main__":
    print("Loading model and intent mapping...")
    with open('../models/intent_mapping.json', 'r') as f:
        intent_to_idx = json.load(f)
    num_classes = len(intent_to_idx)
    print(f"Model has {num_classes} intent classes")

    model = load_model('../models/best_model.pth', num_classes)
    evaluator = ModelEvaluator(model)

    print("\nLoading test data...")
    test_texts, test_labels, label_names = load_test_data(
        '../data/data_full.json')

    print("\nEvaluating model...")
    metrics = evaluator.evaluate_dataset(test_texts, test_labels, label_names)

    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average Confidence: {metrics['average_confidence']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nDetailed results have been saved to the 'models/evaluation' directory.")
