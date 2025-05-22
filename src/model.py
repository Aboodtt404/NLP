import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Tuple


class IntentClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained_model: str = "bert-base-uncased"):
        super(IntentClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        # Improved feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Extract features
        features = self.feature_extractor(pooled_output)

        # Classify
        logits = self.classifier(features)

        return logits

    def prepare_input(self, text: str) -> Dict[str, torch.Tensor]:
        # Tokenize with improved settings
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def predict(self, text: str) -> Tuple[int, float]:
        self.eval()
        inputs = self.prepare_input(text)

        with torch.no_grad():
            logits = self(inputs['input_ids'], inputs['attention_mask'])
            # Convert log_softmax back to probabilities
            probabilities = torch.exp(logits)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence


def load_model(model_path: str, num_classes: int) -> IntentClassifier:
    model = IntentClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
