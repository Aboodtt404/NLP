import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from model import IntentClassifier
import numpy as np
from tqdm import tqdm
import json
import os


class IntentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

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
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class Trainer:
    def __init__(self, model: IntentClassifier, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              num_epochs: int = 10,
              learning_rate: float = 2e-5,
              warmup_steps: int = 1000,
              save_dir: str = '../models'):

        total_steps = len(train_dataloader) * num_epochs

        # Initialize optimizer with weight decay
        optimizer = AdamW(self.model.parameters(),
                          lr=learning_rate, weight_decay=0.01)

        # Initialize scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            train_steps = 0

            progress_bar = tqdm(
                train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                train_steps += 1

                # Update progress bar
                progress_bar.set_postfix(
                    {'training_loss': '{:.3f}'.format(loss.item())})

            avg_train_loss = total_train_loss / train_steps

            # Validation
            val_loss, val_accuracy = self.evaluate(val_dataloader)

            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')
            print(f'Validation accuracy: {val_accuracy:.4f}')

            # Early stopping with patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(self.model.state_dict(),
                           os.path.join(save_dir, 'best_model.pth'))
                print("Saved best model checkpoint")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        return (total_loss / len(dataloader),
                correct_predictions / total_predictions)


def prepare_data(data_path: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, Dict]:
    print(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Combine training data
    train_texts = []
    train_intents = []

    # Add search-related samples
    search_samples = [
        ("show me images of", "search_images"),
        ("find images of", "search_images"),
        ("search for images of", "search_images"),
        ("find news about", "search_news"),
        ("show me news about", "search_news"),
        ("search for news about", "search_news"),
        ("find videos of", "search_videos"),
        ("show me videos of", "search_videos"),
        ("search for videos about", "search_videos"),
    ]

    # Add search samples with common topics
    topics = ["cats", "dogs", "technology", "science", "sports",
              "music", "food", "travel", "health", "education"]
    for prefix, intent in search_samples:
        for topic in topics:
            train_texts.append(f"{prefix} {topic}")
            train_intents.append(intent)

    # Add original training data
    for item in data['train']:
        if isinstance(item, dict):
            text = item.get('text', '')
            intent = item.get('intent', '')
        else:
            text, intent = item[0], item[1]

        if intent != 'oos':  # Skip out-of-scope samples
            train_texts.append(text)
            train_intents.append(intent)

    # Prepare validation data
    val_texts = []
    val_intents = []
    for item in data['val']:
        if isinstance(item, dict):
            text = item.get('text', '')
            intent = item.get('intent', '')
        else:
            text, intent = item[0], item[1]

        if intent != 'oos':
            val_texts.append(text)
            val_intents.append(intent)

    print(
        f"Loaded {len(train_texts)} training samples and {len(val_texts)} validation samples")

    # Create intent mapping
    unique_intents = sorted(list(set(train_intents)))
    intent_to_idx = {intent: idx for idx, intent in enumerate(unique_intents)}

    print(f"Found {len(unique_intents)} unique intents")

    # Convert intents to indices
    train_labels = [intent_to_idx[intent] for intent in train_intents]
    val_labels = [intent_to_idx[intent] for intent in val_intents]

    # Initialize model and datasets
    model = IntentClassifier(num_classes=len(unique_intents))
    train_dataset = IntentDataset(train_texts, train_labels, model.tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, model.tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, intent_to_idx


if __name__ == "__main__":
    data_path = '../data/data_full.json'

    print("Loading and preparing data...")
    train_dataloader, val_dataloader, intent_to_idx = prepare_data(data_path)
    print(f"Number of intents: {len(intent_to_idx)}")

    num_classes = len(intent_to_idx)
    print(f"Initializing model with {num_classes} classes...")
    model = IntentClassifier(num_classes=num_classes)
    trainer = Trainer(model)

    if not os.path.exists('../models'):
        os.makedirs('../models')

    with open('../models/intent_mapping.json', 'w') as f:
        json.dump(intent_to_idx, f)

    print("Starting training...")
    trainer.train(train_dataloader, val_dataloader)
