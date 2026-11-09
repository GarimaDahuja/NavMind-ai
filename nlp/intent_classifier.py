"""
NavMind AI - Intent Classifier
Hybrid NLP module for classifying navigation command intents.

Approach:
1. Rule-based classifier (fast, no model needed)
2. DistilBERT fine-tuned classifier (when model is available)
"""
from __future__ import annotations

import re
import csv
import numpy as np
from pathlib import Path
from typing import Optional


# ─── Rule-Based Classifier ─────────────────────────────────────────────────

INTENT_RULES = {
    "NAVIGATE": [
        r"\b(go|navigate|move|drive|head|reach|take me|find|get to)\b.*(to|towards|toward)\b",
        r"\b(go to|navigate to|move to|head to|drive to|reach)\b",
        r"\b(destination|goal|target|waypoint)\b",
    ],
    "STOP": [
        r"\b(stop|halt|freeze|pause|abort|cancel|stand by|hold|wait|don't move)\b",
        r"\bemergency stop\b",
        r"\bstay (here|in place)\b",
    ],
    "SET_CONSTRAINT": [
        r"\b(avoid|careful|caution|safe|safety|don't hit|no collisions)\b",
        r"\b(watch out|be careful|navigate safely)\b",
    ],
    "OPTIMIZE": [
        r"\b(shortest|fastest|optimal|best|efficient|quickest|direct)\b.*(route|path|way)\b",
        r"\b(minimize|optimize)\b.*(distance|path|time)\b",
    ],
    "DIRECTION": [
        r"\b(turn|rotate|face|go|move)\b.*(left|right|forward|backward|back|ahead|straight)\b",
        r"\b(left|right|forward|backward|straight|ahead)\b",
    ],
    "QUERY": [
        r"\b(where|what|how|are you|tell me|report)\b.*\?",
        r"\b(status|position|location|distance|report)\b",
        r"\?",
    ],
}


class RuleBasedClassifier:
    """Fast rule-based intent classifier using regex patterns."""

    def __init__(self):
        self.compiled_rules = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in INTENT_RULES.items()
        }

    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify text into an intent.

        Returns:
            (intent_label, confidence)
        """
        scores = {}
        for intent, patterns in self.compiled_rules.items():
            score = sum(1 for p in patterns if p.search(text))
            scores[intent] = score

        if max(scores.values()) == 0:
            return "UNKNOWN", 0.3

        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] / 3.0, 1.0)
        return best_intent, confidence


# ─── DistilBERT Classifier ──────────────────────────────────────────────────

class DistilBERTClassifier:
    """
    Fine-tuned DistilBERT intent classifier.
    Falls back to rule-based if model not available.
    """

    LABELS = ["NAVIGATE", "STOP", "SET_CONSTRAINT", "OPTIMIZE", "DIRECTION", "QUERY"]
    MODEL_PATH = "models/intent_classifier"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.MODEL_PATH
        self.tokenizer = None
        self.model = None
        self._loaded = False
        self._fallback = RuleBasedClassifier()

    def load(self) -> bool:
        """Attempt to load the fine-tuned model."""
        try:
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification  # type: ignore
            import torch  # type: ignore
            path = Path(self.model_path)
            if not path.exists():
                return False
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(path))
            self.model = DistilBertForSequenceClassification.from_pretrained(str(path))
            self.model.eval()
            self._loaded = True
            return True
        except Exception:
            return False

    def train(self, data_path: str = "nlp/commands.csv", epochs: int = 5):
        """
        Fine-tune DistilBERT on the navigation command dataset.

        Args:
            data_path: Path to commands CSV
            epochs: Training epochs
        """
        try:
            from transformers import (  # type: ignore
                DistilBertTokenizerFast,
                DistilBertForSequenceClassification,
                Trainer,
                TrainingArguments,
            )
            import torch  # type: ignore
            from torch.utils.data import Dataset  # type: ignore

            # Load data
            commands, labels = [], []
            with open(data_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    commands.append(row["command"])
                    labels.append(self.LABELS.index(row["intent"]))

            # Tokenize
            tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            encodings = tokenizer(commands, truncation=True, padding=True, max_length=64)

            class NavDataset(Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels
                def __getitem__(self, idx):
                    item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item
                def __len__(self):
                    return len(self.labels)

            dataset = NavDataset(encodings, labels)

            # Model
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=len(self.LABELS),
            )

            # Training args
            training_args = TrainingArguments(
                output_dir=self.model_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=16,
                warmup_steps=20,
                weight_decay=0.01,
                logging_dir="results/intent_logs",
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="no",
                load_best_model_at_end=False,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )
            trainer.train()
            trainer.save_model(self.model_path)
            tokenizer.save_pretrained(self.model_path)
            print(f"[NLP] Intent classifier saved to {self.model_path}")
            self._loaded = True
            self.tokenizer = tokenizer
            self.model = model

        except Exception as e:
            print(f"[NLP] Training failed: {e}. Using rule-based fallback.")

    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify intent using fine-tuned model or fallback.

        Returns:
            (intent_label, confidence)
        """
        if not self._loaded:
            return self._fallback.classify(text)

        try:
            import torch  # type: ignore
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=64
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])
            return self.LABELS[predicted_idx], confidence
        except Exception:
            return self._fallback.classify(text)


# ─── Unified Interface ──────────────────────────────────────────────────────

class IntentClassifier:
    """
    Unified intent classifier interface.
    Uses DistilBERT when available, falls back to rule-based.
    """

    def __init__(self, use_bert: bool = True):
        self.rule_classifier = RuleBasedClassifier()
        self.bert_classifier = DistilBERTClassifier() if use_bert else None
        self._bert_loaded = False

        if use_bert:
            self._bert_loaded = self.bert_classifier.load()

    def classify(self, text: str) -> dict:
        """
        Classify a navigation command.

        Args:
            text: Input command text

        Returns:
            Dict with keys: intent, confidence, method
        """
        if self._bert_loaded and self.bert_classifier:
            intent, confidence = self.bert_classifier.classify(text)
            method = "distilbert"
        else:
            intent, confidence = self.rule_classifier.classify(text)
            method = "rule_based"

        return {
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "raw_text": text,
        }

    def train(self, data_path: str = "nlp/commands.csv"):
        """Train the DistilBERT classifier."""
        if self.bert_classifier:
            self.bert_classifier.train(data_path)
            self._bert_loaded = self.bert_classifier.load()

