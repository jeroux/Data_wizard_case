import pandas as pd
from datasets import Dataset, ClassLabel, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import evaluate
import numpy as np
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight

# 1. Load data with pandas
df = pd.read_csv("cleaned_dataset.csv", sep=";", encoding='utf-8')
df = df.rename(columns={"description": "text", "fraudulent": "label"})

# 2. Calculate class weights for imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label']),
    y=df['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights: {class_weights}")

# 3. Convert to HF Dataset with proper ClassLabel type
def create_dataset(df):
    features = {
        'text': df['text'].values,
        'label': ClassLabel(names=['NON_FRAUDULENT', 'FRAUDULENT']).str2int(df['label'].astype(str).values)
    }
    return Dataset.from_dict(features)

full_dataset = create_dataset(df)

# 4. Stratified split (manual implementation)
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    random_state=42,
    stratify=full_dataset['label']
)

dataset = DatasetDict({
    'train': full_dataset.select(train_idx),
    'test': full_dataset.select(test_idx)
})

# 5. Tokenization with shorter sequence length for speed
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,  # Reduced for faster training
        padding='max_length'
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 6. Fixed custom model with class weights handling
class WeightedClassificationModel(AutoModelForSequenceClassification):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove class_weights from kwargs before passing to super()
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(model.device)
            
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 7. Fixed model initialization
model = WeightedClassificationModel.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=2,
    id2label={0: "NON_FRAUDULENT", 1: "FRAUDULENT"},
    label2id={"NON_FRAUDULENT": 0, "FRAUDULENT": 1},
)

# 8. Faster training configuration
training_args = TrainingArguments(
    output_dir="./fast_results",
    per_device_train_batch_size=64,  # Larger batch size
    per_device_eval_batch_size=64,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=3e-5,              # Slightly higher learning rate
    num_train_epochs=2,              # Reduced epochs
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",      # Better for imbalanced data
    logging_dir='./logs',
    logging_steps=20,
    report_to="none",
    fp16=True,                       # Mixed precision for speed
    gradient_accumulation_steps=1    # No accumulation needed with larger batch
)

# 9. Metrics for imbalanced data
def compute_metrics(eval_pred):
    metrics = {
        'f1': evaluate.load("f1", average="binary"),
        'precision': evaluate.load("precision", average="binary"),
        'recall': evaluate.load("recall", average="binary"),
        'accuracy': evaluate.load("accuracy")
    }
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    results = {}
    for name, metric in metrics.items():
        if name == 'accuracy':
            results[name] = metric.compute(predictions=predictions, references=labels)['accuracy']
        else:
            results[name] = metric.compute(predictions=predictions, references=labels, pos_label=1)[name]
    
    return results

# 10. Trainer setup
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 11. Train
print("Starting optimized training...")
trainer.train()
print("Training complete!")

# Show class distribution
print("\nClass distribution:")
print(f"Train: {np.bincount(dataset['train']['label'])}")
print(f"Test: {np.bincount(dataset['test']['label'])}")