import logging
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from flask import Flask, render_template, request

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the SST-5 dataset
logger.info("Loading SST-5 dataset...")
sst5_dataset = load_dataset("sst")
logger.info("SST-5 dataset loaded successfully.")

# Split the dataset into training, validation, and test sets
train_dataset = sst5_dataset["train"]
val_dataset = sst5_dataset["validation"]
test_dataset = sst5_dataset["test"]

# Subset the datasets to 50% of their original size
subset_size = min(len(train_dataset),len(val_dataset),len(test_dataset))//2
train_dataset = train_dataset.select(range(subset_size))
val_dataset = val_dataset.select(range(subset_size))
test_dataset = test_dataset.select(range(subset_size))

# Load pre-trained BERT model and tokenizer
logger.info("Loading pre-trained BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=5)
logger.info("BERT model and tokenizer loaded successfully.")

# Define dataset and data loaders
logger.info("Creating data loaders...")


class SST5Dataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        encoding = self.tokenizer(
            sample["sentence"], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sample["label"], dtype=torch.long)
        }


train_loader = DataLoader(SST5Dataset(
    train_dataset, tokenizer), batch_size=32, shuffle=True)
logger.info("Data loaders created successfully.")

# Training loop
logger.info("Starting training loop...")
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch+1}...")
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"Processing batch {batch_idx+1}...")
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    logger.info(
        f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(train_loader)}")

# Map predicted labels to human-readable sentiment categories
label_map = {0: "Strongly Negative", 1: "Negative",
             2: "Neutral", 3: "Positive", 4: "Strongly Positive"}

# Flask routes


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            text = request.form['text']
            logger.info(f"Received text for prediction: {text}")
            # Tokenize input text
            encoding = tokenizer(
                text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            # Perform sentiment analysis
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
            predicted_sentiment = label_map[predicted_label]
            logger.info(f"Predicted sentiment: {predicted_sentiment}")
            return render_template('index.html', prediction=predicted_sentiment, text=text)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            logger.error(f"Error occurred: {error_message}")
            return render_template('index.html', prediction="Error", text="", error=error_message)


if __name__ == '__main__':
    app.run(debug=True)