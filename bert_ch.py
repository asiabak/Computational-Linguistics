import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class GeoDataset(Dataset):
    def __init__(self, texts, coords=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.coords = coords
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.coords is not None:
            item['coords'] = torch.tensor(self.coords[idx], dtype=torch.float)

        return item

class GeoBERT(nn.Module):
    def __init__(self, bert_model="bert-base-uncased"):
        super(GeoBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # 2 outputs for latitude and longitude
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.regressor(self.dropout(pooled_output))

def load_data(filename):
    """Load data from tab-separated files without headers"""
    try:
        # Load data without headers, using numeric column names
        data = pd.read_csv(filename, sep='\t', header=None)
        # Take first two columns as coordinates and third as text
        coords = data[[0, 1]].values  # latitude and longitude
        texts = data[2].values  # text
        return coords, texts
    except pd.errors.EmptyDataError:
        print(f"Error: {filename} is empty")
        return None, None
    except Exception as e:
        # For test_blind.txt which might only contain text
        try:
            data = pd.read_csv(filename, sep='\t', header=None)
            return None, data[0].values  # return only texts
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None

def train_model(model, train_loader, dev_loader, device, num_epochs=5, patience=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coords = batch['coords'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation on dev set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                coords = batch['coords'].to(device)

                outputs = model(input_ids, attention_mask)
                val_loss += criterion(outputs, coords).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(dev_loader)

        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')

        # Check if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print("Early stopping triggered. Training halted.")
            break


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set with known coordinates"""
    model.eval()
    all_preds = []
    all_coords = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coords = batch['coords'].to(device)

            outputs = model(input_ids, attention_mask)
            all_preds.extend(outputs.cpu().numpy())
            all_coords.extend(coords.cpu().numpy())

    mse = mean_squared_error(all_coords, all_preds)
    print(f'Test MSE: {mse:.4f}')
    return mse

def predict_blind(model, test_loader, device):
    """Generate predictions for blind test set"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)

def main():
    # Load datasets
    train_coords, train_texts = load_data('train.txt')
    dev_coords, dev_texts = load_data('dev.txt')
    test_gold_coords, test_gold_texts = load_data('test_gold.txt')
    _, test_blind_texts = load_data('test_blind.txt')

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = GeoDataset(train_texts, train_coords, tokenizer)
    dev_dataset = GeoDataset(dev_texts, dev_coords, tokenizer)
    test_gold_dataset = GeoDataset(test_gold_texts, test_gold_coords, tokenizer)
    test_blind_dataset = GeoDataset(test_blind_texts, tokenizer=tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16)
    test_gold_loader = DataLoader(test_gold_dataset, batch_size=16)
    test_blind_loader = DataLoader(test_blind_dataset, batch_size=16)

    # Initialize model and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GeoBERT().to(device)

    # Train the model
    train_model(model, train_loader, dev_loader, device)

    # Evaluate on test set with known coordinates
    test_mse = evaluate_model(model, test_gold_loader, device)

    # Generate predictions for blind test set
    blind_predictions = predict_blind(model, test_blind_loader, device)

    # Save blind test predictions
    np.savetxt('blind_test_predictions.txt', blind_predictions, delimiter='\t', fmt='%.6f')

if __name__ == "__main__":
    main()