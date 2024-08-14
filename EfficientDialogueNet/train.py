import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.transformer import SimpleLanguageModel
from data.dataset import get_synthetic_dataloaders
from config import ModelConfig, TrainingConfig

def train():
    # Load configuration
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Get synthetic dataloaders
    save_path = "/Users/san./Documents/GitHub/EfficientDialogueNet/EfficientDialogueNet/data/synthetic_dialogue_data.pkl"
    train_loader, val_loader = get_synthetic_dataloaders(
        batch_size=training_config.batch_size,
        num_samples=training_config.num_samples,
        max_seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        save_path=save_path,
        load_path=save_path
    )

    # Initialize model
    model = SimpleLanguageModel(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        d_ff=model_config.d_ff,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    # Training loop
    model.train()
    for epoch in range(training_config.num_epochs):
        total_loss = 0
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output.view(-1, model_config.vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / 100
                print(f"Epoch {epoch+1}/{training_config.num_epochs}, Batch {batch_idx+1}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                output = model(input_ids)
                loss = criterion(output.view(-1, model_config.vocab_size), target_ids.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{training_config.num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        model.train()

    # Save the model
    torch.save(model.state_dict(), "simple_language_model.pth")

if __name__ == "__main__":
    train()