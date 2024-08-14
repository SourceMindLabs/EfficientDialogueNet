import torch
from torch.utils.data import Dataset
import random
import os
import pickle

class SyntheticDialogueDataset(Dataset):
    def __init__(self, num_samples=10000, max_seq_len=100, vocab_size=5000, min_seq_len=10):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.vocab_size = vocab_size
        self.data = self.generate_synthetic_data()

    def generate_synthetic_data(self):
        data = []
        for _ in range(self.num_samples):
            input_len = random.randint(self.min_seq_len, self.max_seq_len - 1)
            max_target_len = max(self.min_seq_len, self.max_seq_len - input_len)
            target_len = random.randint(self.min_seq_len, max_target_len)
            
            input_seq = torch.randint(0, self.vocab_size, (input_len,))
            target_seq = torch.randint(0, self.vocab_size, (target_len,))
            
            data.append((input_seq, target_seq))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        input_seq, target_seq = self.data[idx]
        
        input_padded = torch.zeros(self.max_seq_len, dtype=torch.long)
        target_padded = torch.zeros(self.max_seq_len, dtype=torch.long)

        input_len = min(len(input_seq), self.max_seq_len)
        target_len = min(len(target_seq), self.max_seq_len)

        input_padded[:input_len] = input_seq[:input_len]
        target_padded[:target_len] = target_seq[:target_len]
        
        return input_padded, target_padded

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f)

    @classmethod
    def load(cls, filepath, num_samples, max_seq_len, vocab_size, min_seq_len=10):
        dataset = cls(num_samples, max_seq_len, vocab_size, min_seq_len)
        with open(filepath, 'rb') as f:
            dataset.data = pickle.load(f)
        return dataset

def get_synthetic_dataloaders(batch_size=64, num_samples=10000, max_seq_len=100, vocab_size=5000, min_seq_len=10, save_path=None, load_path=None):
    if load_path and os.path.exists(load_path):
        print(f"Loading dataset from {load_path}")
        dataset = SyntheticDialogueDataset.load(load_path, num_samples, max_seq_len, vocab_size, min_seq_len)
    else:
        print("Generating new synthetic dataset")
        dataset = SyntheticDialogueDataset(num_samples, max_seq_len, vocab_size, min_seq_len)
        if save_path:
            print(f"Saving dataset to {save_path}")
            dataset.save(save_path)

    actual_size = len(dataset)
    print(f"Dataset size: {actual_size}")

    train_size = int(0.8 * actual_size)
    val_size = actual_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    save_path = "synthetic_dialogue_data.pkl"
    
    train_loader, val_loader = get_synthetic_dataloaders(
        batch_size=64,
        num_samples=10000,
        max_seq_len=100,
        vocab_size=5000,
        min_seq_len=10,
        save_path=save_path,
        load_path=save_path
    )
    
    for batch_input, batch_target in train_loader:
        print("Input shape:", batch_input.shape)
        print("Target shape:", batch_target.shape)
        print("Sample input:", batch_input[0])
        print("Sample target:", batch_target[0])
        break