class ModelConfig:
    vocab_size = 5000
    d_model = 256
    n_heads = 4
    n_layers = 4
    d_ff = 1024
    max_seq_len = 100
    dropout = 0.1

class TrainingConfig:
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-4
    num_samples = 10000