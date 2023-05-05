config = {
    "checkpoint" : "bert-base-uncased",
    "batch_size" : 2,
    "epochs" : 20,
    "seed_val": 42,
    "calculate_score": False
}

# Loss function
loss_config = {
    "smoothing": False,
    "eps": 0.1,
}