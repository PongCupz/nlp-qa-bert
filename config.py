config = {
    "checkpoint" : "distilbert-base-uncased",
    # "checkpoint" : "bert-large-uncased",
    # "checkpoint" : "bert-large-uncased-whole-word-masking-finetuned-squad",
    "batch_size" : 2,
    "epochs" : 20,
    "seed_val": 42
}

# Loss function
loss_config = {
    "smoothing": False,
    "eps": 0.1,
}