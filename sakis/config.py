# the path that checkpoints will be saved to
checkpoints_path = "./checkpoints/"
# the path the vocabulary is saved
vocabulary_path = "./data/vocabulary_info.json"

# sequence length for the model's input (!!! changes will require discarding any checkpoints !!!)
sequence_length = 12
# the batch size for the model's training (!!! changes will require discarding any checkpoints !!!)
batch_size = 128
# the number of epochs when calling keras fit functions
epochs = 15
