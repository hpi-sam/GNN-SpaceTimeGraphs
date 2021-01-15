import argparse

# DEFAULT VALUES FOR TRAINING
BATCH_SIZE = 32
LEARNING_RATE = .01
WEIGHT_DECAY = 0.95
EPOCHS = 100

# DEFAULT VALUES FOR FILES AND FOLDERS
TRAIN_FILE = "./data/metr_la/train.npz"
TEST_FILE = "./data/metr_la/test.npz"
MODEL_SAVE_PATH = "./saved_models/"
ADJACENCY_PKL = "metr_la/adj_mx_la.pkl"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file", type=str, default=TRAIN_FILE, help="File containing the training data"
    )
    parser.add_argument(
        "--test_file", type=str, default=TEST_FILE, help="File containing the testing data"
    )
    parser.add_argument(
        '--lr', type=float, default=LEARNING_RATE, help="Learning rate for training"
    )
    parser.add_argument(
        '--weight_decay', type=int, default=WEIGHT_DECAY, help="Batch size for training"
    )
    parser.add_argument(
        '--batch_size', type=int, default=BATCH_SIZE, help="Batch size for training"
    )
    parser.add_argument(
        '--save_model', action='store_true', help=''
    )
    parser.add_argument(
        '--model_name', type=str, default='model_001', help="Prefix for the saved model inside `~/saved_models`"
    )
    parser.add_argument(
        '--n_epochs', type=int, default=EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        '--pickled_files', type=str, default=ADJACENCY_PKL, help="File containing the adjacency matrix"
    )
    parser.add_argument(
        '--gpu', action='store_true', help="Try to enforce the usage of cuda, but it will use CPU if it fails"
    )
    parser.add_argument(
        '--model', type=str, default="SLGCN", help="The name of the model that should be used"
    )
    parser.add_argument(
        '--forecast_horizon', type=int, default=1, help="number of steps to predict into the future"
    )
    return parser