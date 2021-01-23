import configargparse

# DEFAULT VALUES FOR TRAINING
BATCH_SIZE = 32
LEARNING_RATE = .01
WEIGHT_DECAY = 0.95
EPOCHS = 100

# DEFAULT VALUES FOR FILES AND FOLDERS
CONFIG_FILEPATH = "../configs/config.yml"
TRAIN_FILE = "./data/metr_la/train.npz"
TEST_FILE = "./data/metr_la/test.npz"
VAL_FILE = "./data/metr_la/val.npz"
MODEL_SAVE_PATH = "./saved_models/"
ADJACENCY_PKL = "metr_la/adj_mx_la.pkl"


def parse_arguments():
    """
    Parses arguments passed to the command line.
    :return: Argument Parser containing all arguments passed to the terminal.
    :rtype: argparse.ArgumentParser
    """
    parser = configargparse.ArgumentParser(
        default_config_files=[CONFIG_FILEPATH],
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument(
        '-c', '--my-config', is_config_file=True, help="Use config file to set arguments."
                                                       "You can add new to override the"
                                                       "the config file"
    )
    parser.add_argument(
        "--train_file", type=str, default=TRAIN_FILE, help="File containing the training data"
    )
    parser.add_argument(
        "--test_file", type=str, default=TEST_FILE, help="File containing the testing data"
    )
    parser.add_argument(
        "--val_file", type=str, default=VAL_FILE, help="File containing the validation data"
    )
    parser.add_argument(
        '--toy_data', action='store_true', help="Uses the `--train_file` data with all nodes but only 2.5%"
                                                "timestamps. Use it for debugging purposes"
    )
    parser.add_argument(
        '--lr', type=float, default=LEARNING_RATE, help="Learning rate for training"
    )
    parser.add_argument(
        '--weight_decay', type=float, default=WEIGHT_DECAY, help="Batch size for training"
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
        '--gpu', action='store_true', help="Try to enforce CUDA usage, but it will use CPU if it fails"
    )
    parser.add_argument(
        '--model', type=str, default="SLGCN", help="The name of the model that should be used"
    )
    parser.add_argument(
        '--forecast_horizon', type=int, default=1, help="number of steps to predict into the future"
    )
    parser.add_argument(
        '--n_hid', type=int, default=100, help="number of hidden nodes"
    )
    parser.add_argument(
        '--nclass', type=int, default=1, help="number of classes to output. Put 1 for regression problem."
    )
    parser.add_argument(
        '--num_nodes', type=int, default=207, help="number of nodes"
    )
    parser.add_argument(
        '--num_features', type=int, default=2, help="number of features for each node"
    )
    parser.add_argument(
        '--nhid_multipliers', type=int, nargs='+', help="multiplication factors for each hidden layer"
    )
    parser.add_argument(
        '--k', type=int, default=8, help="number of Chebyhev polynomials for spectral convolution"
    )
    parser.add_argument(
        '--seq_len', type=int, default=12, help="number of time-steps to use for a prediction"
    )
    parser.add_argument(
        '--hidden_state_size', type=int, default=264, help="size of hidden state for RNN"
    )
    parser.add_argument(
        '--num_units', type=int, default=1000, help="number of units per layer (RNN)"
    )
    return parser


if __name__ == "__main__":
    parser = parse_arguments()
    options = parser.parse_args()
    print(options)
    # print(parser.format_help())
    print("----------")
    print(parser.format_values())
