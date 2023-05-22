from DL_PA3_final import train
import argparse
# default hyperparameters for the non-attention model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Seq2Seq model with attention')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding_size', type=int, default=32, help='Embedding size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size')
    parser.add_argument('--num_layers_encoder', type=int, default=3, help='Number of layers in the encoder')
    parser.add_argument('--num_layers_decoder', type=int, default=3, help='Number of layers in the decoder')
    parser.add_argument('--bidirectional', type=str, default='No', help='Use bidirectional encoder (Yes/No)')
    parser.add_argument('--cell_type', type=str, default='LSTM', help='Cell type (LSTM/GRU)')
    parser.add_argument('--teach_ratio', type=float, default=0.6, help='Teacher forcing ratio')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--attention', type=str, default='No', help='Use attention (Yes/No)')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    # defined it here because my train takes config_defaults by default to run

    config_defaults={
    'learn_rate': args.learn_rate,
    'embedding_size': args.embedding_size,
    'batch_size': args.batch_size,
    'hidden_size': args.hidden_size,
    'num_layers_encoder': args.num_layers_encoder,
    'num_layers_decoder': args.num_layers_decoder,
    'bidirectional': args.bidirectional,
    'cell_type': args.cell_type,
    'teach_ratio': args.teach_ratio,
    'dropout': args.dropout,
    'epochs': args.epochs,
    'attention': args.attention
    }
    # Pass the hyperparameters to the train function
    train(sweeps = False, test = True)

if __name__ == '__main__':
    main()
