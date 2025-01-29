# Autoencoder in PyTorch

This repository contains a PyTorch script (autoencoder.py) for training a basic autoencoder on the MNIST dataset of handwritten digits.

## Getting Started

```bash
pip install -r requirements.txt
```

## Running the Script

The script can be run from the command line using the following options:

```
-h, --help: Show this help message and exit.
--loglevel {INFO,DEBUG}: Set the logging level (INFO or DEBUG). Defaults to INFO.
--num_epochs NUM_EPOCHS: Number of epochs to train the autoencoder. Defaults to 10.
--data_dir DATA_DIR: Directory containing the MNIST dataset (downloaded automatically if not provided).
--num_layers NUM_LAYERS: Number of hidden layers in the encoder and decoder (defaults to 2).
--lr LR: Learning rate for the optimizer (defaults to 1e-2)
--batch_size BATCH_SIZE: Batch size during training (defaults to 32).
```

## Example Usage:

```bash
python autoencoder.py --data_dir ./data --num_epochs 20 --lr 0.002
```

This command trains the autoencoder for 20 epochs with a learning rate of 0.002, storing the MNIST dataset in the ./data directory.

## Explanation of the Script

The `autoencoder.py` script implements the following functionalities:

- Loads the MNIST dataset using torchvision.
- Defines an autoencoder architecture with configurable numbers of layers.
- Implements a training loop with an optimizer and loss function.
- Logs training progress based on the specified logging level.
