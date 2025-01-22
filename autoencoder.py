import argparse
import logging
import torch
import torchvision

from torch import nn


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loglevel", type=str, choices=["INFO", "DEBUG"],
                        default="INFO", help="logging level")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--data_dir", type=str, required=True, help="directory to store MNIST")
    parser.add_argument("--num_layers", type=int, required=True, help="number of layers in encoder/decoder")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size during training")
    return parser


class AutoEncoder(nn.Module):

    def __init__(self, dim, num_layers=3):
        super().__init__()

        dim_offset = round(dim/num_layers)

        encoder_layers = []

        for i in range(num_layers-1):
            input_dim = dim - i*dim_offset
            output_dim = dim - (i+1)*dim_offset
            encoder_layers.append(
                nn.Linear(input_dim, output_dim)
            )
            if i < num_layers - 2:
                encoder_layers.append(nn.ReLU())

        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []

        for i in range(num_layers-2, -1, -1):
            input_dim = dim - (i+1)*dim_offset
            output_dim = dim - (i)*dim_offset
            decoder_layers.append(
                nn.Linear(input_dim, output_dim)
            )
            if i > 0:
                decoder_layers.append(nn.ReLU())
            else:
                decoder_layers.append(nn.Sigmoid())

        self._decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        x_flattened = nn.Flatten()(x)
        encoding = self._encoder(x_flattened)
        decoding = self._decoder(encoding)
        return decoding.view(x.size())


    def predict(self, x):
        with torch.no_grad():
            self.eval()
            return self.forward(x)


def train(dataloader, model, loss_fn, optimizer, num_epochs=10, log_iterations=100):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(1, num_epochs+1):
        logging.info(f"Epoch: {epoch}")
        for batch, (X, _) in (enumerate(dataloader)):
            X = X.to(DEVICE) 

            optimizer.zero_grad()
            # Compute reconstruction error
            decoding = model(X)
            loss = loss_fn(decoding, X)

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch % log_iterations == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                logging.debug(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main():
    parser = setup_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=args.loglevel,
        format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
    )

    dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    sample = dataset[0]
    image, _ = sample
    dim = list(image.flatten().size())[0]
    model = AutoEncoder(
        dim=dim,
        num_layers=args.num_layers
    ).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr
    )
    train(dataloader, model, loss_fn, optimizer, args.num_epochs)


if __name__ == "__main__":
    main()
