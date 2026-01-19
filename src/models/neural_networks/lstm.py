import torch
import torch.nn as nn


def check(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in {name}")
        if torch.isinf(param).any():
            print(f"Inf in {name}")


class LSTM(torch.nn.Module):
    """
    Creates sequential model consisting of LSTMs.
    """
    def __init__(
        self,
        timestamps: int = 3,
        features: int = 6,
        output_shape: int = 3,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        lr: float = 1e-5,
        dropout: float = 0.2,
        name: str | None = None,
        debug: bool = False
    ) -> None:
        """
        Initializes instance.

        Args:
            timestamps (int, optional): number of observations passed together. Defaults to 3.
            features (int, optional): number of features. Defaults to 6.
            output_shape (int, optional): output shape. Defaults to 3.
            hidden_layers (int, optional): number of LSTMs in model. Defaults to 32.
            hidden_units (int, optional): dimensionality of LSTMs outputs. Defaults to 32.
            lr (float, optional): learning rate. Defaults to 1e-5.
            dropout (float, optional): dropout. Defaults to 0.2.
            clipnorm (float | None, optional): gradient clipping. Defaults to None.
            name (_type_, optional): model name. Defaults to None.
        """
        super().__init__()

        self.timestamps = timestamps
        self.features = features
        self.hidden_units = hidden_units

        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=hidden_units,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=dropout if hidden_layers > 1 else 0
        )

        self.gru = nn.GRU(
            input_size=hidden_units,
            hidden_size=hidden_units,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=dropout if hidden_layers > 1 else 0
        )

        self.linear = nn.Linear(
            in_features=hidden_units,
            out_features=output_shape
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.debug = debug


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if x.dim() == 2:
            x = x.unsqueeze(0)

        lstm_out, _ = self.lstm(x)

        lstm_out_reshaped = lstm_out
        gru_out, _ = self.gru(lstm_out_reshaped)

        gru_last = gru_out[:, -1, :]

        output = self.linear(gru_last)

        if self.debug:
            if torch.isnan(output).any().item():
                print('NAN detected!!!!!!!', output)
                check(self)
                raise RuntimeError

        return output


    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)


    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()
