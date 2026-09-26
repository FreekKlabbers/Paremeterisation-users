import inspect
import json
from pathlib import Path

import torch
from torch import nn


class MLPRegressorTorch(nn.Module):
    """
    Feedforward network with ReLU activations, and a single linear output unit for regression.
    """

    WEIGHTS_FILENAME = "model.pt"
    LEGACY_WEIGHTS_FILENAME = "mlp_regressor_model.pth"
    CONFIG_FILENAME = "config.json"

    def __init__(self, input_dim: int, hidden_layer_sizes=(128, 64, 32, 16)):
        """
        Initializes the MLPRegressorTorch model.
        """
        super().__init__()
        self._capture_config(locals())  # capture configuration for Hugging Face

        layers = []
        prev_dim = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def _capture_config(self, local_vars: dict):
        """
        Wraps the local variables of the __init__ method to capture the model configuration for saving/loading.
        """
        sig = inspect.signature(self.__class__.__init__)
        param_names = [p for p in sig.parameters if p != "self"]
        self.config = {k: local_vars[k] for k in param_names}

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        map_location: str | None = None,
        **override_kwargs,
    ):
        """
        Load model weights and configuration from a Hugging Face repository.

        Parameters
        ----------
        repo_id : str
            The repository ID on Hugging Face Hub.
        revision : str, optional
            The specific revision (branch, tag, or commit) to load. Defaults to the latest
        map_location : str, optional
            The device to map the model to. If None, defaults to 'cpu'.
        override_kwargs : dict
            Any additional keyword arguments to override the model's configuration.

        Returns
        -------
        MLPRegressorTorch
            An instance of the MLPRegressorTorch model with loaded weights and configuration.
        """
        if model_path.is_dir():
            weights_path = model_path / cls.WEIGHTS_FILENAME
            config_path = model_path / cls.CONFIG_FILENAME
        else:
            raise FileNotFoundError(
                f"Local directory {model_path} does not exist. Please provide a valid local path."
            )

        with open(config_path) as f:
            config = json.load(f)
        config.update(override_kwargs)

        # hidden_layer_sizes round-trips through JSON as a list; restore tuple.
        for k, v in config.items():
            if isinstance(v, list):
                config[k] = tuple(v)

        model = cls(**config)
        state_dict = torch.load(weights_path, map_location=map_location or "cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
