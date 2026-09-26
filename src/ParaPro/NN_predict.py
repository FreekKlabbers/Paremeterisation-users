from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from .model import MLPRegressorTorch

FEATURE_COLUMNS = ["eta", "R", "delta", "Cp", "P0", "sqrt"]


class NN_predictor:
    MODEL_FOLDER = Path(__file__).parent / "assets"
    HF_REPO_ID = "FreekKlabbers/plasma-probe"

    preprocessor: ColumnTransformer
    target_scaler: StandardScaler
    model: MLPRegressorTorch

    def __init__(self, revision: str | None = None):
        """
        Wrapper class for the Neural Network predictor. It loads the preprocessor, target scaler, and model from the specified Hugging Face repository.
        Use the `predict` method to make predictions on new data.

        Parameters
        ----------
        revision : str | None
            The specific revision of the model to load from the Hugging Face repository. If None, the latest revision is used.
        """

        if not self.MODEL_FOLDER.exists() or not any(self.MODEL_FOLDER.iterdir()):
            print(
                f"Model folder {self.MODEL_FOLDER} does not exist or is empty. Downloading model files..."
            )
            self.MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
            self.download(
                repo_id=self.HF_REPO_ID, out_dir=self.MODEL_FOLDER, revision=revision
            )

        self.model = MLPRegressorTorch.from_pretrained(self.MODEL_FOLDER)

        self.preprocessor, self.target_scaler = NN_predictor.load_preprocessor(
            f"{self.MODEL_FOLDER}/preprocessor.pkl",
            f"{self.MODEL_FOLDER}/target_scaler.pkl",
        )

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like
            The input features for which to make predictions.

        Returns
        -------
        array-like
            The predicted values after inverse transforming the scaled predictions.
        """
        X_features = NN_predictor.engineer_features(X)
        X_proc = self.preprocessor.transform(X_features)
        device = next(self.model.parameters()).device
        X_tensor = torch.as_tensor(X_proc, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor)
        y_pred = self.target_scaler.inverse_transform(
            y_pred_scaled.cpu().numpy().reshape(-1, 1)
        ).ravel()
        return np.exp(y_pred)  # Return the predictions in the original scale

    @staticmethod
    def load_preprocessor(
        preprocessor_path: str = "src/NNProbe/weights/preprocessor.pkl",
        target_scaler_path: str = "src/NNProbe/weights/target_scaler.pkl",
    ) -> tuple[ColumnTransformer, StandardScaler]:
        """
        Load the preprocessor and target scaler from disk.

        Parameters
        ----------
        preprocessor_path : str
            Path to load the preprocessor from.
        target_scaler_path : str
            Path to load the target scaler from.

        Returns
        -------
        tuple[ColumnTransformer, StandardScaler]
            A tuple containing the loaded preprocessor and target scaler.
        """
        res_preprocessor_path = NN_predictor._resolve_output_path(
            preprocessor_path, "preprocessor.pkl"
        )
        res_target_scaler_path = NN_predictor._resolve_output_path(
            target_scaler_path, "target_scaler.pkl"
        )

        return joblib.load(res_preprocessor_path), joblib.load(res_target_scaler_path)

    @staticmethod
    def _resolve_output_path(path: str, default_name: str) -> Path:
        resolved = Path(path)
        if resolved.suffix == "" or resolved.exists() and resolved.is_dir():
            return resolved / default_name
        return resolved

    @staticmethod
    def engineer_features(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Add the physics-derived features used by the trained model."""


        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(
                data, columns=["eta", "R", "delta", "Cp"]
            )  # Assuming the input is in the correct order

        features = data.copy()
        features["s_ld"] = 1.2 * (features["eta"] ** 0.75)
        features["P0"] = np.exp(-features["delta"] * features["s_ld"])
        features["sqrt"] = np.sqrt(features["eta"] / features["R"])
        return features[FEATURE_COLUMNS]

    @staticmethod
    def download(
        repo_id: str,
        out_dir: str | Path,
        revision: str | None = None,
        filename: str | None | Path = None,
    ):
        """
        Download the model weights and configuration from the Hugging Face repository.

        Parameters
        ----------
        repo_id : str
            The repository ID on Hugging Face Hub.
        out_dir : str
            The directory to download the files to.
        revision : str | None
            The specific revision of the model to download. If None, the latest revision is used.
        filename : str | None | Path, optional
            The specific file to download. If None, all files in the repository are downloaded.
        """
        if filename:
            files = [filename]
        else:
            files = list_repo_files(repo_id)  # grabs everything in the repo

        for f in files:
            path = hf_hub_download(
                repo_id=repo_id, filename=str(f), local_dir=out_dir, revision=revision
            )
            print(f"Downloaded {f} -> {path}")
