import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from pathlib import Path
from model import NeuroGuard
from preprocess import DataPreprocesor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(name)s = %(levelname)s - %(message)s'
)


class Training:
    def __init__(
        self,
        data_dir: str = "./data/processed",
        artifacts_dir: str = "./artifacts",
        batch_size: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 10,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.val_size = val_size
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = StandardScaler()
        self.model = None

    def _load_data(self):
        x_path = self.data_dir / "X_train.npy"
        y_path = self.data_dir / "y_train.npy"

        if not x_path.exists() or not y_path.exists():
            logger.info("We dont have files for training model x/y paths. Starting preprocessing")
            try:
                DataPreprocesor.preprocesor('KDDTrain+.txt', 'processed')
            except Exception as e:
                raise FileNotFoundError(f"Missing preprocessed data. Expected files: {x_path} and {y_path}")
        
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Failed with generating data, looking for files {x_path}/{y_path}"
            )
        
        x_raw = np.load(x_path)
        y_raw = np.load(y_path)
        return x_raw, y_raw

    def _prepare_dataloaders(self):
        x_raw, y_raw = self._load_data()

        x_train, x_val, y_train, y_val = train_test_split(
            x_raw,
            y_raw,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_raw,
        )

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_val_scaled = self.scaler.transform(x_val)

        train_dataset = TensorDataset(
            torch.from_numpy(x_train_scaled).float(),
            torch.from_numpy(y_train).float().reshape(-1, 1),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(x_val_scaled).float(),
            torch.from_numpy(y_val).float().reshape(-1, 1),
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, x_train_scaled.shape[1]

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    def _validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        running_val_loss = 0.0

        criterion = nn.BCELoss()

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item()

                preds = (outputs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())

        metrics = self._calculate_metrics(np.array(all_targets), np.array(all_preds))
        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        return avg_val_loss, metrics

    def train(self):
        train_loader, val_loader, input_dim = self._prepare_dataloaders()

        self.model = NeuroGuard(input_dim=input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logger.info("Starting training on device: %s", self.device)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / max(len(train_loader), 1)
            avg_val_loss, val_metrics = self._validate(val_loader)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | acc=%.4f | prec=%.4f | rec=%.4f | f1=%.4f",
                epoch + 1,
                self.epochs,
                avg_train_loss,
                avg_val_loss,
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
            )

        self._save_artifacts()

    def _save_artifacts(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.artifacts_dir / "neuroguard_model.pt"
        scaler_path = self.artifacts_dir / "scaler.npz"

        torch.save(self.model.state_dict(), model_path)
        np.savez(scaler_path, mean=self.scaler.mean_, scale=self.scaler.scale_)

        logger.info("Saved model to %s", model_path)
        logger.info("Saved scaler params to %s", scaler_path)


if __name__ == "__main__":
    trainer = Training()
    trainer.train()
