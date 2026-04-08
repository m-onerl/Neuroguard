import numpy as np
import pandas as pd   
import torch
import logging

   
from src.model.preprocess import columns, attack_map
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from .init_model import NeuroGuard

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s = %(levelname)s - %(message)s'
)

CLASS_NAMES = {
    0: "Normal",
    1: "DoS",
    2: "Probe",
    3: "R2L",
    4: "U2R",
}


class Predictor:
    def __init__(
        self,
        model_path: str = "./artifacts/neuroguard_model.pt",
        scaler_path: str = "./artifacts/scaler.npz"
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self):
        logger.info("Loading model and scaler...")

        if not self.model_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Artifacts not found. You have to train the model first.\n"
            )

        state_dict = torch.load(self.model_path, map_location=self.device)
        input_dim = state_dict['network.0.weight'].shape[1]
        num_classes = state_dict['network.11.weight'].shape[0]

        self.model = NeuroGuard(input_dim=input_dim, num_classes=num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval() 
        scaler_data = np.load(self.scaler_path)
        self.scaler = StandardScaler()
        self.scaler.mean_ = scaler_data['mean']
        self.scaler.scale_ = scaler_data['scale']

        logger.info(f"Model and scaler loaded successfully. dimensions={input_dim}, classes={num_classes}")

    def predict(self, features: np.ndarray) -> str:
        features = np.array(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features = (features - self.scaler.mean_) / self.scaler.scale_

        tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            predicted_class = torch.argmax(logits, dim=1).item()

        logger.info("Predicted class: %s", CLASS_NAMES[predicted_class])
        return CLASS_NAMES[predicted_class]
    
    def predict_random_test(
        self,
        test_path: str = "./data/KDDTrain+.txt",
        columns_path: str = "./data/processed/columns.npy"
        ):

        feature_columns = np.load(columns_path, allow_pickle=True).tolist()

        df = pd.read_csv(test_path, names=columns, header=None)
        row = df.sample(1)

        true_label_str = row['label'].values[0]
        true_label = attack_map.get(true_label_str, -1)
        true_name = CLASS_NAMES.get(true_label, f"Unknown ({true_label_str})")

        row = row.drop(columns=['label', 'difficulty_level'])
        row = pd.get_dummies(row, columns=['protocol_type', 'service', 'flag'])
        row = row.reindex(columns=feature_columns, fill_value=0) 

        predicted = self.predict(row.values)
        return predicted, true_name