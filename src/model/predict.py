import numpy as np
import torch
import logging

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
        self.scaler.mean = scaler_data['mean']
        self.scaler.scale = scaler_data['scale']

        logger.info(f"Model and scaler loaded successfully. dimensions={input_dim}, classes={num_classes}")

    def predict(self):
        if not self.model_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Artifacts not found. You have to train the model first.\n"
            )
        
        pass