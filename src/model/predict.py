import numpy as np

from .preprocess import DataPreprocesor


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s = %(levelname)s - %(message)s'
)

class Predictor:
    def __init__(self, model_path: str = "./artifacts/neuroguard_model.pt", scaler_path: str = "./artifacts/scaler.npz"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        
    def load_model(self):
        logger.info("Loading model and scaler ...")