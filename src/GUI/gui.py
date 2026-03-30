import tkinter as tk
from ..preprocess import DataPreprocesor
from src.train import Training
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(name)s = %(levelname)s - %(message)s'
)

class GUINeuroguard(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("NeuroGuard")
        self.geometry("600x400")     
        self.label = tk.Label(self, text = 'NEUROGUARD')
        self.label.pack(pady=20)
        self.button = tk.Button

        self.preprocess = self.button(
            self, 
            text = 'Preproces Data', 
            width = 25, 
            command = self.preprocess_data()
        )
        self.button.pack(pady=10)
        
        self.train = self.button(
            self, 
            text = 'Train model', 
            width = 25, 
            command = self.train_model()
        )
        self.button.pack(pady=20)

        
    def preprocess_data(self):
        try:
            DataPreprocesor.preprocesor('KDDTrain+.txt', 'processed')
        except Exception as e:
            logging.error(f'Error: {e}')
        
    def train_model(self):
        try:
            trainer = Training()
            trainer.train()
        except Exception as e:
            logging.error(f'Error{e}')
            
            
            
if __name__ == "__main__":
    app = GUINeuroguard()
    app.mainloop()