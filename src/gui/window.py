import tkinter as tk
from src.model.preprocess import DataPreprocesor
from src.model.train import Training
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
        self.label = tk.Label(self, text = 'NEUROGUARD').pack(pady = 20)
        self.preprocess = tk.Button(
            self, 
            text = 'Preproces Data', 
            width = 25, 
            command = self.preprocess_data
        ).pack(pady = 10)       
         
        self.train = tk.Button(
            self, 
            text = 'Train model', 
            width = 25, 
            command = self.train_model
        ).pack(pady = 10)


        
    def preprocess_data(self):
        logging.info("Starting data processing...")
        try:
            DataPreprocesor.preprocesor('KDDTrain+.txt', 'processed')
        except Exception as e:
            logging.error(f'Error: {e}')
        
    def train_model(self):
        logging.info("Starting train of model...")
        try:
            trainer = Training()
            trainer.train()
        except Exception as e:
            logging.error(f'Error{e}')
            
            
            
if __name__ == "__main__":
    app = GUINeuroguard()
    app.mainloop()