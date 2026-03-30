import tkinter as tk
from preprocess import DataPreprocesor

class GUINeuroguard(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("NeuroGuard")
        self.geometry("600x400")     
        self.label = tk.Label(self, text = 'NEUROGUARD')
        self.label.pack(pady=20)
        
        self.button = tk.Button(
            self, 
            text = 'Preproces Data', 
            width = 25, 
            command = lambda: DataPreprocesor.preprocesor('KDDTrain+.txt', 'processed')
        )
        self.button.pack(pady=10)
if __name__ == "__main__":
    
    app = GUINeuroguard()
    app.mainloop()