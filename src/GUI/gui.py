import tkinter as tk


class GUINeuroguard(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("NeuroGuard")
        self.geometry("600x400")
        
        self.label = tk.Label(self, text = 'NEUROGUARD')

if __name__ == "__main__":
    
    app = GUINeuroguard()
    app.mainloop()