import tkinter as tk

from sqlmodel import col


class Interface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Ventana Maestra')
        self.geometry("1200x800")
        self.crear_widgets_internos()

    def crear_widgets_internos(self):
        etiqueta = tk.Label(self, text="Esta ventana es una instancia de Interface!", font=("Arial", 18))
        
        boton_cerrar = tk.Button(self, text="Buscar", command="") #A la espera de incorporar funsion
        boton_cerrar.grid(column=0, row=0, padx=1, pady=1)


if __name__ == "__main__":
    app = Interface()

    app.mainloop()

