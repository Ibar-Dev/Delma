# core/modelos_datos.py

class Item:
    def __init__(self, id_item: int, nombre: str, descripcion: str, cantidad: int):
        self.id_item = id_item
        self.nombre = nombre
        self.descripcion = descripcion
        self.cantidad = cantidad

    def __str__(self):
        return f"Item({self.id_item}, '{self.nombre}', Cantidad: {self.cantidad})"

# Aquí se podrían definir otras clases como Solicitud, Tecnico, etc.