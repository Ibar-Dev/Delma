# core/gestion_solicitudes.py

# Aquí se implementará la lógica para gestionar las solicitudes de material.

# from .modelos_datos import Solicitud, Item
# from ..database import queries
"ID,Tipo,Modelo,Fabricante,Frecuencia_Min_MHz,Frecuencia_Max_MHz,Potencia_Max_W,Impedancia_Ohm,Precio_USD,Stock"
import csv
import pandas as pd

df_componentes_csv = pd.read_csv(r'C:\Users\Ibarv\.vscode\WorkSpace_1\Delma-main\Main_charper\database_pruebas1\componentes_db\rf_components.csv')



class GestionSolicitudes:
    def __init__(self) -> None:
        self.id = None
        self.tipo = None
        self.modelo = None
        self.fabricante = None
        self.frecuencia_compra = None
        self.precio_euro = None

    def nuevo_componente(self)-> None:
        pass
    def añadir_a_stock(self, id_a_añadir: str, cantidad: int):
        """
        Añade una cantidad al stock de un componente existente.
        Si el componente no existe, llama a nuevo_componente (aún no implementado)
        
        Args:
            id_a_añadir (str): ID del componente
            cantidad (int): Cantidad a añadir al stock actual
        
        Returns:
            int: Nuevo valor del stock si existe, -1 si hay error
        """

        try:
            if id_a_añadir in df_componentes_csv['ID'].values:
                indice = df_componentes_csv
        
        except Exception as e:
            return f'Problema en funcion "añadi_a_stock"{e}'
    