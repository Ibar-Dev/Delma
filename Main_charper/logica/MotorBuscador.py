import csv
import pandas as pd
import numpy as np
import logging as logg
import re
from database_pruebas1.componentes_db import rf_components as rfcom


class MotorBusqueda:
    def __init__(self):
        self.btn_buscar_escrito_por_usuario = None
        self.df_csv = None
    def _lector_csv(self):
        self.df_csv = pd.read_csv(rfcom)
        return self.df_csv if self.df_csv is not None and not self.df_csv.empty() else f'CSV vacío'


    def _normalize_word(self, dirty_word):
        pass
    

    def _buscar_escrito_por_usuario(self, frame_written):
        try:
            if frame_written is not None and not frame_written.emty():
                search_word = frame_written.strip()
                if search_word:
                    clean_word = search_word._normalize_word(search_word)

        except Exception as e:
            print("ERROR EN LA FUNCION _buscar_escrito_por_usuario ")
