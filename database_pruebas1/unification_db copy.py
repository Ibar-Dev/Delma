import sys
from typing import Optional, TypeGuard, Union
from pathlib import Path
import pymysql
import sqlite3
import psycopg2
from pymongo import MongoClient
import os # os was imported on the same line as sqlite3, moved for clarity
import polars as ps
import xml.etree.ElementTree as ET
import io
# Add the project root directory to sys.path
# The script is at Delma-main/Main_charper/database_pruebas1/unification_db.py
# The project root (containing Main_charper) is Delma-main.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from Main_charper.database_pruebas1.componentes_db import rf_components , power_converters

path_json = power_converters

path_csv = rf_components


class BasedatosUnica():
    def __init__(self):
        self.mysql = None
        self.sqlite = None
        self.postgresql = None
        self.mongodb = None
        self.excel = None
        self.csv = None
        self.json = None
        self.yaml = None
    
    def _get_file_extension(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Extracts the file extension from a file path.
        Example: '/path/to/file.csv' -> 'csv'
        """
        try:
            return os.path.splitext(str(file_path))[1].lstrip('.').lower() if file_path else None

        except Exception as e:
            print(f"Error in _get_file_extension: {e}")
            return None
        
    def read_file(self, file_to_read: Union[str, Path]) -> Optional[ps.DataFrame]:
        """
        Reads a file into a Polars DataFrame based on its extension.
        Stores the DataFrame in the corresponding instance attribute and returns it.
        Returns None if reading fails or the file type is unsupported.
        """
        try:
            type_file = self._get_file_extension(file_to_read)
            if not isinstance(type_file, str):
                print(f"Error: Could not determine file type for {file_to_read} or path is invalid.")
                return None

            if type_file == 'csv':
                self.csv = ps.read_csv(file_to_read)
                return self.csv
            
            elif type_file == "json":
                self.json = ps.read_json(file_to_read)
                return self.json
            
            elif type_file == "yaml" or type_file == "yml": # Added yml as a common alias
                # Note: Polars' read_yaml might not exist or might be part of a different module/plugin.
                # Assuming it exists for this example. If not, you'd use a library like PyYAML and then convert to Polars.
                self.yaml = ps.read_yaml(file_to_read) # Corrected: call the function
                return self.yaml
            
            elif type_file in ["xls", "xlsx"]: # Corrected "xlx" to "xls"
                self.excel = ps.read_excel(file_to_read)
                return self.excel
            else:
                print(f"Unsupported file type: {type_file}")
                return None

        except Exception as e:
            print(f"Error in read_file: {e}")
            return None
        



objeto_unificado = BasedatosUnica()

df = objeto_unificado.read_file(path_json)


print(df)
