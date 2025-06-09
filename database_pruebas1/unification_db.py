
import logging.config
import sys
import logging
from tkinter import NO
from tkinter.messagebox import RETRY
from turtle import st
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import os

# Third-party imports
import polars as ps
import yaml
import json
from contextlib import contextmanager

# Database imports (with error handling for optional dependencies)
try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logging.warning("PyMySQL not available. MySQL functionality disabled.")

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    logging.warning("SQLite3 not available.")

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    logging.warning("psycopg2 not available. PostgreSQL functionality disabled.")

try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logging.warning("PyMongo not available. MongoDB functionality disabled.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level= logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger_my = logging.getLogger(__name__)


class MyFiletype(Enum):
    CSV = "csv"
    JSON = "json"
    YAML = "yaml"
    YML = "yml"
    EXCEL_XLS = "xls"
    EXCEL_XLSX = "xlsx"
    XML = "xml"
    PARQUET = "parquet"

class Databasetype(Enum):
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"



@dataclass
class DatabaseConfig:
    """Config for Database connections"""
    hots:str = "localhost"
    port:int = 3306
    username:str = ""
    password:str = ""
    database:str = ""

    def my_dict(self)-> Dict[str, Any]:
        return {
            'hots': self.hots,
            'port': self.port,
            'user': self.username,
            'password': self.password,
            'database': self.database
        }
    
class BasedataSource(ABC):

    @abstractmethod
    def read(self, source:Union[str, Path], **kwargs)-> Optional[ps.DataFrame]:
        """Read source as data and return as polars Dataframe."""
        pass


    @abstractmethod
    def write(self, df: ps.DataFrame, destination:Union[str, Path], **kwargs)-> bool:
        pass



class FileDataSource(BasedataSource):
    """Voy a manejar tanto los archivos como los dataframe"""

    def __init__(self):
        self._readers = {
            MyFiletype.CSV: self._read_csv,
            MyFiletype.JSON: self._read_json,
            MyFiletype.YAML: self._read_yaml,
            MyFiletype.YML: self._read_yaml,
            MyFiletype.EXCEL_XLS: self._read_excel,
            MyFiletype.EXCEL_XLSX: self._read_excel,
            MyFiletype.XML: self._read_xml,
            MyFiletype.PARQUET: self._read_parquet,
        }

        self.writters = {
            MyFiletype.CSV: self._write_csv,
            MyFiletype.JSON: self._write_json,
            MyFiletype.YAML: self._write_yaml,
            MyFiletype.YML: self._write_yaml,
            MyFiletype.EXCEL_XLS: self._write_excel,
            MyFiletype.EXCEL_XLSX: self._write_excel,
            MyFiletype.XML: self._write_xml,
            MyFiletype.PARQUET: self._write_parquet,
        }

    def _check_df_exists(self, df: Optional[ps.DataFrame])-> bool:
        'Funcion para comprobar si el Dataframe con el que se trabaja tiene algo'
        pass

    def _get_file_type(self, file_path: Union[str, Path])-> Optional[MyFiletype]:

        try:
            extension = Path(file_path).suffix.lstrip('.').lower()
            for file_type in MyFiletype:
                if file_type == extension:
                    return file_type
            return None
        
        except TypeError as e:
            logger_my.error(f'Error determinig file type for {file_path}: {e}')

    
    def read(self, source: Union[str, Path], **kwargs)-> Optional[ps.DataFrame]:
        try:
            file_path = Path(source)
            if not file_path.exists():
                logger_my.error(f'Path del archivo no encontrado {file_path}')
                return None
            


        except TypeError as e:
            logger_my.error(f'Tenemos error en la funcion read por {source}: {e}')