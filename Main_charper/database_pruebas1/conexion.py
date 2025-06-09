from typing import Optional, TypeGuard, Union
from pathlib import Path
import pymysql
import sqlite3
import psycopg2
from pymongo import MongoClient

def is_valid_db_path(value: Optional[Union[str, bytes, Path]]) -> TypeGuard[Union[str, bytes, Path]]:
    return isinstance(value, (str, bytes, Path)) and bool(value)

class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.cursor = None

    def connect(self, db_type: str, **kwargs) -> bool:
        """
        Connect to different types of databases
        
        Parameters:
            db_type: str - Type of database ('mysql', 'postgresql', 'sqlite', 'mongodb')
            **kwargs: Connection parameters specific to each database
        """
        try:
            if db_type.lower() == 'mysql':
                self.connection = pymysql.connect(
                    host=kwargs.get('host', 'localhost'),
                    user=kwargs.get('user', ''),
                    password=kwargs.get('password', ''),
                    database=kwargs.get('database', '')
                )
                
            elif db_type.lower() == 'postgresql':
                self.connection = psycopg2.connect(
                    host=kwargs.get('host', 'localhost'),
                    user=kwargs.get('user'),
                    password=kwargs.get('password'),
                    database=kwargs.get('database')
                )
                
            elif db_type.lower() == 'sqlite':
                database = kwargs.get('database')
                if is_valid_db_path(database):
                    self.connection = sqlite3.connect(database)
                else:
                    raise ValueError("Se requiere una ruta válida para la base de datos SQLite")
                
            elif db_type.lower() == 'mongodb':
                client = MongoClient(kwargs.get('connection_string'))
                database = kwargs.get('database')
                if not database:
                    raise ValueError("Database name is required for MongoDB connection")
                self.connection = client[database]
            
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            if db_type.lower() != 'mongodb':
                self.cursor = self.connection.cursor()
            
            return True

        except Exception as e:
            print(f"Error connecting to {db_type} database: {str(e)}")
            return False

    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if isinstance(self.connection, MongoClient):
            self.connection.close()
        elif self.connection is not None:
            self.connection.close()

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[list]:
        """Execute SQL query"""
        try:
            if not self.cursor:
                raise RuntimeError("Database connection not established")

            if params is not None:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            if query.lower().startswith('select'):
                result = self.cursor.fetchall()
                return list(result)
            else:
                if self.connection is not None and hasattr(self.connection, 'commit'):
                    self.connection.commit()
                return None
                
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None