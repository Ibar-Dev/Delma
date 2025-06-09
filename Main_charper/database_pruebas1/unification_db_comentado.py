import sys
import logging
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import os

# Third-party imports
import polars as pl  # Librería más rápida que pandas para manejo de DataFrames
import yaml
import json
from contextlib import contextmanager  # Para crear context managers personalizados

# Database imports (with error handling for optional dependencies)
# Patrón de importación defensiva: si no está la librería, no se rompe todo el programa
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

# Setup paths
# Obtiene la ruta del proyecto y la agrega al PATH para importar módulos
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
# Configura el sistema de logging para registrar eventos y errores
logging.basicConfig(
    level=logging.INFO,  # Nivel mínimo de mensajes a mostrar
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Formato del mensaje
)
logger = logging.getLogger(__name__)  # Logger específico para este módulo


class FileType(Enum):
    """
    Enumeración de tipos de archivo soportados.
    
    ¿Por qué usar Enum?
    - Evita errores de tipeo (no puedes escribir "scv" en lugar de "csv")
    - Hace el código más legible y mantenible
    - Proporciona autocompletado en IDEs
    - Es más fácil agregar nuevos tipos
    """
    CSV = "csv"
    JSON = "json"
    YAML = "yaml"
    YML = "yml"
    EXCEL_XLS = "xls"
    EXCEL_XLSX = "xlsx"
    XML = "xml"
    PARQUET = "parquet"


class DatabaseType(Enum):
    """
    Enumeración de tipos de base de datos soportados.
    
    Misma lógica que FileType pero para bases de datos.
    """
    MYSQL = "mysql"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


@dataclass
class DatabaseConfig:
    """
    Configuración para conexiones de base de datos.
    
    ¿Qué hace @dataclass?
    - Genera automáticamente __init__, __repr__, __eq__, etc.
    - Reduce código boilerplate
    - Hace la clase más limpia y fácil de usar
    - Proporciona valores por defecto
    """
    host: str = "localhost"
    port: int = 3306
    username: str = ""
    password: str = ""
    database: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la configuración a diccionario.
        
        ¿Por qué este método?
        - Las librerías de BD esperan diccionarios como parámetros
        - Mapea 'username' a 'user' que es lo que esperan las librerías
        """
        return {
            'host': self.host,
            'port': self.port,
            'user': self.username,  # Nota: 'user' no 'username'
            'password': self.password,
            'database': self.database
        }


class BaseDataSource(ABC):
    """
    Clase base abstracta para fuentes de datos.
    
    ¿Por qué usar ABC (Abstract Base Class)?
    - Define un "contrato" que todas las subclases deben cumplir
    - Garantiza que todas las fuentes de datos tengan los mismos métodos
    - Si una subclase no implementa los métodos abstractos, Python lanza error
    - Facilita el polimorfismo (tratar objetos diferentes de la misma manera)
    """
    
    @abstractmethod
    def read(self, source: Union[str, Path], **kwargs) -> Optional[pl.DataFrame]:
        """
        Lee datos de una fuente y retorna un DataFrame de Polars.
        
        ¿Por qué abstractmethod?
        - Obliga a las subclases a implementar este método
        - Define la interfaz común para todas las fuentes de datos
        """
        pass
    
    @abstractmethod
    def write(self, df: pl.DataFrame, destination: Union[str, Path], **kwargs) -> bool:
        """
        Escribe DataFrame a un destino.
        
        Retorna bool para indicar éxito/fallo de forma simple.
        """
        pass


class FileDataSource(BaseDataSource):
    """
    Manejador para fuentes de datos basadas en archivos.
    
    Implementa el patrón Strategy: usa diccionarios para mapear
    tipos de archivo a sus respectivas funciones de lectura/escritura.
    """
    
    def __init__(self):
        """
        Inicializa los mapeos de lectores y escritores.
        
        ¿Por qué usar diccionarios en lugar de if/elif?
        - Más limpio y mantenible
        - Fácil agregar nuevos tipos
        - Lookup O(1) en lugar de O(n)
        - Separa la lógica de detección de la implementación
        """
        self._readers = {
            FileType.CSV: self._read_csv,
            FileType.JSON: self._read_json,
            FileType.YAML: self._read_yaml,
            FileType.YML: self._read_yaml,  # YAML y YML usan el mismo lector
            FileType.EXCEL_XLS: self._read_excel,
            FileType.EXCEL_XLSX: self._read_excel,  # XLS y XLSX usan el mismo lector
            FileType.XML: self._read_xml,
            FileType.PARQUET: self._read_parquet,
        }
        
        self._writers = {
            FileType.CSV: self._write_csv,
            FileType.JSON: self._write_json,
            FileType.YAML: self._write_yaml,
            FileType.YML: self._write_yaml,
            FileType.EXCEL_XLSX: self._write_excel,  # Solo XLSX para escritura
            FileType.PARQUET: self._write_parquet,
        }
    
    def _get_file_type(self, file_path: Union[str, Path]) -> Optional[FileType]:
        """
        Extrae el tipo de archivo de la ruta.
        
        ¿Por qué método privado (con _)?
        - Es lógica interna de la clase
        - No debería ser llamado desde fuera
        - Convención de Python para métodos "privados"
        """
        try:
            # Obtiene la extensión sin el punto y en minúsculas
            extension = Path(file_path).suffix.lstrip('.').lower()
            
            # Busca en todos los tipos de archivo soportados
            for file_type in FileType:
                if file_type.value == extension:
                    return file_type
            return None  # Tipo no soportado
        except Exception as e:
            logger.error(f"Error determining file type for {file_path}: {e}")
            return None
    
    def read(self, source: Union[str, Path], **kwargs) -> Optional[pl.DataFrame]:
        """
        Lee archivo en DataFrame de Polars.
        
        ¿Por qué **kwargs?
        - Permite pasar parámetros específicos a cada lector
        - Flexibilidad sin romper la interfaz común
        - Ejemplo: encoding para CSV, sheet_name para Excel
        """
        try:
            file_path = Path(source)  # Convierte a Path para manejo uniforme
            
            # Validación: el archivo debe existir
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Detecta el tipo de archivo
            file_type = self._get_file_type(file_path)
            if not file_type:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return None
            
            # Obtiene el lector apropiado
            reader = self._readers.get(file_type)
            if not reader:
                logger.error(f"No reader available for file type: {file_type}")
                return None
            
            logger.info(f"Reading {file_type.value.upper()} file: {file_path}")
            return reader(file_path, **kwargs)  # Llama al lector específico
            
        except Exception as e:
            # Manejo de errores defensivo: registra y retorna None
            logger.error(f"Error reading file {source}: {e}")
            return None
    
    def write(self, df: pl.DataFrame, destination: Union[str, Path], **kwargs) -> bool:
        """
        Escribe DataFrame a archivo.
        
        ¿Por qué retornar bool?
        - Interfaz simple para indicar éxito/fallo
        - Consistente con otras operaciones de I/O
        """
        try:
            file_path = Path(destination)
            file_type = self._get_file_type(file_path)
            
            if not file_type:
                logger.error(f"Unsupported file type for writing: {file_path.suffix}")
                return False
            
            writer = self._writers.get(file_type)
            if not writer:
                logger.error(f"No writer available for file type: {file_type}")
                return False
            
            # Crea directorios padre si no existen
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Writing {file_type.value.upper()} file: {file_path}")
            return writer(df, file_path, **kwargs)
            
        except Exception as e:
            logger.error(f"Error writing file {destination}: {e}")
            return False
    
    # Métodos de lectura específicos por formato
    # Cada uno maneja las particularidades de su formato
    
    def _read_csv(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """Lee archivos CSV usando Polars."""
        return pl.read_csv(file_path, **kwargs)
    
    def _read_json(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """Lee archivos JSON usando Polars."""
        return pl.read_json(file_path, **kwargs)
    
    def _read_yaml(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """
        Lee archivos YAML.
        
        ¿Por qué conversión manual?
        - Polars no tiene lector nativo para YAML
        - YAML puede ser un dict o lista de dicts
        - Necesitamos normalizar la estructura
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # Si es un dict, lo convierte en lista de un elemento
        return pl.from_dicts([data] if isinstance(data, dict) else data)
    
    def _read_excel(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """Lee archivos Excel usando Polars."""
        return pl.read_excel(file_path, **kwargs)
    
    def _read_xml(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """
        Lee archivos XML básicos.
        
        Nota: Esta es una implementación básica.
        XML puede ser muy complejo y esta implementación
        asume una estructura simple y plana.
        """
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Convierte XML a lista de diccionarios
        data = []
        for elem in root:
            row = {}
            for child in elem:
                row[child.tag] = child.text
            data.append(row)
        
        return pl.from_dicts(data)
    
    def _read_parquet(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """Lee archivos Parquet usando Polars."""
        return pl.read_parquet(file_path, **kwargs)
    
    # Métodos de escritura específicos por formato
    
    def _write_csv(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """Escribe CSV usando Polars."""
        df.write_csv(file_path, **kwargs)
        return True
    
    def _write_json(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """Escribe JSON usando Polars."""
        df.write_json(file_path, **kwargs)
        return True
    
    def _write_yaml(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """
        Escribe YAML.
        
        Convierte DataFrame a lista de diccionarios y luego a YAML.
        """
        data = df.to_dicts()
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, **kwargs)
        return True
    
    def _write_excel(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """Escribe Excel usando Polars."""
        df.write_excel(file_path, **kwargs)
        return True
    
    def _write_parquet(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """Escribe Parquet usando Polars."""
        df.write_parquet(file_path, **kwargs)
        return True


class DatabaseDataSource(BaseDataSource):
    """
    Manejador para conexiones de base de datos.
    
    Soporta múltiples tipos de BD con una interfaz unificada.
    """
    
    def __init__(self, db_type: DatabaseType, config: DatabaseConfig):
        """
        Inicializa el manejador de BD.
        
        ¿Por qué separar tipo y configuración?
        - Flexibilidad: misma config para diferentes tipos
        - Claridad: el tipo determina qué librería usar
        - Extensibilidad: fácil agregar nuevos tipos
        """
        self.db_type = db_type
        self.config = config
        self._connection = None  # Conexión reutilizable (no usada en este ejemplo)
    
    @contextmanager
    def get_connection(self):
        """
        Context manager para conexiones de base de datos.
        
        ¿Por qué context manager?
        - Garantiza que las conexiones se cierren automáticamente
        - Manejo de errores centralizado
        - Patrón RAII (Resource Acquisition Is Initialization)
        - Evita memory leaks por conexiones abiertas
        
        ¿Cómo funciona?
        1. Se ejecuta el código antes del yield (setup)
        2. yield devuelve la conexión al código que usa el context manager
        3. Se ejecuta el código después del yield (cleanup)
        4. El finally garantiza que cleanup siempre se ejecute
        """
        try:
            # Crea conexión según el tipo de BD
            if self.db_type == DatabaseType.MYSQL and MYSQL_AVAILABLE:
                conn = pymysql.connect(**self.config.to_dict())
            elif self.db_type == DatabaseType.SQLITE and SQLITE_AVAILABLE:
                conn = sqlite3.connect(self.config.database)
            elif self.db_type == DatabaseType.POSTGRESQL and POSTGRESQL_AVAILABLE:
                conn = psycopg2.connect(**self.config.to_dict())
            elif self.db_type == DatabaseType.MONGODB and MONGODB_AVAILABLE:
                client = MongoClient(f"mongodb://{self.config.host}:{self.config.port}")
                conn = client[self.config.database]
            else:
                raise ValueError(f"Database type {self.db_type} not available or supported")
            
            yield conn  # Entrega la conexión al código que la usa
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise  # Re-lanza la excepción para que el código que llama la maneje
        finally:
            # Cleanup: cierra la conexión si existe y no es MongoDB
            if conn and self.db_type != DatabaseType.MONGODB:
                conn.close()
    
    def read(self, source: str, **kwargs) -> Optional[pl.DataFrame]:
        """
        Lee de tabla de BD o ejecuta query.
        
        ¿Por qué source es str y no Path?
        - Para BD, source puede ser nombre de tabla o query SQL
        - Path no tiene sentido en contexto de BD
        """
        try:
            with self.get_connection() as conn:
                if self.db_type == DatabaseType.MONGODB:
                    # MongoDB manejo especial (NoSQL)
                    collection = conn[source]  # source es nombre de colección
                    data = list(collection.find())  # Obtiene todos los documentos
                    return pl.from_dicts(data)
                else:
                    # Bases de datos SQL
                    # Si source no es un SELECT, asume que es nombre de tabla
                    query = f"SELECT * FROM {source}" if not source.upper().startswith('SELECT') else source
                    return pl.read_database(query, conn, **kwargs)
        except Exception as e:
            logger.error(f"Error reading from database: {e}")
            return None
    
    def write(self, df: pl.DataFrame, destination: str, **kwargs) -> bool:
        """
        Escribe DataFrame a tabla de BD.
        
        destination es el nombre de la tabla/colección.
        """
        try:
            with self.get_connection() as conn:
                if self.db_type == DatabaseType.MONGODB:
                    # MongoDB: convierte DataFrame a documentos
                    collection = conn[destination]
                    data = df.to_dicts()  # Lista de diccionarios
                    collection.insert_many(data)
                else:
                    # SQL: usa método nativo de Polars
                    df.write_database(destination, conn, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Error writing to database: {e}")
            return False


class UnifiedDataHandler:
    """
    Manejador unificado para múltiples fuentes de datos.
    
    Esta es la clase principal que unifica todo el sistema.
    Proporciona una interfaz única para trabajar con archivos y BD.
    
    Patrones implementados:
    - Facade: Simplifica el acceso a subsistemas complejos
    - Strategy: Delega a manejadores específicos
    - Cache: Almacena datos leídos para acceso rápido
    """
    
    def __init__(self):
        """
        Inicializa el manejador unificado.
        
        ¿Por qué esta estructura?
        - file_handler: Maneja todos los archivos
        - database_handlers: Dict de conexiones nombradas
        - _data_cache: Cache para evitar re-lecturas costosas
        """
        self.file_handler = FileDataSource()
        self.database_handlers: Dict[str, DatabaseDataSource] = {}
        self._data_cache: Dict[str, pl.DataFrame] = {}  # Cache simple en memoria
    
    def add_database_connection(self, name: str, db_type: DatabaseType, config: DatabaseConfig):
        """
        Agrega una configuración de conexión de BD.
        
        ¿Por qué conexiones nombradas?
        - Permite múltiples conexiones (ej: BD de producción y desarrollo)
        - Interfaz más clara: "main_db:users" vs parámetros complejos
        - Reutilización de configuraciones
        """
        self.database_handlers[name] = DatabaseDataSource(db_type, config)
        logger.info(f"Added database connection: {name} ({db_type.value})")
    
    def read_data(self, source: Union[str, Path], source_type: str = "auto", **kwargs) -> Optional[pl.DataFrame]:
        """
        Lee datos de varias fuentes.
        
        ¿Cómo funciona la auto-detección?
        1. Si source_type es "auto", intenta detectar automáticamente
        2. Si es una ruta que existe → archivo
        3. Si está en database_handlers → base de datos
        4. Por defecto asume archivo
        
        ¿Por qué cache?
        - Evita re-leer archivos grandes
        - Mejora performance en análisis iterativo
        - Usa memoria como trade-off por velocidad
        """
        try:
            # Auto-detección de tipo de fuente
            if source_type == "auto":
                if isinstance(source, (str, Path)) and Path(source).exists():
                    source_type = "file"
                elif isinstance(source, str) and source in self.database_handlers:
                    source_type = "database"
                else:
                    source_type = "file"  # Default a archivo
            
            # Clave de cache para esta operación de lectura
            cache_key = f"{source_type}:{source}"
            
            # Verifica cache primero
            if cache_key in self._data_cache:
                logger.info(f"Returning cached data for: {cache_key}")
                return self._data_cache[cache_key]
            
            # Lee datos según el tipo
            if source_type == "file":
                df = self.file_handler.read(source, **kwargs)
            elif source_type == "database":
                # Para BD, source debe estar en formato "nombre_conexion:tabla"
                if ":" in str(source):
                    conn_name, table_name = str(source).split(":", 1)
                    if conn_name in self.database_handlers:
                        df = self.database_handlers[conn_name].read(table_name, **kwargs)
                    else:
                        logger.error(f"Database connection not found: {conn_name}")
                        return None
                else:
                    logger.error("Database source must be in format 'connection_name:table_name'")
                    return None
            else:
                logger.error(f"Unknown source type: {source_type}")
                return None
            
            # Cachea lecturas exitosas
            if df is not None:
                self._data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading data from {source}: {e}")
            return None
    
    def write_data(self, df: pl.DataFrame, destination: Union[str, Path], 
                   destination_type: str = "auto", **kwargs) -> bool:
        """
        Escribe DataFrame a varios destinos.
        
        Lógica similar a read_data pero para escritura.
        No usa cache porque escribir no se beneficia de caching.
        """
        try:
            # Auto-detección de tipo de destino
            if destination_type == "auto":
                if isinstance(destination, (str, Path)) and ":" not in str(destination):
                    destination_type = "file"
                else:
                    destination_type = "database"
            
            if destination_type == "file":
                return self.file_handler.write(df, destination, **kwargs)
            elif destination_type == "database":
                if ":" in str(destination):
                    conn_name, table_name = str(destination).split(":", 1)
                    if conn_name in self.database_handlers:
                        return self.database_handlers[conn_name].write(df, table_name, **kwargs)
                    else:
                        logger.error(f"Database connection not found: {conn_name}")
                        return False
                else:
                    logger.error("Database destination must be in format 'connection_name:table_name'")
                    return False
            else:
                logger.error(f"Unknown destination type: {destination_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error writing data to {destination}: {e}")
            return False
    
    def get_data_info(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Obtiene información básica sobre un DataFrame.
        
        ¿Por qué este método?
        - Información útil para análisis exploratorio
        - Interfaz consistente para metadatos
        - Fácil debugging y logging
        """
        return {
            "shape": df.shape,  # (filas, columnas)
            "columns": df.columns,  # Lista de nombres de columnas
            "dtypes": dict(zip(df.columns, [str(dtype) for dtype in df.dtypes])),  # Tipos de datos
            "memory_usage": df.estimated_size("mb"),  # Uso de memoria en MB
            "null_counts": df.null_count().to_dicts()[0] if not df.is_empty() else {}  # Conteo de nulos
        }
    
    def clear_cache(self):
        """
        Limpia el cache de datos.
        
        ¿Cuándo usar?
        - Cuando la memoria se agota
        - Después de procesar lotes grandes
        - Para forzar re-lectura de archivos modificados
        """
        self._data_cache.clear()
        logger.info("Data cache cleared")
    
    def list_cached_data(self) -> List[str]:
        """
        Lista todas las fuentes de datos en cache.
        
        Útil para debugging y monitoreo de memoria.
        """
        return list(self._data_cache.keys())


# Función principal de ejemplo
def main():
    """
    Ejemplo de uso del UnifiedDataHandler.
    
    Demuestra cómo usar el sistema completo:
    1. Inicializar el handler
    2. Configurar conexiones de BD (opcional)
    3. Leer datos de diferentes fuentes
    4. Obtener información de los datos
    5. Escribir a diferentes formatos
    """
    # Inicializa el manejador principal
    handler = UnifiedDataHandler()
    
    # Ejemplo de configuración de BD (comentado porque no tenemos BD real)
    # handler.add_database_connection(
    #     "main_db", 
    #     DatabaseType.MYSQL, 
    #     DatabaseConfig(host="localhost", username="user", password="pass", database="mydb")
    # )
    
    try:
        # Intenta importar rutas de componentes (específico del proyecto)
        from Main_charper.database_pruebas1.componentes_db import rf_components, power_converters
        
        # Lee datos JSON
        print("Reading power converters data...")
        power_df = handler.read_data(power_converters)
        if power_df is not None:
            print(f"Power converters shape: {power_df.shape}")
            print(f"Columns: {power_df.columns}")
            print("\nFirst few rows:")
            print(power_df.head())
            
            # Obtiene información del DataFrame
            info = handler.get_data_info(power_df)
            print(f"\nData info: {info}")
        
        # Lee datos CSV
        print("\n" + "="*50)
        print("Reading RF components data...")
        rf_df = handler.read_data(rf_components)
        if rf_df is not None:
            print(f"RF components shape: {rf_df.shape}")
            print(f"Columns: {rf_df.columns}")
            print("\nFirst few rows:")
            print(rf_df.head())
        
        # Ejemplo: Escribe datos a formato diferente
        if power_df is not None:
            output_path = Path("output_power_converters.csv")
            if handler.write_data(power_df, output_path):
                print(f"\nSuccessfully wrote data to {output_path}")
        
    except ImportError as e:
        logger.error(f"Could not import component paths: {e}")
        print("Please ensure the component modules are available.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


# Punto de entrada del script
if __name__ == "__main__":
    """
    Se ejecuta solo si el archivo se ejecuta directamente,
    no si se importa como módulo.
    
    ¿Por qué esta estructura?
    - Permite usar el módulo como librería o como script
    - Evita ejecución accidental al importar
    - Patrón estándar en Python
    """
    main()