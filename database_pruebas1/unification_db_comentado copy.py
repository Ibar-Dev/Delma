# Importaciones necesarias para el funcionamiento del sistema
import logging
import sys
from abc import ABC, abstractmethod  # Para crear clases abstractas
from dataclasses import dataclass    # Para clases de datos simplificadas
from enum import Enum               # Para enumeraciones type-safe
from pathlib import Path           # Para manejo moderno de rutas de archivos
from typing import Any, Dict, List, Optional, Union  # Type hints para mejor documentación
from contextlib import contextmanager  # Para context managers personalizados

# Importaciones de librerías externas principales
import polars as pl  # DataFrame library más rápida que pandas para datasets grandes
import yaml         # Para leer/escribir archivos YAML

# Database imports (with error handling for optional dependencies)
# Estas importaciones se crean para poder manejar diferentes tipos de bases de datos
# sin que el programa falle si alguna librería no está instalada
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
# Esta configuración se crea para poder importar módulos desde el directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
# Esta configuración se crea para poder registrar eventos, errores y debug información
# durante la ejecución del programa
logging.basicConfig(
    level=logging.INFO,  # Nivel mínimo de mensajes a mostrar
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Formato del mensaje
)
logger = logging.getLogger(__name__)  # Logger específico para este módulo


class FileType(Enum):
    """
    Esta enumeración se crea para poder definir de manera segura todos los tipos
    de archivo que el sistema puede procesar, evitando errores de tipeo y
    proporcionando autocompletado en los IDEs.
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
    Esta enumeración se crea para poder definir de manera segura todos los tipos
    de base de datos que el sistema soporta, manteniendo consistencia en
    la identificación de cada motor de base de datos.
    """
    MYSQL = "mysql"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


@dataclass
class DatabaseConfig:
    """
    Esta clase se crea para poder almacenar y gestionar las configuraciones
    de conexión a bases de datos de forma estructurada y type-safe.
    """
    host: str = "localhost"       # Dirección del servidor de BD
    port: int = 3306             # Puerto de conexión
    username: str = ""           # Usuario para autenticación
    password: str = ""           # Contraseña para autenticación
    database: str = ""           # Nombre de la base de datos
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Este método se crea para poder convertir la configuración en un diccionario
        que sea compatible con las librerías de conexión de bases de datos,
        mapeando los nombres de campos según los estándares de cada librería.
        """
        return {
            'host': self.host,
            'port': self.port,
            'user': self.username,  # Nota: las librerías esperan 'user' no 'username'
            'password': self.password,
            'database': self.database
        }


class BaseDataSource(ABC):
    """
    Esta clase abstracta se crea para poder definir un contrato común que todas
    las fuentes de datos deben cumplir, garantizando que tengan los mismos
    métodos básicos de lectura y escritura.
    """
    
    @abstractmethod
    def read(self, source: Union[str, Path], **kwargs) -> Optional[pl.DataFrame]:
        """
        Este método abstracto se crea para poder obligar a todas las subclases
        a implementar una función de lectura que retorne un DataFrame de Polars
        o None en caso de error.
        """
        pass
    
    @abstractmethod
    def write(self, df: pl.DataFrame, destination: Union[str, Path], **kwargs) -> bool:
        """
        Este método abstracto se crea para poder obligar a todas las subclases
        a implementar una función de escritura que retorne True/False indicando
        el éxito o fallo de la operación.
        """
        pass


class FileDataSource(BaseDataSource):
    """
    Esta clase se crea para poder manejar la lectura y escritura de archivos
    de diferentes formatos usando un patrón de mapeo que asocia cada tipo
    de archivo con su función específica de procesamiento.
    """
    
    def __init__(self):
        """
        Este constructor se crea para poder inicializar los diccionarios que mapean
        cada tipo de archivo con sus funciones correspondientes de lectura y escritura,
        facilitando la extensión del sistema con nuevos formatos.
        """
        # Este diccionario se crea para poder mapear cada tipo de archivo con su función lectora
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
        
        # Este diccionario se crea para poder mapear cada tipo de archivo con su función escritora
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
        Este método se crea para poder determinar automáticamente el tipo de archivo
        basándose en su extensión, facilitando el procesamiento automático sin
        necesidad de especificar manualmente el formato.
        """
        try:
            # Esta línea se crea para poder obtener la extensión del archivo sin el punto
            extension = Path(file_path).suffix.lstrip('.').lower()
            
            # Este bucle se crea para poder buscar la extensión en todos los tipos soportados
            for file_type in FileType:
                if file_type.value == extension:
                    return file_type
            return None  # Tipo no soportado
        except Exception as e:
            logger.error(f"Error determining file type for {file_path}: {e}")
            return None
    
    def read(self, source: Union[str, Path], **kwargs) -> Optional[pl.DataFrame]:
        """
        Este método se crea para poder leer archivos de cualquier formato soportado
        de manera uniforme, detectando automáticamente el tipo y usando el lector
        apropiado sin que el usuario tenga que especificar el formato.
        """
        try:
            file_path = Path(source)  # Esta conversión se crea para poder manejar rutas uniformemente
            
            # Esta validación se crea para poder verificar que el archivo existe antes de procesarlo
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Esta detección se crea para poder identificar automáticamente el tipo de archivo
            file_type = self._get_file_type(file_path)
            if not file_type:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return None
            
            # Esta búsqueda se crea para poder obtener la función lectora específica del tipo
            reader = self._readers.get(file_type)
            if not reader:
                logger.error(f"No reader available for file type: {file_type}")
                return None
            
            logger.info(f"Reading {file_type.value.upper()} file: {file_path}")
            return reader(file_path, **kwargs)  # Esta llamada ejecuta el lector específico
            
        except Exception as e:
            logger.error(f"Error reading file {source}: {e}")
            return None
    
    def write(self, df: pl.DataFrame, destination: Union[str, Path], **kwargs) -> bool:
        """
        Este método se crea para poder escribir DataFrames a archivos de cualquier
        formato soportado, detectando automáticamente el tipo por la extensión
        del archivo de destino.
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
            
            # Esta línea se crea para poder crear automáticamente los directorios padre si no existen
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Writing {file_type.value.upper()} file: {file_path}")
            return writer(df, file_path, **kwargs)
            
        except Exception as e:
            logger.error(f"Error writing file {destination}: {e}")
            return False
    
    # Métodos de lectura específicos por formato
    
    def _read_csv(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """
        Este método se crea para poder leer archivos CSV usando las capacidades
        optimizadas de Polars, que es más rápido que pandas para archivos grandes.
        """
        return pl.read_csv(file_path, **kwargs)
    
    def _read_json(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """
        Este método se crea para poder leer archivos JSON usando Polars,
        manejando tanto JSON de líneas como JSON estándar.
        """
        return pl.read_json(file_path, **kwargs)
    
    def _read_yaml(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """
        Este método se crea para poder leer archivos YAML convirtiéndolos
        a formato DataFrame, ya que Polars no tiene soporte nativo para YAML.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # Esta conversión se crea para poder normalizar tanto dict como list de dicts
        return pl.from_dicts([data] if isinstance(data, dict) else data)
    
    def _read_excel(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """
        Este método se crea para poder leer archivos Excel (.xls y .xlsx)
        usando las capacidades nativas de Polars.
        """
        return pl.read_excel(file_path, **kwargs)
    
    def _read_xml(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """
        Este método se crea para poder leer archivos XML básicos con estructura
        tabular, convirtiendo los elementos XML en filas de DataFrame.
        Nota: Para XML complejo se necesitaría un parser más sofisticado.
        """
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Este bucle se crea para poder convertir elementos XML en diccionarios
        data = []
        for elem in root:
            row = {}
            for child in elem:
                row[child.tag] = child.text
            data.append(row)
        
        return pl.from_dicts(data)
    
    def _read_parquet(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """
        Este método se crea para poder leer archivos Parquet, que son muy
        eficientes para almacenar y procesar grandes volúmenes de datos.
        """
        return pl.read_parquet(file_path, **kwargs)
    
    # Métodos de escritura específicos por formato
    
    def _write_csv(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """
        Este método se crea para poder escribir DataFrames en formato CSV
        usando las capacidades optimizadas de Polars.
        """
        df.write_csv(file_path, **kwargs)
        return True
    
    def _write_json(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """
        Este método se crea para poder escribir DataFrames en formato JSON
        manteniendo la estructura tabular.
        """
        df.write_json(file_path, **kwargs)
        return True
    
    def _write_yaml(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """
        Este método se crea para poder escribir DataFrames en formato YAML,
        convirtiendo primero a lista de diccionarios para mantener la estructura.
        """
        data = df.to_dicts()
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, **kwargs)
        return True
    
    def _write_excel(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """
        Este método se crea para poder escribir DataFrames en formato Excel
        usando las capacidades nativas de Polars.
        """
        df.write_excel(file_path, **kwargs)
        return True
    
    def _write_parquet(self, df: pl.DataFrame, file_path: Path, **kwargs) -> bool:
        """
        Este método se crea para poder escribir DataFrames en formato Parquet,
        ideal para almacenar grandes volúmenes de datos de forma eficiente.
        """
        df.write_parquet(file_path, **kwargs)
        return True


class DatabaseDataSource(BaseDataSource):
    """
    Esta clase se crea para poder manejar conexiones a diferentes tipos de
    bases de datos (SQL y NoSQL) con una interfaz unificada que abstrae
    las diferencias entre los diferentes motores.
    """
    
    def __init__(self, db_type: DatabaseType, config: DatabaseConfig):
        """
        Este constructor se crea para poder inicializar una fuente de datos
        de base de datos con su tipo específico y configuración de conexión.
        """
        self.db_type = db_type
        self.config = config
        self._connection = None  # Esta variable se crea para poder reutilizar conexiones en el futuro
    
    @contextmanager
    def get_connection(self):
        """
        Este context manager se crea para poder manejar conexiones de base de datos
        de forma segura, garantizando que siempre se cierren automáticamente
        y evitando memory leaks.
        """
        conn = None  # Esta inicialización se crea para poder usar la variable en finally
        try:
            # Estas condiciones se crean para poder crear la conexión apropiada según el tipo de BD
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
            
            yield conn  # Esta línea entrega la conexión al código que la usa
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise  # Esta línea re-lanza la excepción para manejo en código superior
        finally:
            # Este bloque se crea para poder cerrar la conexión automáticamente
            if conn and self.db_type != DatabaseType.MONGODB:
                conn.close()
    
    def read(self, source: str, **kwargs) -> Optional[pl.DataFrame]:
        """
        Este método se crea para poder leer datos de bases de datos, ya sea
        de una tabla específica o ejecutando una consulta SQL personalizada.
        """
        try:
            with self.get_connection() as conn:
                if self.db_type == DatabaseType.MONGODB:
                    # Esta sección se crea para poder manejar MongoDB (NoSQL)
                    collection = conn[source]  # source es nombre de colección
                    data = list(collection.find())  # Obtiene todos los documentos
                    return pl.from_dicts(data)
                else:
                    # Esta sección se crea para poder manejar bases de datos SQL
                    # Esta condición se crea para poder aceptar tanto nombres de tabla como queries SQL
                    query = f"SELECT * FROM {source}" if not source.upper().startswith('SELECT') else source
                    return pl.read_database(query, conn, **kwargs)
        except Exception as e:
            logger.error(f"Error reading from database: {e}")
            return None
    
    def write(self, df: pl.DataFrame, destination: str, **kwargs) -> bool:
        """
        Este método se crea para poder escribir DataFrames a bases de datos,
        manejando tanto SQL como NoSQL de forma transparente.
        """
        try:
            with self.get_connection() as conn:
                if self.db_type == DatabaseType.MONGODB:
                    # Esta sección se crea para poder escribir a MongoDB
                    collection = conn[destination]
                    data = df.to_dicts()  # Convierte DataFrame a lista de diccionarios
                    collection.insert_many(data)
                else:
                    # Esta sección se crea para poder escribir a bases de datos SQL
                    df.write_database(destination, conn, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Error writing to database: {e}")
            return False


class UnifiedDataHandler:
    """
    Esta clase se crea para poder proporcionar una interfaz única y simplificada
    para trabajar con múltiples fuentes de datos (archivos y bases de datos)
    sin que el usuario tenga que conocer los detalles de implementación.
    """
    
    def __init__(self):
        """
        Este constructor se crea para poder inicializar el manejador unificado
        con sus componentes principales: manejador de archivos, conexiones de BD
        y sistema de cache para optimizar el rendimiento.
        """
        self.file_handler = FileDataSource()
        self.database_handlers: Dict[str, DatabaseDataSource] = {}
        self._data_cache: Dict[str, pl.DataFrame] = {}  # Este cache se crea para evitar re-lecturas costosas
    
    def add_database_connection(self, name: str, db_type: DatabaseType, config: DatabaseConfig):
        """
        Este método se crea para poder agregar configuraciones de conexión de
        bases de datos con nombres descriptivos, permitiendo múltiples conexiones
        simultáneas (desarrollo, producción, testing, etc.).
        """
        self.database_handlers[name] = DatabaseDataSource(db_type, config)
        logger.info(f"Added database connection: {name} ({db_type.value})")
    
    def read_data(self, source: Union[str, Path], source_type: str = "auto", **kwargs) -> Optional[pl.DataFrame]:
        """
        Este método se crea para poder leer datos de cualquier fuente soportada
        (archivos o bases de datos) con detección automática del tipo de fuente
        y sistema de cache para optimizar lecturas repetidas.
        """
        try:
            # Esta sección se crea para poder detectar automáticamente el tipo de fuente
            if source_type == "auto":
                if isinstance(source, (str, Path)) and Path(source).exists():
                    source_type = "file"
                elif isinstance(source, str) and ":" in source and source.split(":")[0] in self.database_handlers:
                    source_type = "database"
                else:
                    source_type = "file"  # Default a archivo
            
            # Esta clave se crea para poder identificar únicamente cada operación de lectura
            cache_key = f"{source_type}:{source}"
            
            # Esta verificación se crea para poder retornar datos del cache si ya existen
            if cache_key in self._data_cache:
                logger.info(f"Returning cached data for: {cache_key}")
                return self._data_cache[cache_key]
            
            # Estas condiciones se crean para poder dirigir la lectura al manejador apropiado
            if source_type == "file":
                df = self.file_handler.read(source, **kwargs)
            elif source_type == "database":
                # Esta sección se crea para poder manejar el formato "conexion:tabla"
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
            
            # Esta línea se crea para poder cachear solo lecturas exitosas
            if df is not None:
                self._data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading data from {source}: {e}")
            return None
    
    def write_data(self, df: pl.DataFrame, destination: Union[str, Path], 
                   destination_type: str = "auto", **kwargs) -> bool:
        """
        Este método se crea para poder escribir DataFrames a cualquier destino
        soportado (archivos o bases de datos) con detección automática del
        tipo de destino basándose en el formato.
        """
        try:
            # Esta sección se crea para poder detectar automáticamente el tipo de destino
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
        Este método se crea para poder obtener información resumida y útil
        sobre un DataFrame, facilitando el análisis exploratorio y debugging.
        """
        return {
            "shape": df.shape,  # Dimensiones del DataFrame
            "columns": df.columns,  # Nombres de las columnas
            "dtypes": dict(zip(df.columns, [str(dtype) for dtype in df.dtypes])),  # Tipos de datos
            "memory_usage": df.estimated_size("mb"),  # Uso de memoria en MB
            "null_counts": df.null_count().to_dicts()[0] if not df.is_empty() else {}  # Conteo de valores nulos
        }
    
    def clear_cache(self):
        """
        Este método se crea para poder limpiar el cache de datos cuando sea
        necesario liberar memoria o forzar la re-lectura de archivos modificados.
        """
        self._data_cache.clear()
        logger.info("Data cache cleared")
    
    def list_cached_data(self) -> List[str]:
        """
        Este método se crea para poder listar todas las fuentes de datos
        que están actualmente en cache, útil para debugging y monitoreo.
        """
        return list(self._data_cache.keys())


def main():
    """
    Esta función se crea para poder demostrar el uso completo del sistema
    UnifiedDataHandler, sirviendo como ejemplo ejecutable y documentación
    de las capacidades del sistema.
    """
    # Esta inicialización se crea para poder crear una instancia del manejador principal
    handler = UnifiedDataHandler()
    
    # Este ejemplo se crea para poder mostrar cómo configurar conexiones de BD
    # (comentado porque no tenemos BD real en este ejemplo)
    # handler.add_database_connection(
    #     "main_db", 
    #     DatabaseType.MYSQL, 
    #     DatabaseConfig(host="localhost", username="user", password="pass", database="mydb")
    # )
    
    try:
        # Esta importación se crea para poder usar rutas específicas del proyecto
        from Main_charper.database_pruebas1.componentes_db import rf_components, power_converters
        
        # Esta sección se crea para poder leer y procesar datos de convertidores de potencia
        print("Reading power converters data...")
        power_df = handler.read_data(power_converters)
        if power_df is not None:
            print(f"Power converters shape: {power_df.shape}")
            print(f"Columns: {power_df.columns}")
            print("\nFirst few rows:")
            print(power_df.head())
            
            # Esta llamada se crea para poder obtener información resumida del DataFrame
            info = handler.get_data_info(power_df)
            print(f"\nData info: {info}")
        
        # Esta sección se crea para poder leer y procesar datos de componentes RF
        print("\n" + "="*50)
        print("Reading RF components data...")
        rf_df = handler.read_data(rf_components)
        if rf_df is not None:
            print(f"RF components shape: {rf_df.shape}")
            print(f"Columns: {rf_df.columns}")
            print("\nFirst few rows:")
            print(rf_df.head())
        
        # Este ejemplo se crea para poder demostrar la escritura a formato diferente
        if power_df is not None:
            output_path = Path("output_power_converters.csv")
            if handler.write_data(power_df, output_path):
                print(f"\nSuccessfully wrote data to {output_path}")
        
    except ImportError as e:
        logger.error(f"Could not import component paths: {e}")
        print("Please ensure the component modules are available.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


# Este bloque se crea para poder ejecutar la función main solo cuando el archivo
# se ejecuta directamente, no cuando se importa como módulo
if __name__ == "__main__":
    main()