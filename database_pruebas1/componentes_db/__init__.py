from pathlib import Path

# Define rf_components como la ruta absoluta al archivo rf_components.csv
# Se asume que rf_components.csv está en el mismo directorio que este archivo __init__.py.
rf_components = str(Path(__file__).resolve().parent / "rf_components.csv")
power_converters = str(Path(__file__).resolve().parent / "power_converters.json")