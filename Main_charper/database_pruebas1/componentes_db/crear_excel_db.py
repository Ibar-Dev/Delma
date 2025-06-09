import pandas as pd
import openpyxl
import numpy


# Crear datos para antenas
antenas_data = {
    'ID': ['ANT001', 'ANT002', 'ANT003', 'ANT004', 'ANT005'],
    'Tipo': ['Yagi', 'Dipolo', 'Parche', 'Helicoidal', 'Log-periódica'],
    'Frecuencia_MHz': [433, 915, 2400, 5800, 1800],
    'Ganancia_dBi': [12, 2.15, 6, 10, 8],
    'Polarización': ['Lineal', 'Vertical', 'Circular', 'Circular', 'Lineal'],
    'Impedancia_Ohm': [50, 50, 50, 50, 75],
    'Precio_USD': [45.99, 15.99, 29.99, 89.99, 129.99],
    'Stock': [10, 25, 15, 8, 12]
}

# Crear DataFrame y guardar como Excel
df_antenas = pd.DataFrame(antenas_data)
df_antenas.to_excel('antenas_rf.xlsx', index=False)

# Crear datos para conectores RF
conectores_data = {
    'ID': ['CON001', 'CON002', 'CON003', 'CON004'],
    'Tipo': ['SMA', 'N-Type', 'BNC', 'UHF'],
    'Material': ['Oro', 'Níquel', 'Plata', 'Latón'],
    'Impedancia_Ohm': [50, 50, 50, 50],
    'Frecuencia_Max_GHz': [18, 11, 4, 3],
    'Precio_USD': [8.99, 12.99, 5.99, 4.99],
    'Stock': [50, 30, 45, 60]
}

# Crear DataFrame y guardar como Excel
df_conectores = pd.DataFrame(conectores_data)
df_conectores.to_excel('conectores_rf.xlsx', index=False)
