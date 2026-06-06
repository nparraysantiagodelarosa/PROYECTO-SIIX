import pandas as pd
import numpy as np
from pathlib import Path
import locale
import datetime

locale.setlocale(locale.LC_TIME, 'English_United States.1252')

meses_es_en = {
    'Ene': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Abr': 'Apr',
    'May': 'May', 'Jun': 'Jun', 'Jul': 'Jul', 'Ago': 'Aug',
    'Sep': 'Sep', 'Oct': 'Oct', 'Nov': 'Nov', 'Dic': 'Dec'
}

def traducir_y_convertir(fecha):
    if pd.isna(fecha) or isinstance(fecha, pd.Timestamp):
        return fecha
    f_str = str(fecha)
    for mes_es, mes_en in meses_es_en.items():
        if mes_es in f_str:
            f_str = f_str.replace(mes_es, mes_en)
            break
    return pd.to_datetime(f_str, format='%d/%b/%y', errors='coerce', dayfirst=True)

def time_to_minutes(t):
    if isinstance(t, datetime.time):
        return int(t.hour * 60 + t.minute + t.second / 60)
    return None

ruta_base_nueva = Path(r"C:\Users\marma\OneDrive\Escritorio\basenueva.xlsx")
ruta_base_antigua = Path(r"C:\Users\marma\OneDrive\Escritorio\HC proyecto.xlsx")

df_old = pd.read_excel(ruta_base_antigua)
df_new = pd.read_excel(ruta_base_nueva)

df_old = df_old.rename(columns={'Número': 'IDENTIFIER', 'NÓMINA': 'IDENTIFIER'})
df_new = df_new.rename(columns={'Número': 'IDENTIFIER', 'NÓMINA': 'IDENTIFIER'})

df_merged = pd.merge(df_new, df_old, on="IDENTIFIER")

if "TIEMPO DE TRASLADO A SIIX" in df_merged.columns:
    df_merged["TIEMPO DE TRASLADO A SIIX MINUTOS"] = df_merged["TIEMPO DE TRASLADO A SIIX"].apply(time_to_minutes)

if "Nacimiento" in df_merged.columns:
    df_merged['Nacimiento'] = pd.to_datetime(df_merged['Nacimiento'], format='%d/%m/%Y', errors='coerce', dayfirst=True)

for col_fecha in ['Fecha Ingreso', 'Baja RH']:
    if col_fecha in df_merged.columns:
        df_merged[col_fecha] = df_merged[col_fecha].apply(traducir_y_convertir)

if 'Fecha Ingreso' in df_merged.columns and 'Nacimiento' in df_merged.columns:
    df_merged["EDAD EN FECHA DE INGRESO"] = np.floor((df_merged['Fecha Ingreso'] - df_merged['Nacimiento']).dt.days / 365)

if 'Baja RH' in df_merged.columns and 'Fecha Ingreso' in df_merged.columns:
    df_merged["TARGET"] = (df_merged['Baja RH'] - df_merged['Fecha Ingreso']).dt.days

columnas_a_eliminar = [
    "IDENTIFIER", "Status", "Puesto", "Fecha Ingreso", "Tipo de Contrato",
    "Fecha de término", "Baja RH", "Antigüedad", "Nacimiento", "Edad",
    "Género", "Estado Civil", "Estudios", "Ruta Transporte", "Hijos",
    "No.", "TIEMPO DE TRASLADO A SIIX", "EDAD", "No..1", "AP. PATERNO ",
    "AP. MATERNO", "NOMBRE ", "Unnamed: 35", "Unnamed: 36", "ANTIGÜEDAD",
    "Unnamed: 34", "Unnamed: 30", "Unnamed: 31", "Departamento",
    "FECHA DE INGRESO", "FECHA DE BAJA", "¿CUÁNTO?", "FONACOT ", "¿CUÁNTO? ",
    "Número", "FECHA BAJA RH", "TIEMPO EXTRA", "ACIERTOS EVALUACION DE CONOCIMIENTOS"
]

columnas_presentes = [col for col in columnas_a_eliminar if col in df_merged.columns]
df_cl = df_merged.drop(columns=columnas_presentes)

df_cl['SALARIO DIARIO ANTERIOR '] = df_cl['SALARIO DIARIO ANTERIOR '].replace({'PRIMER EMPLEO ': 0})

for col in df_cl.columns:
    missing_count = df_cl[col].isna().sum()
    print(f"{col} --- Datos Faltantes: {missing_count}")

df_cl["TIEMPO EN LA INDUSTRIA"] = df_cl["TIEMPO EN LA INDUSTRIA"].fillna("0 MESES")

columnas_con_pocos_nulos = df_cl.columns[df_cl.isna().sum() < 10]
df_cl = df_cl.dropna(subset=columnas_con_pocos_nulos)

df_cl.columns = [col.strip() for col in df_cl.columns]

valores_por_defecto = {
    "SEXO": "DESCONOCIDO",
    "TIEMPO DE TRASLADO A SIIX": -1,
    "TIEMPO DE CASA A TRASLADO A PARADA": "0:00",
    "TIEMPO EN LA INDUSTRIA": "0 MESES",
    "CUANTO DURO EN SU ULTIMO EMPLEO": "0 MESES",
    "ROTACION DE TRABAJOS EN 1 AÑO": "NO",
    "CANTIDAD": 0,
    "A QUE SE DEDICABA EN SU ULTIMO EMPLEO ": "DESCONOCIDO",
    "DE QUE TIPO DE GIRO ERA SU ULTIMO TRABAJO": "DESCONOCIDO",
    "MAYORES DE 3 AÑOS": "NO",
    "MENORES DE 3 AÑOS ": "NO"
}

for columna, valor in valores_por_defecto.items():
    if columna in df_cl.columns:
        df_cl[columna] = df_cl[columna].fillna(valor)

df_cl = df_cl.rename(columns={
    "MAYORES DE 3 AÑOS": "MAYORES DE 3",
    "MENORES DE 3 AÑOS": "MENORES DE 3",
    "ROTACION DE TRABAJOS EN 1 AÑO": "ROTACION DE TRABAJOS EN 12 MESES"
})

ca = df_cl["CANTIDAD"]

for col in df_cl.select_dtypes(include=['object']).columns:
    df_cl[col] = df_cl[col].str.strip()

df_cl["CANTIDAD"] = ca
df_cl.loc[df_cl["ROTACION DE TRABAJOS EN 12 MESES"] == "N", "ROTACION DE TRABAJOS EN 12 MESES"] = "NO"
df_cl["A QUE SE DEDICABA EN SU ULTIMO EMPLEO"] = df_cl["A QUE SE DEDICABA EN SU ULTIMO EMPLEO"].fillna("PRIMER EMPLEO")

salida = Path(Path(ruta_base_nueva).parent, "dtbasenueva4.xlsx")
df_cl.to_excel(salida, index=False)
print(f"\n\n\nArchivo guardado en: {salida}")