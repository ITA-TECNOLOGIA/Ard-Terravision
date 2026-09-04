import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

data_path = os.getenv("THARSIS_WATER_DATA_PATH")
input_excel = os.path.join(data_path, "water_sensors.xlsx")
output_csv = os.path.join(data_path, "water_sensors.csv")

sheet_name = 0

df = pd.read_excel(input_excel, sheet_name=sheet_name)

df.columns = (
    df.columns
    .str.strip()
    .str.replace("\n", " ", regex=False)
)

df = df.rename(columns={
    "Fecha": "time",
    "Punto de muestreo": "sensor",
    "Conductividad": "conductivity",
    "Temperatura": "temperature",
    "Lugar": "place",
    "Observaciones": "observations",
    "pH": "pH"
})

df["time"] = pd.to_datetime(df["time"], dayfirst=True, errors="coerce")

df = df.dropna(subset=["time"])
df = df.sort_values(["time", "sensor"])

numeric_cols = ["pH", "conductivity", "temperature"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(output_csv, index=False, encoding="utf-8", float_format="%.6f")

print(f"CSV saved in: {Path(output_csv).resolve()}")
print(df.head())