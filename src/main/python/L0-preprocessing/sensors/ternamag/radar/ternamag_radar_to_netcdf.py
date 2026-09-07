from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import glob


def matlab_datenum_to_datetime(datenum_series):
    return pd.to_datetime(datenum_series, unit="D", origin="1899-12-30")


def read_sensor_csv(file):
    file = Path(file)

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]

    sensor_name = None
    for line in lines:
        if line.startswith("Time;"):
            sensor_name = line.split(";", 1)[1].strip()
            break
    if sensor_name is None:
        sensor_name = file.stem

    x_coord = float([l for l in lines if l.startswith("X")][0].split(";")[1])
    y_coord = float([l for l in lines if l.startswith("Y")][0].split(";")[1])
    z_coord = float([l for l in lines if l.startswith("Z")][0].split(";")[1])

    start_idx = None
    for i, line in enumerate(lines):
        if ";" in line:
            left, right = line.split(";", 1)
            try:
                float(left)
                float(right)
                start_idx = i
                break
            except:
                pass

    if start_idx is None:
        raise ValueError(f"No numerical data found in {file}")

    data_lines = lines[start_idx:]

    df = pd.DataFrame(
        [l.split(";") for l in data_lines],
        columns=["time", "displacement"]
    )

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["displacement"] = pd.to_numeric(df["displacement"], errors="coerce")
    df = df.dropna()

    df["time"] = matlab_datenum_to_datetime(df["time"])

    return sensor_name, x_coord, y_coord, z_coord, df


def csvs_to_datacube(input_path, output_nc, pattern="*.csv"):
    input_path = Path(input_path)
    output_nc = Path(output_nc)

    if input_path.is_dir():
        csv_files = sorted(glob.glob(str(input_path / pattern)))
    else:
        raise ValueError("input_path must be a folder containing CSVs")

    if len(csv_files) == 0:
        raise ValueError("CSV files not found.")

    sensor_data = {}
    sensor_coords = {}

    all_times = set()

    for csv in csv_files:
        sensor, x, y, z, df = read_sensor_csv(csv)
        sensor_data[sensor] = df
        sensor_coords[sensor] = (x, y, z)
        all_times.update(df["time"].values)

    times = np.array(sorted(all_times), dtype="datetime64[ns]")
    sensors = sorted(sensor_data.keys())

    data = np.full((len(times), len(sensors)), np.nan, dtype=float)

    time_index = {t: i for i, t in enumerate(times)}

    for j, sensor in enumerate(sensors):
        df = sensor_data[sensor]
        for t, val in zip(df["time"].values.astype("datetime64[ns]"), df["displacement"].values):
            i = time_index[t]
            data[i, j] = val

    xs = np.array([sensor_coords[s][0] for s in sensors], dtype=float)
    ys = np.array([sensor_coords[s][1] for s in sensors], dtype=float)
    zs = np.array([sensor_coords[s][2] for s in sensors], dtype=float)

    ds = xr.Dataset(
        data_vars={
            "displacement": (("time", "sensor"), data),
            "time_str": ("time", pd.to_datetime(times).strftime("%d/%m/%Y  %H:%M:%S").astype(str))
        },
        coords={
            "time": times,
            "sensor": sensors,
            "sensor_x": ("sensor", xs),
            "sensor_y": ("sensor", ys),
            "sensor_z": ("sensor", zs),
        }
    )

    ds["displacement"].attrs["units"] = "mm"
    ds.attrs["description"] = "Datacube generated from sensor CSVs"
    ds.attrs["time_format"] = "dd/mm/YYYY  HH:MM:SS"

    ds.to_netcdf(output_nc)

    print(f"Datacube saved in: {output_nc}")
    return ds


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    data_path = os.getenv("TERNAMAG_RADAR_DATA_PATH")
    output_file = os.path.join(data_path, "ternamag_radar.nc")

    csvs_to_datacube(data_path, output_file)