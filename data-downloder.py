import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import cdsapi

# Setup directories
download_path = Path("./data/downloads").expanduser()
raw_path = download_path / "raw_daily"
raw_path.mkdir(parents=True, exist_ok=True)

# comment 1 to 2 to test the stage 3
# CDS API Client
c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="18210a1d-23ff-4a62-b06e-e22ec7f5be0f",  # Replace with your actual key
)

# SWIO Bounding Box: North, West, South, East
swio_area = [0, 20, -45, 120]
time_steps = ["00:00", "06:00", "12:00", "18:00"]

# 1. Download Static Variables (Only needs to be done once)
static_file = download_path / "static.nc"
if not static_file.exists():
    print("Downloading static variables...")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["geopotential", "land_sea_mask"],
            "year": "2020",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
            "area": swio_area,
        },
        str(static_file),
    )
    print("Static variables downloaded!")

## 2. Daily Download Loop for 10 Days
dates = pd.date_range(start="1986-11-30", end="2000-01-01")

for dt in dates:
    day_str = dt.strftime("%Y-%m-%d")
    year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
    
    surf_file = raw_path / f"{day_str}-surface.nc"
    atm_file = raw_path / f"{day_str}-atmospheric.nc"
#    
    # Download daily surface variables
    if not surf_file.exists():
        print(f"Downloading surface data for {day_str}...")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",
                ],
                "year": year,
                "month": month,
                "day": day,
                "time": time_steps,
                "format": "netcdf",
                "area": swio_area,
            },
            str(surf_file),
        )
        
#    # Download daily atmospheric variables
    if not atm_file.exists():
        print(f"Downloading atmospheric data for {day_str}...")
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": ["u_component_of_wind",
                              "v_component_of_wind", 
                              "vorticity",
                              "relative_humidity"],

                "pressure_level": ["200", "250", "300", "400", "500", "600", "700", "850"],
                "year": year,
                "month": month,
                "day": day,
                "time": time_steps,
                "format": "netcdf",
                "area": swio_area,
            },
            str(atm_file),
        )

#print("All daily downloads complete. Starting feature derivation...")

# 3. Concatenate and Compute Derived Features using Xarray
# Open all daily files at once
ds_surf = xr.open_mfdataset(str(raw_path / "*-surface.nc"), combine='by_coords')
ds_atm = xr.open_mfdataset(str(raw_path / "*-atmospheric.nc"), combine='by_coords')

# Standardize 'valid_time' dimension to 'time' if present
if "valid_time" in ds_surf.dims:
    ds_surf = ds_surf.rename({"valid_time": "time"})
if "valid_time" in ds_atm.dims:
    ds_atm = ds_atm.rename({"valid_time": "time"})

# Detect pressure level coordinate name ('pressure_level' vs 'level')
level_coord = "pressure_level" if "pressure_level" in ds_atm.coords else "level"

# Calculate Steering Flow (Mean between 850 hPa and 500 hPa)
steering_levels = [850, 700, 600, 500, 300, 200]
u_steer = ds_atm['u'].sel(**{level_coord: steering_levels}).mean(dim=level_coord)
v_steer = ds_atm['v'].sel(**{level_coord: steering_levels}).mean(dim=level_coord)

# Calculate Vertical Wind Shear (Difference between 200 hPa and 850 hPa)
u_shear = ds_atm['u'].sel(**{level_coord: 200}) - ds_atm['u'].sel(**{level_coord: 850})
v_shear = ds_atm['v'].sel(**{level_coord: 200}) - ds_atm['v'].sel(**{level_coord: 850})

# Create a new dataset for the derived features
ds_derived = xr.Dataset({
    'u_steer': u_steer,
    'v_steer': v_steer,
    'u_shear': u_shear,
    'v_shear': v_shear
})

# Merge surface variables with derived features
ds_final = xr.merge([ds_surf, ds_derived])

# 4. Rename variables to match AuroraLite expectations
# ERA5 default names -> Model names
rename_map = {
    'u10': '10u',
    'v10': '10v',
    't2m': '2t'
    # 'msl' defaults to 'msl' in ERA5, so it requires no change
}
# Safely rename only if the keys exist in the dataset
rename_map = {k: v for k, v in rename_map.items() if k in ds_final.data_vars}
ds_final = ds_final.rename(rename_map)
ds_final = ds_final.assign(
    vort850=ds_atm['vo'].sel(**{level_coord: 850}),
    rh700=ds_atm['r'].sel(**{level_coord: 700}),
)

# 5. Save the final combined dataset
output_file = download_path / "swio_10days_surface_expanded.nc"
ds_final.to_netcdf(output_file)

print(f"Data successfully processed, renamed, and saved to {output_file}")
