from dashboard import load_and_process_data, find_nc_file
import numpy as np
import xarray as xr

basin = "Amman Zarqua"
# We need to simulate the data loading parts to ensure the logic isn't broken
# But since we don't have real NetCDF files for Amman Zarqua (only structure), the load_and_process_data might fail if files are missing.
# Wait, I saw NetCDF folder earlier?
