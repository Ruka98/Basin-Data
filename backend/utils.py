
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import fiona
from shapely.geometry import shape as shp_shape, mapping
from shapely import wkb as shp_wkb
import plotly.express as px
import plotly.graph_objects as go
import json

# ======================
# FILE / PATH UTILITIES
# ======================

BASE_DIR = os.getcwd()
BASIN_DIR = os.path.join(BASE_DIR, "basins")

def _first_existing(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort()
            return hits[-1]
    return None

def find_nc_file(basin_name: str, variable_type: str):
    """Find a representative NetCDF per variable type in a basin folder."""
    netcdf_dir = os.path.join(BASIN_DIR, basin_name, "NetCDF")
    if not os.path.isdir(netcdf_dir):
        return None
    if variable_type == "P":
        pats = [os.path.join(netcdf_dir, "*_P_*.nc"), os.path.join(netcdf_dir, "*P*.nc")]
    elif variable_type == "ET":
        pats = [os.path.join(netcdf_dir, "*_ETa_*.nc"), os.path.join(netcdf_dir, "*_ET_*.nc"), os.path.join(netcdf_dir, "*ET*.nc")]
    elif variable_type == "LU":
        pats = [os.path.join(netcdf_dir, "*_LU_*.nc"), os.path.join(netcdf_dir, "*LandUse*.nc"), os.path.join(netcdf_dir, "*LU*.nc")]
    else:
        return None
    return _first_existing(pats)

def find_shp_file(basin_name: str):
    shp_dir = os.path.join(BASIN_DIR, basin_name, "Shapefile")
    if not os.path.isdir(shp_dir):
        return None
    return _first_existing([os.path.join(shp_dir, "*.shp")])

def read_basin_intro(basin_name: str) -> str:
    """Read the introduction text for a basin."""
    intro_path = os.path.join(BASIN_DIR, basin_name, "intro.txt")
    if os.path.exists(intro_path):
        try:
            with open(intro_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading intro file: {e}")
    return ""

def find_yearly_csv(basin_name: str, year: int):
    """Find yearly CSV file for a basin and year."""
    results_dir = os.path.join(BASIN_DIR, basin_name, "Results", "yearly")
    if not os.path.isdir(results_dir):
        return None

    patterns = [
        os.path.join(results_dir, f"sheet1_{year}.csv"),
        os.path.join(results_dir, f"*{year}*.csv"),
        os.path.join(results_dir, "*.csv")  # Fallback to any CSV
    ]

    return _first_existing(patterns)

# =========================
# XARRAY / NETCDF UTILITIES
# =========================

def _open_xr_dataset(fp: str) -> xr.Dataset:
    """Open NetCDF with engine fallback to avoid backend errors."""
    for eng in ("h5netcdf", "netcdf4", None):
        try:
            return xr.open_dataset(fp, decode_times=True, engine=eng)
        except Exception:
            pass
    raise RuntimeError(f"Failed to open dataset with available engines: {fp}")

def _standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Normalize latitude/longitude names to 'latitude'/'longitude' and ensure ascending latitude."""
    lat_names = ["latitude", "lat", "y"]
    lon_names = ["longitude", "lon", "x"]
    lat = next((n for n in lat_names if n in ds.coords or n in ds.variables), None)
    lon = next((n for n in lon_names if n in ds.coords or n in ds.variables), None)
    if lat and lat != "latitude":
        ds = ds.rename({lat: "latitude"})
    if lon and lon != "longitude":
        ds = ds.rename({lon: "longitude"})
    if "latitude" in ds.dims:
        lat_vals = ds["latitude"].values
        if lat_vals.size > 1 and lat_vals[1] < lat_vals[0]:
            ds = ds.sortby("latitude")
    return ds

def _pick_data_var(ds: xr.Dataset):
    """Pick the first 2D/3D field with latitude/longitude dims."""
    exclude = {"time", "latitude", "longitude", "crs", "spatial_ref"}
    cands = [v for v in ds.data_vars if v not in exclude]
    if not cands:
        return None
    with_ll = [v for v in cands if {"latitude", "longitude"}.issubset(set(ds[v].dims))]
    return with_ll[0] if with_ll else cands[0]

def _compute_mode(arr, axis=None):
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)] if counts.size else np.nan

def _coarsen_to_1km(da: xr.DataArray, is_categorical=False) -> xr.DataArray:
    if "latitude" not in da.dims or "longitude" not in da.dims:
        return da
    lat_vals, lon_vals = da["latitude"].values, da["longitude"].values
    lat_res = float(np.abs(np.diff(lat_vals)).mean()) if lat_vals.size > 1 else 0.009
    lon_res = float(np.abs(np.diff(lon_vals)).mean()) if lon_vals.size > 1 else 0.009
    target_deg = 1.0 / 111.0
    f_lat = max(1, int(round(target_deg / (lat_res if lat_res else target_deg))))
    f_lon = max(1, int(round(target_deg / (lon_res if lon_res else target_deg))))
    coarsen_dict = {"latitude": f_lat, "longitude": f_lon}

    if is_categorical:
        try:
            return da.coarsen(coarsen_dict, boundary="trim").reduce(_compute_mode)
        except Exception:
            return da
    else:
        try:
            return da.coarsen(coarsen_dict, boundary="trim").mean(skipna=True)
        except Exception:
            return da

def load_and_process_data(basin_name: str, variable_type: str,
                          year_start: int | None = None, year_end: int | None = None,
                          aggregate_time: bool = True):
    fp = find_nc_file(basin_name, variable_type)
    if not fp:
        return None, None, "NetCDF file not found"
    try:
        ds = _open_xr_dataset(fp)
        ds = _standardize_latlon(ds)
        var = _pick_data_var(ds)
        if not var:
            return None, None, "No suitable data variable in file"

        da = ds[var]

        if "time" in ds.coords and (year_start is not None or year_end is not None):
            ys = int(year_start) if year_start is not None else pd.to_datetime(ds["time"].values).min().year
            ye = int(year_end)   if year_end   is not None else pd.to_datetime(ds["time"].values).max().year
            da = da.sel(time=slice(f"{ys}-01-01", f"{ye}-12-31"))

        if "time" in da.dims:
            if aggregate_time and da.sizes.get("time", 0) > 1 and variable_type in ["P", "ET"]:
                da = da.mean(dim="time", skipna=True)
            elif variable_type == "LU" and da.sizes.get("time", 0) > 0:
                da = da.isel(time=-1)
            elif not aggregate_time:
                pass
            else:
                da = da.isel(time=0)

        da = _coarsen_to_1km(da, is_categorical=(variable_type == "LU"))
        return da, var, os.path.basename(fp)

    except Exception as e:
        return None, None, f"Error processing file: {e}"

# ===================
# SHAPEFILE UTILITIES
# ===================

def _force_2d(geom):
    try:
        return shp_wkb.loads(shp_wkb.dumps(geom, output_dimension=2))
    except Exception:
        return geom

def _repair_poly(geom):
    try:
        g = geom.buffer(0)
        return g if (g is not None and not g.is_empty) else geom
    except Exception:
        return geom

def load_all_basins_geodata() -> gpd.GeoDataFrame:
    """Load ALL basins' shapefiles (exploded, fixed, EPSG:4326)."""
    rows = []
    if not os.path.isdir(BASIN_DIR):
        return gpd.GeoDataFrame(columns=["basin", "geometry"], geometry="geometry", crs="EPSG:4326")

    for b in sorted([d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))]):
        shp = find_shp_file(b)
        if not shp or not os.path.exists(shp):
            continue
        try:
            with fiona.open(shp) as src:
                crs_wkt = src.crs_wkt
                crs_obj = None
                if crs_wkt:
                    try:
                        crs_obj = gpd.GeoSeries([0], crs=crs_wkt).crs
                    except Exception:
                        crs_obj = None

                geoms = []
                for feat in src:
                    if not feat or not feat.get("geometry"):
                        continue
                    geom = shp_shape(feat["geometry"])
                    geom = _force_2d(geom)
                    geom = _repair_poly(geom)
                    if geom and not geom.is_empty and geom.geom_type in ("Polygon", "MultiPolygon"):
                        geoms.append(geom)
                if not geoms:
                    continue

                gdf = gpd.GeoDataFrame({"basin": [b]*len(geoms)}, geometry=geoms, crs=crs_obj or "EPSG:4326")
                try:
                    gdf = gdf.to_crs("EPSG:4326")
                except Exception:
                    gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)

                try:
                    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
                except Exception:
                    gdf = gdf.explode().reset_index(drop=True)

                gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
                rows.append(gdf[["basin", "geometry"]])
        except Exception as e:
            print(f"[WARN] Problem with {b}: {e}")
            continue

    if not rows:
        return gpd.GeoDataFrame(columns=["basin", "geometry"], geometry="geometry", crs="EPSG:4326")

    return gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")

ALL_BASINS_GDF = load_all_basins_geodata()

def basins_geojson(gdf: gpd.GeoDataFrame | None = None):
    gdf = ALL_BASINS_GDF if gdf is None else gdf
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            feats.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {"basin": row["basin"]},
                }
            )
        except Exception as e:
            print(f"[WARN] Could not convert geometry for basin {row['basin']}: {e}")
    return {"type": "FeatureCollection", "features": feats}

def parse_basin_overview(csv_file: str):
    """Parse CSV file and extract key basin overview metrics."""
    try:
        df = pd.read_csv(csv_file, sep=';')

        # Initialize metrics dictionary
        metrics = {}

        # Extract key metrics based on the CSV structure
        for _, row in df.iterrows():
            class_val = row.get('CLASS', '')
            subclass = row.get('SUBCLASS', '')
            variable = row.get('VARIABLE', '')
            value = row.get('VALUE', 0)

            # Skip empty values
            if pd.isna(value) or value == '':
                continue

            try:
                # Multiply by 1000 to convert to Mm³/year
                value = float(value) * 1000
            except (ValueError, TypeError):
                continue

            # Total inflows (sum of all INFLOW)
            if class_val == 'INFLOW':
                metrics['total_inflows'] = metrics.get('total_inflows', 0) + value

            # Precipitation components
            if class_val == 'INFLOW' and subclass == 'PRECIPITATION':
                if variable == 'Rainfall':
                    metrics['precipitation_rainfall'] = value
                elif variable == 'Snowfall':
                    metrics['precipitation_snowfall'] = value
                metrics['total_precipitation'] = metrics.get('total_precipitation', 0) + value

            # Surface water imports
            if (class_val == 'INFLOW' and subclass == 'SURFACE WATER' and
                variable in ['Main riverstem', 'Tributaries']):
                metrics['surface_water_imports'] = metrics.get('surface_water_imports', 0) + value

            # Total landscape water consumption (sum of all ET)
            if class_val == 'OUTFLOW' and 'ET' in subclass:
                # Exclude Manmade and Consumed Water to avoid double counting or mismatch with user expectations
                if variable not in ['Manmade', 'Consumed Water']:
                    metrics['total_water_consumption'] = metrics.get('total_water_consumption', 0) + value

            # Manmade water consumption
            if (class_val == 'OUTFLOW' and subclass == 'ET INCREMENTAL' and
                variable == 'Manmade'):
                metrics['manmade_consumption'] = value

            # Treated wastewater
            if (class_val == 'OUTFLOW' and subclass == 'OTHER' and
                variable == 'Treated Waste Water'):
                metrics['treated_wastewater'] = value

            # Non-irrigated water consumption
            if (class_val == 'OUTFLOW' and subclass == 'ET INCREMENTAL' and
                variable == 'Consumed Water'):
                metrics['non_irrigated_consumption'] = value

            # Recharge
            if (class_val == 'STORAGE' and subclass == 'CHANGE' and
                'Surface storage' in variable):
                metrics['recharge'] = abs(value) if value < 0 else value

        # Calculate percentages
        if metrics.get('total_inflows', 0) > 0:
            metrics['precipitation_percentage'] = (metrics.get('total_precipitation', 0) /
                                                 metrics['total_inflows'] * 100)

        return metrics

    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return {}

def get_basin_overview_metrics_for_range(basin_name: str, start_year: int, end_year: int):
    """Get comprehensive basin overview metrics averaged over a year range."""
    all_metrics = []

    for year in range(start_year, end_year + 1):
        csv_file = find_yearly_csv(basin_name, year)
        if csv_file:
            metrics = parse_basin_overview(csv_file)
            if metrics:
                all_metrics.append(metrics)

    if not all_metrics:
        return None

    # Calculate average metrics across all years
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m.get(key, 0) for m in all_metrics]
        avg_metrics[key] = np.mean(values)

    return avg_metrics


# ==================
# FIGURE CONSTRUCTORS
# ==================

def _clean_nan_data(da: xr.DataArray):
    """Remove NaN values and return clean data for plotting"""
    if da is None:
        return None, None, None

    # Create mask for valid (non-NaN) data
    valid_mask = np.isfinite(da.values)

    if not np.any(valid_mask):
        return None, None, None

    # Get coordinates
    x = np.asarray(da["longitude"].values)
    y = np.asarray(da["latitude"].values)

    # For imshow, we need to handle the entire grid but mask NaN areas
    z_clean = da.values.copy()

    return z_clean, x, y

def _create_clean_heatmap(da: xr.DataArray, title: str, colorscale="Viridis", z_label="value"):
    """Create a clean heatmap that properly handles NaN values"""
    if da is None or "latitude" not in da.coords or "longitude" not in da.coords:
        return _empty_fig("No data to display")

    z, x, y = _clean_nan_data(da)
    if z is None:
        return _empty_fig("No valid data values")

    # Create figure using go.Heatmap for better NaN handling
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        zmid=0,  # Center colorscale at 0 for diverging data
        colorbar=dict(
            title=z_label,
            thickness=15,
            len=0.75,
            yanchor="middle",
            y=0.5
        ),
        hoverinfo="x+y+z",
        hovertemplate='Longitude: %{x:.2f}<br>Latitude: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="#1e293b"),
        margin=dict(l=50, r=50, t=60, b=50)
    )

    return fig

def add_shapefile_to_fig(fig: go.Figure, basin_name: str) -> go.Figure:
    """Overlay basin boundary on a cartesian image figure."""
    shp_file = find_shp_file(basin_name)
    if not shp_file or not os.path.exists(shp_file):
        return fig
    try:
        gdf = gpd.read_file(shp_file)
        try:
            gdf = gdf.to_crs("EPSG:4326")
        except Exception:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        for geom in gdf.geometry:
            geom = _repair_poly(_force_2d(geom))
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                x, y = geom.exterior.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines",
                                         line=dict(color="black", width=1),  # Changed to solid black thin line
                                         name="Basin Boundary",
                                         showlegend=False,
                                         hoverinfo='skip'))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines",
                                             line=dict(color="black", width=1),  # Changed to solid black thin line
                                             name="Basin Boundary",
                                             showlegend=False,
                                             hoverinfo='skip'))
    except Exception as e:
        print(f"[WARN] Could not overlay shapefile: {e}")
    return fig

def _empty_fig(msg="No data to display"):
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": msg, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}],
        margin=dict(l=0, r=0, t=35, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

# Land use class information
class_info = {
    1: {"name": "Protected forests", "color": "rgb(0,40,0)"},
    2: {"name": "Protected shrubland", "color": "rgb(190,180,60)"},
    3: {"name": "Protected natural grasslands", "color": "rgb(176,255,33)"},
    4: {"name": "Protected natural waterbodies", "color": "rgb(83,142,213)"},
    5: {"name": "Protected wetlands", "color": "rgb(40,250,180)"},
    6: {"name": "Glaciers", "color": "rgb(255,255,255)"},
    7: {"name": "Protected other", "color": "rgb(219,214,0)"},
    8: {"name": "Closed deciduous forest", "color": "rgb(0,70,0)"},
    9: {"name": "Open deciduous forest", "color": "rgb(0,124,0)"},
    10: {"name": "Closed evergreen forest", "color": "rgb(0,100,0)"},
    11: {"name": "Open evergreen forest", "color": "rgb(0,140,0)"},
    12: {"name": "Closed savanna", "color": "rgb(155,150,50)"},
    13: {"name": "Open savanna", "color": "rgb(255,190,90)"},
    14: {"name": "Shrub land & mesquite", "color": "rgb(120,150,30)"},
    15: {"name": "Herbaceous cover", "color": "rgb(90,115,25)"},
    16: {"name": "Meadows & open grassland", "color": "rgb(140,190,100)"},
    17: {"name": "Riparian corridor", "color": "rgb(30,190,170)"},
    18: {"name": "Deserts", "color": "rgb(245,255,230)"},
    19: {"name": "Wadis", "color": "rgb(200,230,255)"},
    20: {"name": "Natural alpine pastures", "color": "rgb(86,134,0)"},
    21: {"name": "Rocks & gravel & stones & boulders", "color": "rgb(255,210,110)"},
    22: {"name": "Permafrosts", "color": "rgb(230,230,230)"},
    23: {"name": "Brooks & rivers & waterfalls", "color": "rgb(0,100,240)"},
    24: {"name": "Natural lakes", "color": "rgb(0,55,154)"},
    25: {"name": "Flood plains & mudflats", "color": "rgb(165,230,100)"},
    26: {"name": "Saline sinks & playas & salinized soil", "color": "rgb(210,230,210)"},
    27: {"name": "Bare soil", "color": "rgb(240,165,20)"},
    28: {"name": "Waste land", "color": "rgb(230,220,210)"},
    29: {"name": "Moorland", "color": "rgb(190,160,140)"},
    30: {"name": "Wetland", "color": "rgb(33,193,132)"},
    31: {"name": "Mangroves", "color": "rgb(28,164,112)"},
    32: {"name": "Alien invasive species", "color": "rgb(100,255,150)"},
    33: {"name": "Rainfed forest plantations", "color": "rgb(245,250,194)"},
    34: {"name": "Rainfed production pastures", "color": "rgb(237,246,152)"},
    35: {"name": "Rainfed crops - cereals", "color": "rgb(226,240,90)"},
    36: {"name": "Rainfed crops - root/tuber", "color": "rgb(209,229,21)"},
    37: {"name": "Rainfed crops - legumious", "color": "rgb(182,199,19)"},
    38: {"name": "Rainfed crops - sugar", "color": "rgb(151,165,15)"},
    39: {"name": "Rainfed crops - fruit and nuts", "color": "rgb(132,144,14)"},
    40: {"name": "Rainfed crops - vegetables and melons", "color": "rgb(112,122,12)"},
    41: {"name": "Rainfed crops - oilseed", "color": "rgb(92,101,11)"},
    42: {"name": "Rainfed crops - beverage and spice", "color": "rgb(71,80,8)"},
    43: {"name": "Rainfed crops - other", "color": "rgb(51,57,5)"},
    44: {"name": "Mixed species agro-forestry", "color": "rgb(80,190,40)"},
    45: {"name": "Fallow & idle land", "color": "rgb(180,160,180)"},
    46: {"name": "Dump sites & deposits", "color": "rgb(145,130,115)"},
    47: {"name": "Rainfed homesteads and gardens (urban cities) - outdoor", "color": "rgb(120,5,25)"},
    48: {"name": "Rainfed homesteads and gardens (rural villages) - outdoor", "color": "rgb(210,10,40)"},
    49: {"name": "Rainfed industry parks - outdoor", "color": "rgb(255,130,45)"},
    50: {"name": "Rainfed parks (leisure & sports)", "color": "rgb(250,101,0)"},
    51: {"name": "Rural paved surfaces (lots, roads, lanes)", "color": "rgb(255,150,150)"},
    52: {"name": "Irrigated forest plantations", "color": "rgb(179,243,241)"},
    53: {"name": "Irrigated production pastures", "color": "rgb(158,240,238)"},
    54: {"name": "Irrigated crops - cereals", "color": "rgb(113,233,230)"},
    55: {"name": "Irrigated crops - root/tubers", "color": "rgb(82,228,225)"},
    56: {"name": "Irrigated crops - legumious", "color": "rgb(53,223,219)"},
    57: {"name": "Irrigated crops - sugar", "color": "rgb(33,205,201)"},
    58: {"name": "Irrigated crops - fruit and nuts", "color": "rgb(29,179,175)"},
    59: {"name": "Irrigated crops - vegetables and melons", "color": "rgb(25,151,148)"},
    60: {"name": "Irrigated crops - Oilseed", "color": "rgb(21,125,123)"},
    61: {"name": "Irrigated crops - beverage and spice", "color": "rgb(17,101,99)"},
    62: {"name": "Irrigated crops - other", "color": "rgb(13,75,74)"},
    63: {"name": "Managed water bodies (reservoirs, canals, harbors, tanks)", "color": "rgb(0,40,112)"},
    64: {"name": "Greenhouses - indoor", "color": "rgb(255,204,255)"},
    65: {"name": "Aquaculture", "color": "rgb(47,121,255)"},
    66: {"name": "Domestic households - indoor (sanitation)", "color": "rgb(255,60,10)"},
    67: {"name": "Manufacturing & commercial industry - indoor", "color": "rgb(180,180,180)"},
    68: {"name": "Irrigated homesteads and gardens (urban cities) - outdoor", "color": "rgb(255,139,255)"},
    69: {"name": "Irrigated homesteads and gardens (rural villages) - outdoor", "color": "rgb(255,75,255)"},
    70: {"name": "Irrigated industry parks - outdoor", "color": "rgb(140,140,140)"},
    71: {"name": "Irrigated parks (leisure, sports)", "color": "rgb(150,0,205)"},
    72: {"name": "Urban paved Surface (lots, roads, lanes)", "color": "rgb(120,120,120)"},
    73: {"name": "Livestock and domestic husbandry", "color": "rgb(180,130,130)"},
    74: {"name": "Managed wetlands & swamps", "color": "rgb(30,130,115)"},
    75: {"name": "Managed other inundation areas", "color": "rgb(20,150,130)"},
    76: {"name": "Mining/ quarry & shale exploiration", "color": "rgb(100,100,100)"},
    77: {"name": "Evaporation ponds", "color": "rgb(30,90,130)"},
    78: {"name": "Waste water treatment plants", "color": "rgb(60,60,60)"},
    79: {"name": "Hydropower plants", "color": "rgb(40,40,40)"},
    80: {"name": "Thermal power plants", "color": "rgb(0,0,0)"},
}
