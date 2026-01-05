import os, glob
import numpy as np
import pandas as pd
import collections

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import xarray as xr
import geopandas as gpd
import fiona
from shapely.geometry import shape as shp_shape, mapping
from shapely import wkb as shp_wkb


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


# ======================
# BASIN OVERVIEW UTILITIES
# ======================

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
print(
    "Basins:",
    ALL_BASINS_GDF["basin"].nunique() if not ALL_BASINS_GDF.empty else 0,
    "| Features:",
    len(ALL_BASINS_GDF) if not ALL_BASINS_GDF.empty else 0,
    "| CRS:",
    ALL_BASINS_GDF.crs,
)


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


# ==============
# DATA PIPELINE
# ==============

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

def _safe_imshow(da: xr.DataArray, title: str, colorscale="Viridis", z_label="value"):
    """Robust imshow that handles NaNs and locks aspect ratio; returns placeholder if empty."""
    if da is None or "latitude" not in da.coords or "longitude" not in da.coords:
        return _empty_fig("No data to display")

    z, x, y = _clean_nan_data(da)
    if z is None:
        return _empty_fig("No valid data values")

    # Create figure with masked array to hide NaN areas
    z_masked = np.ma.masked_invalid(z)
    
    fig = px.imshow(
        z_masked, x=x, y=y, origin="lower", aspect="equal",
        color_continuous_scale=colorscale, title=title, labels={"color": z_label}
    )
    
    # Update layout for better appearance
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#1e293b"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Improve colorbar
    fig.update_coloraxes(
        colorbar=dict(
            thickness=15,
            len=0.75,
            yanchor="middle",
            y=0.5,
            x=1.02
        )
    )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

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


# =========================
# BASIN SELECTOR (MAPBOX)
# =========================

def make_basin_selector_map(selected_basin=None) -> go.Figure:
    gdf = ALL_BASINS_GDF if (not selected_basin or selected_basin == "all") else ALL_BASINS_GDF[ALL_BASINS_GDF["basin"] == selected_basin]
    if gdf is None or gdf.empty:
        return _empty_fig("No basin shapefiles found.")

    gj = basins_geojson(gdf)
    locations = [f["properties"]["basin"] for f in gj["features"]]
    z_vals = [1] * len(locations)

    ch = go.Choroplethmapbox(
        geojson=gj,
        locations=locations,
        featureidkey="properties.basin",
        z=z_vals,
        colorscale=[[0, "rgba(0, 102, 255, 0.18)"], [1, "rgba(0, 102, 255, 0.18)"]],
        marker=dict(line=dict(width=3 if selected_basin and selected_basin != "all" else 1.8,
                              color="rgb(0, 90, 200)")),
        hovertemplate="%{location}<extra></extra>",
        showscale=False,
    )
    fig = go.Figure(ch)

    # center/zoom from bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = (maxx - minx) * 0.08 if maxx > minx else 0.1
    pad_y = (maxy - miny) * 0.08 if maxy > miny else 0.1
    west, east = float(minx - pad_x), float(maxx + pad_x)
    south, north = float(miny - pad_y), float(maxy + pad_y)

    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    span_lon = max(east - west, 0.001)
    span_lat = max(north - south, 0.001)

    import math
    map_w, map_h = 900.0, 600.0
    lon_zoom = math.log2(360.0 / (span_lon * 1.1)) + math.log2(map_w / 512.0)
    lat_zoom = math.log2(180.0 / (span_lat * 1.1)) + math.log2(map_h / 512.0)
    zoom = max(0.0, min(16.0, lon_zoom, lat_zoom))

    fig.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lon=center_lon, lat=center_lat), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep",
        clickmode="event+select",
        height=500,
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


# ===========
# DASH LAYOUT
# ===========

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Modern CSS styles
MODERN_STYLES = {
    "container": {
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "20px",
        "fontFamily": "'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "backgroundColor": "#f8fafc",
        "minHeight": "100vh"
    },
    "header": {
        "textAlign": "center",
        "color": "#1e293b",
        "fontSize": "2.5rem",
        "fontWeight": "700",
        "marginBottom": "10px",
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "WebkitBackgroundClip": "text",
        "WebkitTextFillColor": "transparent"
    },
    "subheader": {
        "textAlign": "center",
        "color": "#64748b",
        "fontSize": "1.1rem",
        "marginBottom": "30px",
        "fontWeight": "400"
    },
    "card": {
        "backgroundColor": "white",
        "borderRadius": "12px",
        "padding": "24px",
        "marginBottom": "24px",
        "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
        "border": "1px solid #e2e8f0"
    },
    "section_title": {
        "color": "#1e293b",
        "fontSize": "1.5rem",
        "fontWeight": "600",
        "marginBottom": "20px",
        "borderLeft": "4px solid #3b82f6",
        "paddingLeft": "12px"
    },
    "dropdown": {
        "backgroundColor": "white",
        "border": "1px solid #e2e8f0",
        "borderRadius": "8px"
    },
    "info_box": {
        "backgroundColor": "#f1f5f9",
        "border": "1px solid #cbd5e1",
        "borderRadius": "8px",
        "padding": "12px",
        "marginTop": "12px",
        "fontSize": "14px",
        "color": "#475569"
    },
    "graph_container": {
        "backgroundColor": "white",
        "borderRadius": "8px",
        "padding": "15px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
        "border": "1px solid #e2e8f0"
    }
}

basin_folders = [d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))] if os.path.isdir(BASIN_DIR) else []
basin_options = [{"label": "View All Basins", "value": "all"}] + [{"label": b, "value": b} for b in sorted(basin_folders)]

app.layout = html.Div(
    style=MODERN_STYLES["container"],
    children=[
        # Header Section
        html.Div([
            html.H1("🌊 Basin Data Dashboard", style=MODERN_STYLES["header"]),
            html.P("Interactive visualization of precipitation, evapotranspiration, and land use data across river basins", 
                   style=MODERN_STYLES["subheader"])
        ]),

        # Basin Selection Card
        html.Div(
            style=MODERN_STYLES["card"],
            children=[
                html.H3("📍 Basin Selection", style=MODERN_STYLES["section_title"]),
                
                html.Div([
                    html.Div([
                        html.Label("Select Basin", style={"fontWeight": "600", "marginBottom": "8px", "display": "block"}),
                        dcc.Dropdown(
                            id="basin-dropdown",
                            options=basin_options,
                            value="all" if basin_folders else None,
                            clearable=False,
                            style=MODERN_STYLES["dropdown"]
                        ),
                    ], style={"width": "48%", "display": "inline-block"}),
                    
                    html.Div([
                        html.Label("Analysis Period", style={"fontWeight": "600", "marginBottom": "8px", "display": "block"}),
                        html.Div([
                            html.Div([
                                html.Label("Start Year", style={"fontSize": "14px", "color": "#64748b"}),
                                dcc.Dropdown(
                                    id="global-start-year-dropdown", 
                                    searchable=True, 
                                    clearable=False,
                                    style={**MODERN_STYLES["dropdown"], "width": "100%"}
                                )
                            ], style={"width": "48%", "display": "inline-block", "marginRight": "4%"}),
                            
                            html.Div([
                                html.Label("End Year", style={"fontSize": "14px", "color": "#64748b"}),
                                dcc.Dropdown(
                                    id="global-end-year-dropdown", 
                                    searchable=True, 
                                    clearable=False,
                                    style={**MODERN_STYLES["dropdown"], "width": "100%"}
                                )
                            ], style={"width": "48%", "display": "inline-block"}),
                        ])
                    ], style={"width": "48%", "display": "inline-block", "float": "right"}),
                ]),
                
                # Map
                html.Div([
                    dcc.Graph(
                        id="basin-map", 
                        style={"height": "500px", "borderRadius": "8px"}
                    )
                ], style={"marginTop": "20px"}),
                
                # File Info
                html.Div(
                    id="file-info-feedback",
                    style=MODERN_STYLES["info_box"]
                ),
            ]
        ),

        # Basin Overview Section
        html.Div(
            style=MODERN_STYLES["card"],
            children=[
                html.H3("📊 Basin Overview", style=MODERN_STYLES["section_title"]),
                html.Div(id="basin-overview-content", children=[
                    html.Div("Select a specific basin and year range to view overview metrics.", 
                            style={"textAlign": "center", "color": "#64748b", "padding": "40px"})
                ])
            ]
        ),

        # Data Visualization Sections
        html.Div([
            # Land Use Section
            html.Div(
                style=MODERN_STYLES["card"],
                children=[
                    html.H3("🗺️ Land Use / Land Cover", style=MODERN_STYLES["section_title"]),
                    dcc.Loading(
                        dcc.Graph(id="lu-map-graph"),
                        type="circle"
                    )
                ]
            ),

            # Precipitation Section
            html.Div(
                style=MODERN_STYLES["card"],
                children=[
                    html.H3("🌧️ Precipitation (P)", style=MODERN_STYLES["section_title"]),
                    html.Div([
                        html.Div(
                            dcc.Loading(dcc.Graph(id="p-map-graph"), type="circle"),
                            style={"width": "48%", "display": "inline-block", "padding": "10px"}
                        ),
                        html.Div(
                            dcc.Loading(dcc.Graph(id="p-bar-graph"), type="circle"),
                            style={"width": "48%", "display": "inline-block", "padding": "10px", "float": "right"}
                        ),
                    ]),
                    html.Div(id="p-explanation", style={"marginTop": "15px", "padding": "15px", "backgroundColor": "#f8fafc", "borderRadius": "8px", "color": "#475569", "fontSize": "14px", "lineHeight": "1.6"})
                ]
            ),

            # Evapotranspiration Section
            html.Div(
                style=MODERN_STYLES["card"],
                children=[
                    html.H3("💧 Evapotranspiration (ET)", style=MODERN_STYLES["section_title"]),
                    html.Div([
                        html.Div(
                            dcc.Loading(dcc.Graph(id="et-map-graph"), type="circle"),
                            style={"width": "48%", "display": "inline-block", "padding": "10px"}
                        ),
                        html.Div(
                            dcc.Loading(dcc.Graph(id="et-bar-graph"), type="circle"),
                            style={"width": "48%", "display": "inline-block", "padding": "10px", "float": "right"}
                        ),
                    ]),
                    html.Div(id="et-explanation", style={"marginTop": "15px", "padding": "15px", "backgroundColor": "#f8fafc", "borderRadius": "8px", "color": "#475569", "fontSize": "14px", "lineHeight": "1.6"})
                ]
            ),

            # Water Balance Section
            html.Div(
                style=MODERN_STYLES["card"],
                children=[
                    html.H3("⚖️ Water Balance (P - ET)", style=MODERN_STYLES["section_title"]),
                    html.Div([
                        html.Div(
                            dcc.Loading(dcc.Graph(id="p-et-map-graph"), type="circle"),
                            style={"width": "48%", "display": "inline-block", "padding": "10px"}
                        ),
                        html.Div(
                            dcc.Loading(dcc.Graph(id="p-et-bar-graph"), type="circle"),
                            style={"width": "48%", "display": "inline-block", "padding": "10px", "float": "right"}
                        ),
                    ]),
                    html.Div(id="p-et-explanation", style={"marginTop": "15px", "padding": "15px", "backgroundColor": "#f8fafc", "borderRadius": "8px", "color": "#475569", "fontSize": "14px", "lineHeight": "1.6"})
                ]
            ),
        ]),

        # Footer
        html.Div(
            style={"textAlign": "center", "marginTop": "40px", "padding": "20px", "color": "#64748b", "borderTop": "1px solid #e2e8f0"},
            children=[
                html.P("Basin Data Dashboard • Interactive Hydrological Analysis"),
                html.P("Built with Dash & Plotly", style={"fontSize": "12px", "marginTop": "5px"})
            ]
        )
    ]
)


# ===========
# CALLBACKS
# ===========

@app.callback(Output("basin-map", "figure"), [Input("basin-dropdown", "value")])
def sync_map_with_dropdown(basin):
    return make_basin_selector_map(selected_basin=basin)

@app.callback(
    Output("basin-dropdown", "value"),
    [Input("basin-map", "clickData")],
    [State("basin-dropdown", "value")],
)
def sync_dropdown_with_map(clickData, current_value):
    if clickData and "points" in clickData and clickData["points"]:
        p0 = clickData["points"][0]
        basin = p0.get("location")
        if basin:
            return basin
    return current_value

@app.callback(
    [
        Output("global-start-year-dropdown", "options"),
        Output("global-start-year-dropdown", "value"),
        Output("global-end-year-dropdown", "options"),
        Output("global-end-year-dropdown", "value"),
        Output("file-info-feedback", "children"),
    ],
    [Input("basin-dropdown", "value")],
)
def init_global_year_controls(basin):
    if not basin or basin == "all":
        empty_options, empty_value = [], None
        return empty_options, empty_value, empty_options, empty_value, "🔍 All basins view — select a specific basin for detailed analysis."

    p_fp = find_nc_file(basin, "P")
    et_fp = find_nc_file(basin, "ET")
    lu_fp = find_nc_file(basin, "LU")

    p_min_yr = et_min_yr = lu_min_yr = 1990
    p_max_yr = et_max_yr = lu_max_yr = 2025

    try:
        if p_fp:
            with _open_xr_dataset(p_fp) as ds:
                if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
                    t = pd.to_datetime(ds["time"].values)
                    p_min_yr, p_max_yr = int(t.min().year), int(t.max().year)
        if et_fp:
            with _open_xr_dataset(et_fp) as ds:
                if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
                    t = pd.to_datetime(ds["time"].values)
                    et_min_yr, et_max_yr = int(t.min().year), int(t.max().year)
        if lu_fp:
            with _open_xr_dataset(lu_fp) as ds:
                if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
                    t = pd.to_datetime(ds["time"].values)
                    lu_min_yr, lu_max_yr = int(t.min().year), int(t.max().year)
    except Exception as e:
        print(f"[WARN] init_global_year_controls year scan: {e}")

    common_min = max(p_min_yr, et_min_yr)
    common_max = min(p_max_yr, et_max_yr)
    if common_min > common_max:
        common_min = min(p_min_yr, et_min_yr)
        common_max = max(p_max_yr, et_max_yr)

    def opts(a, b):
        years = list(range(a, b + 1)) if a <= b else []
        return [{"label": str(y), "value": y} for y in years]

    year_options = opts(common_min, common_max)

    if year_options:
        default_start = year_options[-3]["value"] if len(year_options) > 2 else year_options[0]["value"]
        default_end = year_options[-1]["value"]
    else:
        default_start = common_min
        default_end = common_max

    files_found = [
        html.Span("📁 Files found: ", style={"fontWeight": "600"}),
        html.Span(f"P: {os.path.basename(p_fp) if p_fp else '❌ Not Found'} • ", style={"fontFamily": "monospace", "fontSize": "13px"}),
        html.Span(f"ET: {os.path.basename(et_fp) if et_fp else '❌ Not Found'} • ", style={"fontFamily": "monospace", "fontSize": "13px"}),
        html.Span(f"LU: {os.path.basename(lu_fp) if lu_fp else '❌ Not Found'}", style={"fontFamily": "monospace", "fontSize": "13px"})
    ]

    return year_options, default_start, year_options, default_end, files_found

@app.callback(
    Output("basin-overview-content", "children"),
    [Input("basin-dropdown", "value"),
     Input("global-start-year-dropdown", "value"),
     Input("global-end-year-dropdown", "value")]
)
def update_basin_overview(basin, start_year, end_year):
    if basin == "all" or not basin or not start_year or not end_year:
        return html.Div("Select a specific basin and year range to view overview metrics.", 
                       style={"textAlign": "center", "color": "#64748b", "padding": "40px"})
    
    try:
        # Use the selected year range for overview
        start_year = int(start_year)
        end_year = int(end_year)
        
        metrics = get_basin_overview_metrics_for_range(basin, start_year, end_year)
        
        if not metrics:
            return html.Div(
                f"No overview data available for {basin} in {start_year}-{end_year}.",
                style={"textAlign": "center", "color": "#64748b", "padding": "40px"}
            )
        
        # Read intro text
        intro_text = read_basin_intro(basin)

        # Format values for summary
        total_inflows = f"{metrics.get('total_inflows', 0):.0f}"
        precip_pct = f"{metrics.get('precipitation_percentage', 0):.0f}"
        imports = f"{metrics.get('surface_water_imports', 0):.0f}"
        total_consumption = f"{metrics.get('total_water_consumption', 0):.0f}"
        manmade_consumption = f"{metrics.get('manmade_consumption', 0):.0f}"
        treated_wastewater = f"{metrics.get('treated_wastewater', 0):.0f}"
        non_irrigated = f"{metrics.get('non_irrigated_consumption', 0):.0f}"
        recharge = f"{metrics.get('recharge', 0):.0f}"

        year_range_str = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"

        # Create dynamic summary using the requested template
        summary_items = [
            f"The total water inflows into the {basin} basin in {year_range_str} is {total_inflows} Mm3/ year.",
            f"Precipitation accounts for {precip_pct}% of the gross inflows and the remaining from imports for domestic purposes.",
            f"{imports} Mm3/ year of water is imported into the basin for domestic use.",
            f"The total landscape water consumption is {total_consumption} Mm3/ year.",
            f"The manmade water consumption is {manmade_consumption} Mm3/ year",
            f"About {treated_wastewater} Mm3/ year of treated wastewater that is discharged to streams.",
            f"The average sectorial non-irrigated water consumption is {non_irrigated} Mm3/ year.",
            f"On average {recharge} Mm3/ year recharged the basin."
        ]

        # Create modern metric cards
        metric_cards = []
        
        # Key metrics to display
        key_metrics = [
            {
                'title': 'Total Water Inflows',
                'value': metrics.get('total_inflows', 0),
                'unit': 'Mm³/year',
                'icon': '🌊',
                'color': '#3b82f6'
            },
            {
                'title': 'Precipitation',
                'value': metrics.get('total_precipitation', 0),
                'unit': 'Mm³/year',
                'percentage': metrics.get('precipitation_percentage'),
                'icon': '🌧️',
                'color': '#06b6d4'
            },
            {
                'title': 'Water Imports',
                'value': metrics.get('surface_water_imports', 0),
                'unit': 'Mm³/year',
                'icon': '🚚',
                'color': '#8b5cf6'
            },
            {
                'title': 'Total Water Consumption',
                'value': metrics.get('total_water_consumption', 0),
                'unit': 'Mm³/year',
                'icon': '💧',
                'color': '#ef4444'
            },
            {
                'title': 'Manmade Consumption',
                'value': metrics.get('manmade_consumption', 0),
                'unit': 'Mm³/year',
                'icon': '🏭',
                'color': '#f59e0b'
            },
            {
                'title': 'Treated Wastewater',
                'value': metrics.get('treated_wastewater', 0),
                'unit': 'Mm³/year',
                'icon': '♻️',
                'color': '#10b981'
            },
            {
                'title': 'Non-irrigated Consumption',
                'value': metrics.get('non_irrigated_consumption', 0),
                'unit': 'Mm³/year',
                'icon': '🏘️',
                'color': '#6366f1'
            },
            {
                'title': 'Groundwater Recharge',
                'value': metrics.get('recharge', 0),
                'unit': 'Mm³/year',
                'icon': '⤵️',
                'color': '#06b6d4'
            }
        ]
        
        for metric in key_metrics:
            value = metric['value']
            if value == 0 or pd.isna(value):
                continue
                
            # Format the value
            if abs(value) < 0.001:
                formatted_value = "0"
            elif abs(value) < 1:
                formatted_value = f"{value:.3f}"
            elif abs(value) < 10:
                formatted_value = f"{value:.2f}"
            elif abs(value) < 100:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.0f}"
            
            card_content = [
                html.Div([
                    # Icon and title
                    html.Div([
                        html.Span(metric['icon'], style={"fontSize": "24px", "marginRight": "10px"}),
                        html.Div([
                            html.H4(metric['title'], style={"margin": "0", "fontSize": "16px", "fontWeight": "600", "color": "#374151"}),
                            html.P(f"{formatted_value} {metric['unit']}", style={
                                "margin": "0", 
                                "fontSize": "24px", 
                                "fontWeight": "700", 
                                "color": metric['color'],
                                "marginTop": "5px"
                            })
                        ])
                    ], style={"display": "flex", "alignItems": "center"}),
                    
                    # Percentage if available
                    html.Div([
                        html.P(f"{metric.get('percentage', 0):.1f}% of inflows", 
                              style={"margin": "0", "fontSize": "12px", "color": "#6b7280"})
                    ]) if metric.get('percentage') else None
                ], style={"padding": "0"})
            ]
            
            # Remove None values
            card_content = [c for c in card_content if c is not None]
            
            metric_card = html.Div(
                card_content,
                style={
                    "backgroundColor": "white",
                    "border": f"2px solid {metric['color']}20",
                    "borderRadius": "12px",
                    "padding": "20px",
                    "margin": "8px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
                    "minWidth": "200px",
                    "flex": "1",
                    "minHeight": "120px",
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "space-between"
                }
            )
            metric_cards.append(metric_card)
        
        if not metric_cards:
            return html.Div(
                "No valid metrics found in the data.",
                style={"textAlign": "center", "color": "#64748b", "padding": "40px"}
            )
        
        # Create a responsive grid layout
        return html.Div([
            # Intro Section
            html.Div([
                html.H4(f"Introduction to {basin} Basin",
                       style={"color": "#1e293b", "marginBottom": "10px", "fontSize": "20px"}),
                html.P(intro_text, style={"color": "#475569", "fontSize": "16px", "lineHeight": "1.6"})
            ], style={"marginBottom": "30px", "padding": "20px", "backgroundColor": "white", "borderRadius": "8px", "border": "1px solid #e2e8f0"}) if intro_text else None,

            html.Div([
                html.H4(f"Water Balance Overview - {start_year} to {end_year} Average", 
                       style={"color": "#1e293b", "marginBottom": "20px", "fontSize": "20px"})
            ], style={"width": "100%", "marginBottom": "15px"}),
            
            # Dynamic Summary Section
            html.Div([
                html.H5("💡 Executive Summary", style={"color": "#1e293b", "marginBottom": "15px", "fontSize": "18px"}),
                html.Ul([
                    html.Li(item, style={"marginBottom": "8px"}) for item in summary_items
                ], style={"color": "#475569", "fontSize": "16px", "lineHeight": "1.6"})
            ], style={
                "backgroundColor": "#f8fafc",
                "borderRadius": "8px",
                "padding": "20px",
                "marginBottom": "25px",
                "borderLeft": "4px solid #3b82f6"
            }),

            html.Div(
                metric_cards,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                    "gap": "15px",
                    "width": "100%"
                }
            )
        ])
        
    except Exception as e:
        print(f"Error generating overview: {e}")
        return html.Div(
            f"Error generating overview: {str(e)}",
            style={"textAlign": "center", "color": "#ef4444", "padding": "40px"}
        )

def _generate_explanation(vtype: str, basin: str, start_year: int, end_year: int, y_vals: np.ndarray, months: list):
    """Generate a dynamic explanation for the hydro graphs."""
    if len(y_vals) == 0 or not np.any(np.isfinite(y_vals)):
        return "No sufficient data to generate an explanation."

    mean_val = np.nanmean(y_vals)
    max_val = np.nanmax(y_vals)
    min_val = np.nanmin(y_vals)
    max_month = months[np.nanargmax(y_vals)]
    min_month = months[np.nanargmin(y_vals)]

    if vtype == "P":
        return (f"**Precipitation Analysis ({start_year}–{end_year}):** "
                f"The average monthly precipitation across the {basin} basin is **{mean_val:.2f} mm**. "
                f"The wettest month is typically **{max_month}** with an average of **{max_val:.2f} mm**, "
                f"while the driest month is **{min_month}** with **{min_val:.2f} mm**. "
                f"This seasonal pattern indicates the primary rainy season and dry periods, essential for water resource planning.")
    elif vtype == "ET":
        return (f"**Evapotranspiration Analysis ({start_year}–{end_year}):** "
                f"The average monthly evapotranspiration is **{mean_val:.2f} mm**. "
                f"Peak water consumption occurs in **{max_month}** (**{max_val:.2f} mm**), likely driven by higher temperatures and vegetation growth. "
                f"The lowest rates are observed in **{min_month}** (**{min_val:.2f} mm**).")
    elif vtype == "P-ET":
        status = "positive water yield" if mean_val > 0 else "water deficit"
        return (f"**Water Balance Analysis ({start_year}–{end_year}):** "
                f"The basin shows an average monthly {status} of **{mean_val:.2f} mm**. "
                f"The maximum surplus occurs in **{max_month}** (**{max_val:.2f} mm**), representing potential recharge or runoff periods. "
                f"The maximum deficit occurs in **{min_month}** (**{min_val:.2f} mm**), indicating periods where consumption exceeds precipitation.")
    return ""

def _hydro_figs(basin: str, start_year: int | None, end_year: int | None, vtype: str):
    if basin == "all" or not basin:
        return _empty_fig("Select a specific basin to view data."), _empty_fig("Select a specific basin to view data."), ""
    if start_year is None or end_year is None:
        return _empty_fig("Select year range."), _empty_fig("Select year range."), ""

    ys, ye = int(start_year), int(end_year)
    if ys > ye:
        ys, ye = ye, ys

    da_ts, _, msg = load_and_process_data(basin, vtype, year_start=ys, year_end=ye, aggregate_time=False)
    if da_ts is None or da_ts.sizes.get("time", 0) == 0:
        return _empty_fig(f"{vtype} data unavailable: {msg or ''}"), _empty_fig(), f"Data unavailable for {vtype}."

    da_map = da_ts.mean(dim="time", skipna=True)
    
    # Use the new clean heatmap function for P, ET, and P-ET plots
    if vtype in ["P", "ET"]:
        colorscale = "Blues" if vtype == "P" else "YlOrRd"
        fig_map = _create_clean_heatmap(
            da_map,
            title=f"Mean {vtype} ({ys}–{ye})",
            colorscale=colorscale,
            z_label="mm"
        )
    else:  # P-ET
        fig_map = _create_clean_heatmap(
            da_map,
            title=f"Mean Water Yield (P-ET) ({ys}–{ye})",
            colorscale="RdBu",
            z_label="mm"
        )
    
    fig_map = add_shapefile_to_fig(fig_map, basin)

    spatial_dims = [d for d in ["latitude", "longitude"] if d in da_ts.dims]
    spatial_mean_ts = da_ts.mean(dim=spatial_dims, skipna=True)
    explanation = ""

    try:
        monthly = spatial_mean_ts.groupby("time.month").mean(skipna=True).rename({"month": "Month"})
        months = [pd.to_datetime(m, format="%m").strftime("%b") for m in monthly["Month"].values]
        y_vals = np.asarray(monthly.values).flatten()
        if np.isfinite(y_vals).any():
            fig_bar = px.bar(x=months, y=y_vals, title=f"Mean Monthly {vtype} ({ys}–{ye})",
                             labels={"x": "Month", "y": f"Mean Daily {vtype} (mm)"})
            # Modern bar chart styling
            fig_bar.update_traces(marker_color='#3b82f6', marker_line_color='#1d4ed8', 
                                marker_line_width=1, opacity=0.8)
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#1e293b")
            )
            explanation = _generate_explanation(vtype, basin, ys, ye, y_vals, months)
        else:
            fig_bar = _empty_fig(f"No valid monthly data for {vtype} in {ys}–{ye}.")
            explanation = "No valid data to generate explanation."
    except Exception:
        fig_bar = _empty_fig(f"No monthly grouping available for {vtype}.")
        explanation = "Error generating explanation."

    return fig_map, fig_bar, dcc.Markdown(explanation)

@app.callback(
    [Output("p-map-graph", "figure"), Output("p-bar-graph", "figure"), Output("p-explanation", "children")],
    [Input("basin-dropdown", "value"),
     Input("global-start-year-dropdown", "value"),
     Input("global-end-year-dropdown", "value")],
)
def update_p_outputs(basin, start_year, end_year):
    return _hydro_figs(basin, start_year, end_year, "P")

@app.callback(
    [Output("et-map-graph", "figure"), Output("et-bar-graph", "figure"), Output("et-explanation", "children")],
    [Input("basin-dropdown", "value"),
     Input("global-start-year-dropdown", "value"),
     Input("global-end-year-dropdown", "value")],
)
def update_et_outputs(basin, start_year, end_year):
    return _hydro_figs(basin, start_year, end_year, "ET")

@app.callback(
    Output("lu-map-graph", "figure"),
    [Input("basin-dropdown", "value")],
)
def update_lu_map(basin):
    """Show static land use map for the latest available year"""
    if basin == "all" or not basin:
        return _empty_fig("Select a specific basin to view land use data.")

    lu_fp = find_nc_file(basin, "LU")
    if not lu_fp:
        return _empty_fig("Land Use data not found for this basin.")
    
    try:
        with _open_xr_dataset(lu_fp) as ds:
            if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
                t = pd.to_datetime(ds["time"].values)
                latest_year = int(t.max().year)
            else:
                latest_year = 2020
    except Exception:
        latest_year = 2020

    da, _, msg = load_and_process_data(basin, "LU", year_start=latest_year, year_end=latest_year)
    if da is None:
        return _empty_fig(f"Land Use data not available: {msg or ''}")

    vals = np.asarray(da.values)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
        return _empty_fig("No valid land use classes found")

    class_list = sorted(np.unique(finite_vals).astype(int).tolist())
    x = np.asarray(da["longitude"].values)
    y = np.asarray(da["latitude"].values)

    idx_map = {c: i for i, c in enumerate(class_list)}
    z_idx = np.full(vals.shape, np.nan, dtype=float)
    for c, i in idx_map.items():
        z_idx[np.isclose(vals, c)] = i

    colors = [class_info.get(c, {"color": "gray"})["color"] for c in class_list]
    names  = [class_info.get(c, {"name": f"Class {c}"})["name"] for c in class_list]

    fig_map = go.Figure()
    fig_map.add_trace(
        go.Heatmap(
            z=z_idx, x=x, y=y,
            zmin=-0.5, zmax=len(class_list) - 0.5,
            colorscale=[[i/(len(colors)-1) if len(colors) > 1 else 0, col] for i, col in enumerate(colors)],
            showscale=False, hoverinfo="skip"
        )
    )
    for name, col in zip(names, colors):
        fig_map.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=10, color=col),
                       name=name, showlegend=True, hoverinfo="skip")
        )

    actual_year = latest_year
    try:
        if "time" in da.coords and da.sizes.get("time", 0) > 0:
            time_val = pd.to_datetime(da["time"].values).item()
            actual_year = time_val.year
    except Exception:
        pass

    fig_map.update_layout(
        title=f"Land Use / Land Cover - {actual_year}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(itemsizing="constant", bgcolor='rgba(255,255,255,0.9)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig_map = add_shapefile_to_fig(fig_map, basin)
    return fig_map

@app.callback(
    [Output("p-et-map-graph", "figure"), Output("p-et-bar-graph", "figure"), Output("p-et-explanation", "children")],
    [Input("basin-dropdown", "value"),
     Input("global-start-year-dropdown", "value"),
     Input("global-end-year-dropdown", "value")],
)
def update_p_et_outputs(basin, start_year, end_year):
    if basin == "all" or not basin:
        return _empty_fig("Select a specific basin to view data."), _empty_fig("Select a specific basin to view data."), ""
    if start_year is None or end_year is None:
        return _empty_fig("Select year range."), _empty_fig("Select year range."), ""

    ys, ye = int(start_year), int(end_year)
    if ys > ye:
        ys, ye = ye, ys

    da_p_ts, _, _ = load_and_process_data(basin, "P",  year_start=ys, year_end=ye, aggregate_time=False)
    da_et_ts, _, _ = load_and_process_data(basin, "ET", year_start=ys, year_end=ye, aggregate_time=False)
    if da_p_ts is None or da_et_ts is None:
        return _empty_fig("P or ET data missing."), _empty_fig(), "Data missing."

    da_p_aligned, da_et_aligned = xr.align(da_p_ts, da_et_ts, join="inner")
    if da_p_aligned.sizes.get("time", 0) == 0:
        return _empty_fig("No overlapping time steps for P and ET."), _empty_fig(), "No data overlap."

    da_p_et_ts = da_p_aligned - da_et_aligned

    da_map = da_p_et_ts.mean(dim="time", skipna=True)
    
    # Use clean heatmap for P-ET
    fig_map = _create_clean_heatmap(
        da_map, 
        title=f"Mean Water Yield (P-ET) ({ys}–{ye})", 
        colorscale="RdBu", 
        z_label="mm"
    )
    fig_map = add_shapefile_to_fig(fig_map, basin)

    spatial_dims = [d for d in ["latitude", "longitude"] if d in da_p_et_ts.dims]
    spatial_mean = da_p_et_ts.mean(dim=spatial_dims, skipna=True)
    explanation = ""

    try:
        monthly = spatial_mean.groupby("time.month").mean(skipna=True).rename({"month": "Month"})
        months = [pd.to_datetime(m, format="%m").strftime("%b") for m in monthly["Month"].values]
        y_vals = np.asarray(monthly.values).flatten()
        if np.isfinite(y_vals).any():
            fig_bar = px.bar(x=months, y=y_vals, title=f"Mean Monthly Water Yield (P-ET) ({ys}–{ye})",
                             labels={"x": "Month", "y": "Mean Daily P-ET (mm)"})
            fig_bar.update_traces(marker_color=["#ef4444" if v < 0 else "#10b981" for v in y_vals], 
                                marker_line_color='#1e293b', marker_line_width=1, opacity=0.8)
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#1e293b")
            )
            explanation = _generate_explanation("P-ET", basin, ys, ye, y_vals, months)
        else:
            fig_bar = _empty_fig(f"No valid monthly data for P-ET in {ys}–{ye}.")
            explanation = "No valid data to generate explanation."
    except Exception:
        fig_bar = _empty_fig("No monthly grouping available for P-ET.")
        explanation = "Error generating explanation."

    return fig_map, fig_bar, dcc.Markdown(explanation)


# =====
# MAIN
# =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)