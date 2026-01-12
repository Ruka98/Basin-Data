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
    # Try text/introduction.txt
    intro_path = os.path.join(BASIN_DIR, basin_name, "text", "introduction.txt")
    if os.path.exists(intro_path):
         try:
            with open(intro_path, 'r') as f:
                return f.read()
         except Exception as e:
            print(f"Error reading intro file: {e}")
    return "Introduction text not available."

def read_text_section(basin_name: str, section: str) -> str:
    """Read specific text section (introduction, methodology, assumptions, limitations, objectives)."""
    text_path = os.path.join(BASIN_DIR, basin_name, "text", f"{section}.txt")
    if os.path.exists(text_path):
        try:
            with open(text_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {section} file: {e}")
            if section == "introduction":
                return read_basin_intro(basin_name)
    elif section == "introduction":
         return read_basin_intro(basin_name)
    elif section == "objectives":
        # Fallback or generic objectives if file missing
        return ("The primary objective of this dashboard is to provide a comprehensive, rapid assessment of "
                f"water resources in the {basin_name} basin. By integrating remote sensing data, "
                "hydrological modeling, and water accounting principles (WA+), this tool aims to support "
                "decision-makers in sustainable water management, identifying water deficits, and optimizing allocation.")

    return f"No {section} text available."

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

def parse_wa_sheet(csv_file: str):
    """Robust parsing of sheet1 CSV for WA+."""
    try:
        df = pd.read_csv(csv_file, sep=';')
        
        cleaned_rows = []
        for _, row in df.iterrows():
            try:
                val = float(row.get('VALUE', 0)) * 1000
            except (ValueError, TypeError):
                val = 0
            
            cleaned_rows.append({
                'CLASS': row.get('CLASS', '').strip(),
                'SUBCLASS': row.get('SUBCLASS', '').strip(),
                'VARIABLE': row.get('VARIABLE', '').strip(),
                'VALUE': val
            })
        
        return pd.DataFrame(cleaned_rows)
    except Exception as e:
        print(f"Error parsing WA sheet: {e}")
        return pd.DataFrame()

def get_wa_data(basin_name: str, start_year: int, end_year: int):
    """Aggregates WA+ data for a range of years."""
    all_data = []

    for year in range(start_year, end_year + 1):
        csv_file = find_yearly_csv(basin_name, year)
        if csv_file:
            df = parse_wa_sheet(csv_file)
            if not df.empty:
                df['Year'] = year
                all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # Group by CLASS, SUBCLASS, VARIABLE and mean the VALUE
    agg_df = combined_df.groupby(['CLASS', 'SUBCLASS', 'VARIABLE'])['VALUE'].mean().reset_index()
    return agg_df

def get_basin_overview_metrics_for_range(basin_name: str, start_year: int, end_year: int):
    """Get comprehensive basin overview metrics averaged over a year range."""
    agg_df = get_wa_data(basin_name, start_year, end_year)
    if agg_df.empty:
        return None

    metrics = {}
    metrics['total_inflows'] = agg_df[agg_df['CLASS'] == 'INFLOW']['VALUE'].sum()

    metrics['total_precipitation'] = agg_df[
        (agg_df['CLASS'] == 'INFLOW') & (agg_df['SUBCLASS'] == 'PRECIPITATION')
    ]['VALUE'].sum()

    metrics['precipitation_rainfall'] = agg_df[
        (agg_df['CLASS'] == 'INFLOW') & (agg_df['SUBCLASS'] == 'PRECIPITATION') & (agg_df['VARIABLE'] == 'Rainfall')
    ]['VALUE'].sum()

    metrics['surface_water_imports'] = agg_df[
        (agg_df['CLASS'] == 'INFLOW') & (agg_df['SUBCLASS'] == 'SURFACE WATER') &
        (agg_df['VARIABLE'].isin(['Main riverstem', 'Tributaries']))
    ]['VALUE'].sum()

    et_rows = agg_df[(agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'].str.contains('ET'))]
    metrics['total_water_consumption'] = et_rows[~et_rows['VARIABLE'].isin(['Manmade', 'Consumed Water'])]['VALUE'].sum()

    metrics['manmade_consumption'] = agg_df[
        (agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'] == 'ET INCREMENTAL') & (agg_df['VARIABLE'] == 'Manmade')
    ]['VALUE'].sum()

    metrics['non_irrigated_consumption'] = agg_df[
        (agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'] == 'ET INCREMENTAL') & (agg_df['VARIABLE'] == 'Consumed Water')
    ]['VALUE'].sum()

    metrics['treated_wastewater'] = agg_df[
         (agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'] == 'OTHER') & (agg_df['VARIABLE'] == 'Treated Waste Water')
    ]['VALUE'].sum()

    recharge_val = agg_df[
        (agg_df['CLASS'] == 'STORAGE') & (agg_df['SUBCLASS'] == 'CHANGE') & (agg_df['VARIABLE'].str.contains('Surface storage'))
    ]['VALUE'].sum()
    metrics['recharge'] = abs(recharge_val) if recharge_val < 0 else recharge_val

    if metrics['total_inflows'] > 0:
        metrics['precipitation_percentage'] = (metrics['total_precipitation'] / metrics['total_inflows'] * 100)
    
    return metrics


# ======================
# INDICATOR UTILITIES
# ======================

def parse_indicators(csv_file: str):
    """Parse indicators CSV."""
    try:
        df = pd.read_csv(csv_file, sep=';')
        return df
    except Exception as e:
        print(f"Error parsing indicators: {e}")
        return pd.DataFrame()

def get_indicators(basin_name: str, start_year: int, end_year: int):
    """Aggregates indicators for a range of years."""
    all_data = []

    # We need to find indicator files
    results_dir = os.path.join(BASIN_DIR, basin_name, "Results", "indicators")
    if not os.path.isdir(results_dir):
        return pd.DataFrame()

    for year in range(start_year, end_year + 1):
        pat = os.path.join(results_dir, f"indicators_{year}.csv")
        csv_file = _first_existing([pat])
        if csv_file:
            df = parse_indicators(csv_file)
            if not df.empty:
                df['Year'] = year
                all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    numeric_cols = ['VALUE']
    meta_cols = ['UNIT', 'DEFINITION', 'TRAFFIC_LIGHT']
    
    agg_df = combined_df.groupby('INDICATOR')[numeric_cols].mean().reset_index()

    meta_df = combined_df[['INDICATOR'] + meta_cols].drop_duplicates('INDICATOR')
    agg_df = pd.merge(agg_df, meta_df, on='INDICATOR', how='left')

    return agg_df

# ======================
# VALIDATION UTILITIES
# ======================

def get_validation_data(basin_name: str, var_type: str):
    """Get validation data for rainfall or ET."""
    filename = "rainfall_validation.csv" if var_type == "P" else "et_validation.csv"
    filepath = os.path.join(BASIN_DIR, basin_name, "Results", "validation", filename)

    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, sep=';')
        return df
    except Exception as e:
        print(f"Error reading validation data: {e}")
        return pd.DataFrame()


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
    """Load ALL basins' shapefiles."""
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
    if da is None:
        return None, None, None
    valid_mask = np.isfinite(da.values)
    if not np.any(valid_mask):
        return None, None, None
    x = np.asarray(da["longitude"].values)
    y = np.asarray(da["latitude"].values)
    z_clean = da.values.copy()
    return z_clean, x, y

def _safe_imshow(da: xr.DataArray, title: str, colorscale="Viridis", z_label="value"):
    if da is None or "latitude" not in da.coords or "longitude" not in da.coords:
        return _empty_fig("No data to display")
    z, x, y = _clean_nan_data(da)
    if z is None:
        return _empty_fig("No valid data values")
    z_masked = np.ma.masked_invalid(z)
    fig = px.imshow(
        z_masked, x=x, y=y, origin="lower", aspect="equal",
        color_continuous_scale=colorscale, title=title, labels={"color": z_label}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#1e293b"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

def _create_clean_heatmap(da: xr.DataArray, title: str, colorscale="Viridis", z_label="value"):
    if da is None or "latitude" not in da.coords or "longitude" not in da.coords:
        return _empty_fig("No data to display")
    z, x, y = _clean_nan_data(da)
    if z is None:
        return _empty_fig("No valid data values")
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=x, y=y, colorscale=colorscale, zmid=0,
        colorbar=dict(title=z_label, thickness=15, len=0.75, yanchor="middle", y=0.5),
        hoverinfo="x+y+z",
        hovertemplate='Longitude: %{x:.2f}<br>Latitude: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Longitude", yaxis_title="Latitude",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(color="#1e293b"), margin=dict(l=50, r=50, t=60, b=50)
    )
    return fig

def add_shapefile_to_fig(fig: go.Figure, basin_name: str) -> go.Figure:
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
                                         line=dict(color="black", width=1), name="Basin Boundary",
                                         showlegend=False, hoverinfo='skip'))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines",
                                             line=dict(color="black", width=1), name="Basin Boundary",
                                             showlegend=False, hoverinfo='skip'))
    except Exception as e:
        print(f"[WARN] Could not overlay shapefile: {e}")
    return fig

def _empty_fig(msg="No data to display"):
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": msg, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}],
        margin=dict(l=0, r=0, t=35, b=0),
        plot_bgcolor='white', paper_bgcolor='white'
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

    # IWMI Colors: Green #60a730, Blue #0076a5
    ch = go.Choroplethmapbox(
        geojson=gj,
        locations=locations,
        featureidkey="properties.basin",
        z=z_vals,
        colorscale=[[0, "rgba(96, 167, 48, 0.4)"], [1, "rgba(96, 167, 48, 0.4)"]],
        marker=dict(line=dict(width=3 if selected_basin and selected_basin != "all" else 1.8,
                              color="#0076a5")),
        hovertemplate="%{location}<extra></extra>",
        showscale=False,
    )
    fig = go.Figure(ch)

    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = (maxx - minx) * 0.08 if maxx > minx else 0.1
    pad_y = (maxy - miny) * 0.08 if maxy > miny else 0.1
    west, east = float(minx - pad_x), float(maxx + pad_x)
    south, north = float(miny - pad_y), float(maxy + pad_y)

    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0

    import math
    map_w, map_h = 900.0, 600.0
    span_lon = max(east - west, 0.001)
    span_lat = max(north - south, 0.001)
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


# Land use class information (Same as before)
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

# IWMI-Themed Styles
# Colors: Green #60a730, Blue #0076a5, White #ffffff
MODERN_STYLES = {
    "container": {
        "maxWidth": "1600px",
        "margin": "0 auto",
        "padding": "20px",
        "fontFamily": "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh"
    },
    "header_bar": {
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "padding": "20px 40px",
        "backgroundColor": "white",
        "borderRadius": "0 0 16px 16px",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
        "marginBottom": "30px",
        "borderTop": "6px solid #60a730"
    },
    "title_container": {
        "textAlign": "left"
    },
    "main_title": {
        "color": "#0076a5",
        "fontSize": "2.5rem",
        "fontWeight": "800",
        "marginBottom": "5px",
        "letterSpacing": "-0.5px"
    },
    "sub_title": {
        "color": "#60a730",
        "fontSize": "1.8rem",
        "fontWeight": "600",
        "marginTop": "0"
    },
    "logo_container": {
        "display": "flex",
        "gap": "20px",
        "alignItems": "center"
    },
    "logo_img": {
        "height": "60px",
        "objectFit": "contain"
    },
    "card": {
        "backgroundColor": "white",
        "borderRadius": "16px",
        "padding": "30px",
        "marginBottom": "30px",
        "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.05)",
        "border": "1px solid #edf2f7"
    },
    "section_title": {
        "color": "#0076a5",
        "fontSize": "1.6rem",
        "fontWeight": "700",
        "marginBottom": "20px",
        "borderLeft": "5px solid #60a730",
        "paddingLeft": "15px"
    },
    "text_content": {
        "fontSize": "1.1rem",
        "lineHeight": "1.7",
        "color": "#4a5568",
        "marginBottom": "20px"
    },
    "dropdown": {
        "fontSize": "1.1rem"
    },
    "tab_selected": {
        "borderTop": "4px solid #60a730",
        "color": "#0076a5",
        "fontWeight": "bold",
        "backgroundColor": "white"
    },
    "tab": {
        "padding": "15px",
        "fontWeight": "600",
        "color": "#718096"
    }
}

basin_folders = [d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))] if os.path.isdir(BASIN_DIR) else []
basin_options = [{"label": "View All Basins", "value": "all"}] + [{"label": b, "value": b} for b in sorted(basin_folders)]

app.layout = html.Div(
    style=MODERN_STYLES["container"],
    children=[
        # Header Section
        html.Div(
            style=MODERN_STYLES["header_bar"],
            children=[
                html.Div(
                    style=MODERN_STYLES["title_container"],
                    children=[
                        html.H1("Rapid Water Accounting Dashboard", style=MODERN_STYLES["main_title"]),
                        html.H2("Jordan", style=MODERN_STYLES["sub_title"])
                    ]
                ),
                html.Div(
                    style=MODERN_STYLES["logo_container"],
                    children=[
                        html.Img(src=app.get_asset_url("iwmi.png"), style=MODERN_STYLES["logo_img"]),
                        html.Img(src=app.get_asset_url("cgiar.png"), style=MODERN_STYLES["logo_img"])
                    ]
                )
            ]
        ),

        # 1. Basin Selection (First Step)
        html.Div(
            style=MODERN_STYLES["card"],
            children=[
                html.H3("📍 Select Basin", style=MODERN_STYLES["section_title"]),
                html.Div([
                     html.Div([
                        html.Label("Basin", style={"fontWeight": "600", "marginBottom": "8px", "display": "block", "color": "#0076a5"}),
                        dcc.Dropdown(
                            id="basin-dropdown",
                            options=basin_options,
                            value="all" if basin_folders else None,
                            clearable=False,
                            style=MODERN_STYLES["dropdown"]
                        ),
                        html.Br(),
                        html.Label("Analysis Period", style={"fontWeight": "600", "marginBottom": "8px", "display": "block", "color": "#0076a5"}),
                        html.Div([
                            html.Div([
                                dcc.Dropdown(id="global-start-year-dropdown", clearable=False, placeholder="Start Year")
                            ], style={"width": "48%", "display": "inline-block"}),
                            html.Div([
                                dcc.Dropdown(id="global-end-year-dropdown", clearable=False, placeholder="End Year")
                            ], style={"width": "48%", "display": "inline-block", "float": "right"}),
                        ])
                    ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "20px"}),

                    html.Div([
                        dcc.Graph(id="basin-map", style={"height": "400px", "borderRadius": "12px", "overflow": "hidden"})
                    ], style={"width": "68%", "display": "inline-block", "verticalAlign": "top"})
                ]),
                html.Div(id="file-info-feedback", style={"marginTop": "15px", "color": "#718096", "fontSize": "0.9rem"})
            ]
        ),

        # Content Containers - Only visible after selection
        html.Div(id="main-content-area")
    ]
)

# LAYOUT GENERATORS

def get_introduction_layout(basin):
    text = read_text_section(basin, "introduction")
    return html.Div(
        style=MODERN_STYLES["card"],
        children=[
            html.H3("📖 Introduction", style=MODERN_STYLES["section_title"]),
            dcc.Markdown(text, style=MODERN_STYLES["text_content"])
        ]
    )

def get_objectives_layout(basin):
    text = read_text_section(basin, "objectives")
    return html.Div(
        style=MODERN_STYLES["card"],
        children=[
            html.H3("🎯 Objectives", style=MODERN_STYLES["section_title"]),
            dcc.Markdown(text, style=MODERN_STYLES["text_content"])
        ]
    )

def get_methodology_layout(basin):
    text = read_text_section(basin, "methodology")
    return html.Div(
        style=MODERN_STYLES["card"],
        children=[
            html.H3("🔬 Methodology", style=MODERN_STYLES["section_title"]),
            dcc.Markdown(text, style=MODERN_STYLES["text_content"])
        ]
    )

def get_results_layout(basin):
    return html.Div(
        style=MODERN_STYLES["card"],
        children=[
            html.H3("📊 Results", style=MODERN_STYLES["section_title"]),
            dcc.Tabs(
                id="dashboard-tabs",
                value="overview",
                selected_style=MODERN_STYLES["tab_selected"],
                style={"marginBottom": "20px"},
                children=[
                    dcc.Tab(label="Overview", value="overview", style=MODERN_STYLES["tab"], selected_style=MODERN_STYLES["tab_selected"]),
                    dcc.Tab(label="Spatial Analysis", value="spatial", style=MODERN_STYLES["tab"], selected_style=MODERN_STYLES["tab_selected"]),
                    dcc.Tab(label="Water Accounting", value="wa", style=MODERN_STYLES["tab"], selected_style=MODERN_STYLES["tab_selected"]),
                    dcc.Tab(label="Climate & Validation", value="climate", style=MODERN_STYLES["tab"], selected_style=MODERN_STYLES["tab_selected"]),
                ]
            ),
            html.Div(id="tab-content")
        ]
    )

# --- RESULTS TABS GENERATORS ---

def get_overview_tab_content():
    return html.Div(id="basin-overview-content")

def get_spatial_tab_content():
    return html.Div([
        html.Div([
             html.H4("Land Use / Land Cover", style={"color": "#0076a5"}),
             html.Div([
                 html.Div(dcc.Loading(dcc.Graph(id="lu-map-graph"), type="dot"), style={"width": "49%", "display": "inline-block"}),
                 html.Div(dcc.Loading(dcc.Graph(id="lu-bar-graph"), type="dot"), style={"width": "49%", "display": "inline-block", "float": "right"})
             ]),
             html.Div(id="lu-explanation", style={"padding": "15px", "backgroundColor": "#f0fdf4", "borderRadius": "8px", "marginTop": "10px"})
        ], style={"marginBottom": "40px"}),

        html.Div([
            html.H4("Land Use - Water Coupling", style={"color": "#0076a5"}),
            dcc.Loading(dcc.Graph(id="lu-et-p-bar"), type="dot")
        ], style={"marginBottom": "40px"}),

        html.Div([
            html.H4("Water Balance (P - ET)", style={"color": "#0076a5"}),
            html.Div([
                 html.Div(dcc.Loading(dcc.Graph(id="p-et-map-graph"), type="dot"), style={"width": "49%", "display": "inline-block"}),
                 html.Div(dcc.Loading(dcc.Graph(id="p-et-bar-graph"), type="dot"), style={"width": "49%", "display": "inline-block", "float": "right"})
             ]),
             html.Div(id="p-et-explanation", style={"padding": "15px", "backgroundColor": "#f0fdf4", "borderRadius": "8px", "marginTop": "10px"})
        ])
    ])

def get_wa_tab_content():
    return html.Div([
        html.Div([
            html.H4("Resource Base (Sankey Diagram)", style={"color": "#0076a5"}),
            dcc.Loading(dcc.Graph(id="wa-resource-base-sankey"), type="dot")
        ], style={"marginBottom": "40px"}),

        html.Div([
            html.H4("Sectoral Water Consumption", style={"color": "#0076a5"}),
            dcc.Loading(dcc.Graph(id="wa-sectoral-bar"), type="dot")
        ], style={"marginBottom": "40px"}),

        html.Div([
            html.H4("Performance Indicators", style={"color": "#0076a5"}),
            html.Div(id="wa-indicators-container")
        ])
    ])

def get_climate_tab_content():
    return html.Div([
        html.Div([
            html.H4("Precipitation Analysis", style={"color": "#0076a5"}),
             html.Div([
                 html.Div(dcc.Loading(dcc.Graph(id="p-map-graph"), type="dot"), style={"width": "49%", "display": "inline-block"}),
                 html.Div(dcc.Loading(dcc.Graph(id="p-bar-graph"), type="dot"), style={"width": "49%", "display": "inline-block", "float": "right"})
             ]),
             html.Div(id="p-explanation", style={"padding": "15px", "backgroundColor": "#f0fdf4", "borderRadius": "8px", "marginTop": "10px"})
        ], style={"marginBottom": "40px"}),

        html.Div([
            html.H4("Evapotranspiration Analysis", style={"color": "#0076a5"}),
             html.Div([
                 html.Div(dcc.Loading(dcc.Graph(id="et-map-graph"), type="dot"), style={"width": "49%", "display": "inline-block"}),
                 html.Div(dcc.Loading(dcc.Graph(id="et-bar-graph"), type="dot"), style={"width": "49%", "display": "inline-block", "float": "right"})
             ]),
             html.Div(id="et-explanation", style={"padding": "15px", "backgroundColor": "#f0fdf4", "borderRadius": "8px", "marginTop": "10px"})
        ], style={"marginBottom": "40px"}),

        html.Div([
            html.H4("Validation (Station vs Satellite)", style={"color": "#0076a5"}),
             html.Div([
                 html.Div(dcc.Loading(dcc.Graph(id="val-p-scatter"), type="dot"), style={"width": "49%", "display": "inline-block"}),
                 html.Div(dcc.Loading(dcc.Graph(id="val-et-scatter"), type="dot"), style={"width": "49%", "display": "inline-block", "float": "right"})
             ])
        ])
    ])


# ==================
# CALLBACKS
# ==================

@app.callback(
    Output("main-content-area", "children"),
    [Input("basin-dropdown", "value")]
)
def render_main_content(basin):
    if not basin or basin == "all":
        return html.Div(
            html.H3("Please select a basin from the map or dropdown to view the report and results.",
                   style={"textAlign": "center", "color": "#718096", "marginTop": "50px"})
        )

    return html.Div([
        get_introduction_layout(basin),
        get_objectives_layout(basin),
        get_methodology_layout(basin),
        get_results_layout(basin)
    ])

@app.callback(Output("tab-content", "children"), [Input("dashboard-tabs", "value")])
def render_tab_content(tab):
    if tab == "overview":
        return get_overview_tab_content()
    elif tab == "spatial":
        return get_spatial_tab_content()
    elif tab == "wa":
        return get_wa_tab_content()
    elif tab == "climate":
        return get_climate_tab_content()
    return html.Div("Tab not found")

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
        return empty_options, empty_value, empty_options, empty_value, ""

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
    except Exception:
        pass

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
        default_start = year_options[0]["value"]
        default_end = year_options[-1]["value"]
    else:
        default_start = common_min
        default_end = common_max

    files_found = [
        html.Span(f"P: {os.path.basename(p_fp) if p_fp else '❌'} | ", style={"marginRight": "10px"}),
        html.Span(f"ET: {os.path.basename(et_fp) if et_fp else '❌'} | ", style={"marginRight": "10px"}),
        html.Span(f"LU: {os.path.basename(lu_fp) if lu_fp else '❌'}")
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
        return html.Div("Select a basin and year range.")
    
    try:
        start_year, end_year = int(start_year), int(end_year)
        metrics = get_basin_overview_metrics_for_range(basin, start_year, end_year)
        if not metrics:
            return html.Div("Data not available for selected range.")

        def make_card(title, val, unit, color):
            return html.Div([
                html.P(title, style={"color": "#718096", "fontSize": "0.9rem", "margin": "0"}),
                html.H3(f"{val:,.0f} {unit}", style={"color": color, "margin": "5px 0", "fontWeight": "bold"})
            ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "boxShadow": "0 2px 5px rgba(0,0,0,0.05)", "borderLeft": f"5px solid {color}"})

        return html.Div([
            html.Div([
                make_card("Total Inflows", metrics.get('total_inflows', 0), "Mm³", "#3b82f6"),
                make_card("Precipitation", metrics.get('total_precipitation', 0), "Mm³", "#06b6d4"),
                make_card("Water Consumption", metrics.get('total_water_consumption', 0), "Mm³", "#ef4444"),
                make_card("Treated Wastewater", metrics.get('treated_wastewater', 0), "Mm³", "#10b981"),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "20px", "marginBottom": "20px"}),
            
            html.Div([
                html.P("Executive Summary", style={"fontWeight": "bold", "fontSize": "1.2rem", "color": "#0076a5"}),
                html.Ul([
                    html.Li(f"Total inflows: {metrics.get('total_inflows', 0):.0f} Mm³"),
                    html.Li(f"Precipitation: {metrics.get('total_precipitation', 0):.0f} Mm³ ({metrics.get('precipitation_percentage', 0):.1f}%)"),
                    html.Li(f"Total landscape consumption: {metrics.get('total_water_consumption', 0):.0f} Mm³"),
                    html.Li(f"Recharge: {metrics.get('recharge', 0):.0f} Mm³")
                ], style={"lineHeight": "1.8", "color": "#4a5568"})
            ], style={"backgroundColor": "#f0fdf4", "padding": "20px", "borderRadius": "12px"})
        ])
    except Exception as e:
        return html.Div(f"Error: {e}")

@app.callback(
    [Output("wa-resource-base-sankey", "figure"),
     Output("wa-sectoral-bar", "figure"),
     Output("wa-indicators-container", "children")],
    [Input("basin-dropdown", "value"),
     Input("global-start-year-dropdown", "value"),
     Input("global-end-year-dropdown", "value")]
)
def update_wa_module(basin, start_year, end_year):
    if not basin or basin == "all" or not start_year:
        return _empty_fig(), _empty_fig(), ""
    try:
        start_year, end_year = int(start_year), int(end_year)
        df = get_wa_data(basin, start_year, end_year)
        if df.empty: return _empty_fig("No Data"), _empty_fig(), "No Data"

        precip = df[(df['CLASS'] == 'INFLOW') & (df['SUBCLASS'] == 'PRECIPITATION')]['VALUE'].sum()
        sw_in = df[(df['CLASS'] == 'INFLOW') & (df['SUBCLASS'] == 'SURFACE WATER')]['VALUE'].sum()
        gw_in = df[(df['CLASS'] == 'INFLOW') & (df['SUBCLASS'] == 'GROUNDWATER')]['VALUE'].sum()

        et = df[(df['CLASS'] == 'OUTFLOW') & (df['SUBCLASS'].str.contains('ET'))]['VALUE'].sum()
        sw_out = df[(df['CLASS'] == 'OUTFLOW') & (df['SUBCLASS'] == 'SURFACE WATER')]['VALUE'].sum()
        gw_out = df[(df['CLASS'] == 'OUTFLOW') & (df['SUBCLASS'] == 'GROUNDWATER')]['VALUE'].sum()

        label = ["Precipitation", "SW Inflow", "GW Inflow", "Basin", "Evapotranspiration", "SW Outflow", "GW Outflow"]
        color = ["#60a5fa", "#3b82f6", "#1d4ed8", "#e2e8f0", "#fbbf24", "#3b82f6", "#1d4ed8"]

        source, target, value = [], [], []
        if precip > 0: source.append(0); target.append(3); value.append(precip)
        if sw_in > 0: source.append(1); target.append(3); value.append(sw_in)
        if gw_in > 0: source.append(2); target.append(3); value.append(gw_in)
        if et > 0: source.append(3); target.append(4); value.append(et)
        if sw_out > 0: source.append(3); target.append(5); value.append(sw_out)
        if gw_out > 0: source.append(3); target.append(6); value.append(gw_out)

        fig_sankey = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=label, color=color),
            link=dict(source=source, target=target, value=value)
        ))
        fig_sankey.update_layout(title_text=f"Resource Base ({start_year}-{end_year})", font_size=10)

        sector_df = df[
            (df['CLASS'] == 'OUTFLOW') &
            (df['SUBCLASS'].isin(['ET RAIN', 'ET INCREMENTAL'])) &
            (~df['VARIABLE'].isin(['Natural', 'Manmade', 'Consumed Water']))
        ]
        if sector_df.empty:
             sector_df = df[(df['CLASS'] == 'OUTFLOW') & (df['SUBCLASS'] == 'ET INCREMENTAL')]

        fig_bar = px.bar(sector_df, x='VARIABLE', y='VALUE', color='SUBCLASS', title="Water Consumption by Sector")
        fig_bar.update_layout(plot_bgcolor='white')

        ind_df = get_indicators(basin, start_year, end_year)
        inds = []
        if not ind_df.empty:
            for _, r in ind_df.iterrows():
                c_map = {'Red': '#ef4444', 'Orange': '#f97316', 'Green': '#22c55e'}
                c = c_map.get(r.get('TRAFFIC_LIGHT'), '#94a3b8')
                inds.append(html.Div([
                    html.H5(r['INDICATOR'], style={"margin": "0", "fontSize": "0.9rem"}),
                    html.H3(f"{r['VALUE']:.1f} {r['UNIT']}", style={"color": c, "margin": "5px 0"}),
                ], style={"display":"inline-block", "padding":"15px", "margin":"5px", "borderLeft":f"4px solid {c}", "backgroundColor": "white", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)", "width": "200px"}))

        return fig_sankey, fig_bar, inds
    except:
        return _empty_fig(), _empty_fig(), "Error"

def _generate_explanation(vtype, basin, sy, ey, vals, months):
    mean_val = np.nanmean(vals)
    return (f"**{vtype} Analysis ({sy}–{ey}):** "
            f"The average monthly value is **{mean_val:.2f} mm**. "
            f"This chart shows the seasonal distribution.")

def _hydro_figs(basin, start_year, end_year, vtype):
    if not basin or basin=="all" or not start_year:
        return _empty_fig(), _empty_fig(), ""
    
    da, _, _ = load_and_process_data(basin, vtype, year_start=int(start_year), year_end=int(end_year), aggregate_time=False)
    if da is None: return _empty_fig(), _empty_fig(), "No Data"
    
    da_map = da.mean(dim="time", skipna=True)
    cscale = "Blues" if "P" in vtype and "ET" not in vtype else "YlOrRd" if vtype=="ET" else "RdBu"
    fig_map = _create_clean_heatmap(da_map, f"Mean {vtype}", cscale, "mm")
    fig_map = add_shapefile_to_fig(fig_map, basin)
    
    spatial_mean = da.mean(dim=["latitude", "longitude"], skipna=True)
    try:
        monthly = spatial_mean.groupby("time.month").mean(skipna=True)
        months = [pd.to_datetime(m, format="%m").strftime("%b") for m in monthly.month.values]
        y_vals = monthly.values
        fig_bar = px.bar(x=months, y=y_vals, title=f"Monthly {vtype} Cycle", labels={"y": "mm"})
        fig_bar.update_layout(plot_bgcolor='white')
        explanation = _generate_explanation(vtype, basin, start_year, end_year, y_vals, months)
    except:
        fig_bar = _empty_fig("Time aggregation failed")
        explanation = "Error"

    return fig_map, fig_bar, dcc.Markdown(explanation)

@app.callback(
    [Output("p-map-graph", "figure"), Output("p-bar-graph", "figure"), Output("p-explanation", "children")],
    [Input("basin-dropdown", "value"), Input("global-start-year-dropdown", "value"), Input("global-end-year-dropdown", "value")]
)
def update_p(basin, sy, ey):
    return _hydro_figs(basin, sy, ey, "P")

@app.callback(
    [Output("et-map-graph", "figure"), Output("et-bar-graph", "figure"), Output("et-explanation", "children")],
    [Input("basin-dropdown", "value"), Input("global-start-year-dropdown", "value"), Input("global-end-year-dropdown", "value")]
)
def update_et(basin, sy, ey):
    return _hydro_figs(basin, sy, ey, "ET")

@app.callback(
    [Output("p-et-map-graph", "figure"), Output("p-et-bar-graph", "figure"), Output("p-et-explanation", "children")],
    [Input("basin-dropdown", "value"), Input("global-start-year-dropdown", "value"), Input("global-end-year-dropdown", "value")]
)
def update_pet(basin, sy, ey):
    return _hydro_figs(basin, sy, ey, "P-ET")

@app.callback(
    [Output("lu-map-graph", "figure"), Output("lu-bar-graph", "figure"), Output("lu-explanation", "children"), Output("lu-et-p-bar", "figure")],
    [Input("basin-dropdown", "value"), Input("global-start-year-dropdown", "value"), Input("global-end-year-dropdown", "value")]
)
def update_lu(basin, sy, ey):
    if not basin: return _empty_fig(), _empty_fig(), "", _empty_fig()
    
    da_lu, _, _ = load_and_process_data(basin, "LU", year_start=2020, year_end=2020)
    if da_lu is None: return _empty_fig(), _empty_fig(), "No LU Data", _empty_fig()
    
    # Map
    vals = da_lu.values
    z_clean, x, y = _clean_nan_data(da_lu)

    fig_map = go.Figure(go.Heatmap(z=z_clean, x=x, y=y, colorscale="Viridis", showscale=False))
    fig_map.update_layout(title="Land Use Map", plot_bgcolor='white')
    fig_map = add_shapefile_to_fig(fig_map, basin)

    # Bar
    flat = vals.flatten()
    u, c = np.unique(flat[np.isfinite(flat)], return_counts=True)
    df_lu = pd.DataFrame({"Class": u, "Count": c})
    df_lu["Name"] = df_lu["Class"].apply(lambda x: class_info.get(int(x), {}).get("name", str(x)))
    df_lu["Color"] = df_lu["Class"].apply(lambda x: class_info.get(int(x), {}).get("color", "gray"))
    df_lu = df_lu.sort_values("Count", ascending=False).head(5)

    fig_bar = px.bar(df_lu, x="Count", y="Name", orientation='h', title="Top 5 Land Use Classes")
    fig_bar.update_traces(marker_color=df_lu["Color"])
    fig_bar.update_layout(plot_bgcolor='white')

    # Coupling
    fig_coupling = _empty_fig("Coupling Analysis")
    if sy and ey:
        try:
             da_p, _, _ = load_and_process_data(basin, "P", year_start=int(sy), year_end=int(ey))
             da_et, _, _ = load_and_process_data(basin, "ET", year_start=int(sy), year_end=int(ey))
             if da_p is not None and da_et is not None:
                 # Simplified coupling: Just taking global means for now as detailed spatial alignment is complex
                 # and was causing issues in full implementation without precise alignment logic.
                 # Assuming aligned grids for demo.
                 fig_coupling = go.Figure(data=[
                     go.Bar(name='Precipitation', x=df_lu['Name'], y=[np.nanmean(da_p.values)]*len(df_lu)),
                     go.Bar(name='ET', x=df_lu['Name'], y=[np.nanmean(da_et.values)]*len(df_lu))
                 ])
                 fig_coupling.update_layout(title="Mean P vs ET (Basin Average)", plot_bgcolor='white')
        except:
             pass

    return fig_map, fig_bar, "Top Land Use Classes shown.", fig_coupling

@app.callback(
    [Output("val-p-scatter", "figure"), Output("val-et-scatter", "figure")],
    [Input("basin-dropdown", "value")]
)
def update_val(basin):
    if not basin or basin == "all": return _empty_fig(), _empty_fig()

    p_df = get_validation_data(basin, "P")
    et_df = get_validation_data(basin, "ET")

    def mk_scat(df, t):
        if df.empty: return _empty_fig(f"No {t} Data")
        fig = px.scatter(df, x='Observed', y='Satellite', hover_data=['Station'], title=t)
        fig.add_trace(go.Scatter(x=[0, df.Observed.max()], y=[0, df.Observed.max()], mode='lines', line=dict(color='red', dash='dash')))
        return fig

    return mk_scat(p_df, "P Validation"), mk_scat(et_df, "ET Validation")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
