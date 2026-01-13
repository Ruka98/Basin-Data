import streamlit as st
from streamlit_option_menu import option_menu
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
import base64

# =========================
# CONFIGURATION & STYLING
# =========================

st.set_page_config(
    page_title="Water Accounting Jordan",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# IWMI Colors
PRIMARY_COLOR = "#2B587A" # Updated Theme Color
THEME_COLOR = "#2B587A"
TEXT_COLOR = "#333333"
BG_COLOR = "#F8F9FA"

# Custom CSS
st.markdown(f"""
<style>
    /* Global Font */
    html, body, [class*="css"] {{
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        color: {TEXT_COLOR};
    }}

    /* Remove top padding */
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
    }}

    /* Header Styling */
    header {{
        visibility: hidden;
    }}

    /* Navbar Customization */
    .nav-link {{
        font-size: 16px !important;
        text-align: center !important;
        margin: 0px !important;
        padding: 10px !important;
    }}

    /* Card Styling */
    .stCard {{
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }}

    /* Hero Section */
    .hero-container {{
        background: linear-gradient(rgba(43, 88, 122, 0.7), rgba(43, 88, 122, 0.8)), url('https://images.unsplash.com/photo-1505144809822-ba3c40331397?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        padding: 6rem 2rem;
        color: white;
        text-align: center;
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
    }}

    .hero-title {{
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }}

    .hero-subtitle {{
        font-size: 1.5rem;
        font-weight: 300;
        max-width: 800px;
        margin: 0 auto;
    }}

    /* Metrics */
    div[data-testid="metric-container"] {{
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid {THEME_COLOR};
    }}

    /* Footer */
    .footer {{
        background-color: {THEME_COLOR};
        color: white;
        padding: 2rem;
        margin-top: 4rem;
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)


# =========================
# UTILITIES (Ported from dashboard.py)
# =========================

BASE_DIR = os.getcwd()
BASIN_DIR = os.path.join(BASE_DIR, "basins")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

def load_image_as_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

def _open_xr_dataset(fp: str) -> xr.Dataset:
    for eng in ("h5netcdf", "netcdf4", None):
        try:
            return xr.open_dataset(fp, decode_times=True, engine=eng)
        except Exception:
            pass
    return None

def _standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    if ds is None: return None
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
    exclude = {"time", "latitude", "longitude", "crs", "spatial_ref"}
    cands = [v for v in ds.data_vars if v not in exclude]
    if not cands: return None
    with_ll = [v for v in cands if {"latitude", "longitude"}.issubset(set(ds[v].dims))]
    return with_ll[0] if with_ll else cands[0]

def _first_existing(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort()
            return hits[-1]
    return None

def find_nc_file(basin_name: str, variable_type: str):
    netcdf_dir = os.path.join(BASIN_DIR, basin_name, "NetCDF")
    if not os.path.isdir(netcdf_dir): return None
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
    if not os.path.isdir(shp_dir): return None
    return _first_existing([os.path.join(shp_dir, "*.shp")])

def read_basin_text(basin_name: str, filename: str) -> str:
    path = os.path.join(BASIN_DIR, basin_name, filename)
    if os.path.exists(path):
         try:
            with open(path, 'r', encoding='utf-8') as f: return f.read()
         except: pass
    path = os.path.join(BASIN_DIR, basin_name, "text", filename)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f: return f.read()
        except: pass
    return f"No text available for {filename}."

def find_yearly_csv(basin_name: str, year: int):
    results_dir = os.path.join(BASIN_DIR, basin_name, "Results", "yearly")
    if not os.path.isdir(results_dir): return None
    patterns = [
        os.path.join(results_dir, f"sheet1_{year}.csv"),
        os.path.join(results_dir, f"*{year}*.csv"),
        os.path.join(results_dir, "*.csv")
    ]
    return _first_existing(patterns)

def parse_wa_sheet(csv_file: str):
    try:
        df = pd.read_csv(csv_file, sep=';')
        cleaned_rows = []
        for _, row in df.iterrows():
            try: val = float(row.get('VALUE', 0)) * 1000
            except: val = 0
            cleaned_rows.append({
                'CLASS': row.get('CLASS', '').strip(),
                'SUBCLASS': row.get('SUBCLASS', '').strip(),
                'VARIABLE': row.get('VARIABLE', '').strip(),
                'VALUE': val
            })
        return pd.DataFrame(cleaned_rows)
    except: return pd.DataFrame()

def get_wa_data(basin_name: str, start_year: int, end_year: int):
    all_data = []
    for year in range(start_year, end_year + 1):
        csv_file = find_yearly_csv(basin_name, year)
        if csv_file:
            df = parse_wa_sheet(csv_file)
            if not df.empty:
                df['Year'] = year
                all_data.append(df)
    if not all_data: return pd.DataFrame()
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df.groupby(['CLASS', 'SUBCLASS', 'VARIABLE'])['VALUE'].mean().reset_index()

def get_basin_overview_metrics(basin_name: str, start_year: int, end_year: int):
    agg_df = get_wa_data(basin_name, start_year, end_year)
    if agg_df.empty: return None
    metrics = {}
    metrics['total_inflows'] = agg_df[agg_df['CLASS'] == 'INFLOW']['VALUE'].sum()
    metrics['total_precipitation'] = agg_df[(agg_df['CLASS'] == 'INFLOW') & (agg_df['SUBCLASS'] == 'PRECIPITATION')]['VALUE'].sum()
    metrics['precipitation_rainfall'] = agg_df[(agg_df['CLASS'] == 'INFLOW') & (agg_df['SUBCLASS'] == 'PRECIPITATION') & (agg_df['VARIABLE'] == 'Rainfall')]['VALUE'].sum()
    metrics['surface_water_imports'] = agg_df[(agg_df['CLASS'] == 'INFLOW') & (agg_df['SUBCLASS'] == 'SURFACE WATER') & (agg_df['VARIABLE'].isin(['Main riverstem', 'Tributaries']))]['VALUE'].sum()
    et_rows = agg_df[(agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'].str.contains('ET'))]
    metrics['total_water_consumption'] = et_rows[~et_rows['VARIABLE'].isin(['Manmade', 'Consumed Water'])]['VALUE'].sum()
    metrics['manmade_consumption'] = agg_df[(agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'] == 'ET INCREMENTAL') & (agg_df['VARIABLE'] == 'Manmade')]['VALUE'].sum()
    metrics['non_irrigated_consumption'] = agg_df[(agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'] == 'ET INCREMENTAL') & (agg_df['VARIABLE'] == 'Consumed Water')]['VALUE'].sum()
    metrics['treated_wastewater'] = agg_df[(agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'] == 'OTHER') & (agg_df['VARIABLE'] == 'Treated Waste Water')]['VALUE'].sum()
    recharge_val = agg_df[(agg_df['CLASS'] == 'STORAGE') & (agg_df['SUBCLASS'] == 'CHANGE') & (agg_df['VARIABLE'].str.contains('Surface storage'))]['VALUE'].sum()
    metrics['recharge'] = abs(recharge_val) if recharge_val < 0 else recharge_val
    if metrics['total_inflows'] > 0:
        metrics['precipitation_percentage'] = (metrics['total_precipitation'] / metrics['total_inflows'] * 100)
    return metrics

def get_validation_data(basin_name: str, var_type: str):
    filename = "rainfall_validation.csv" if var_type == "P" else "et_validation.csv"
    filepath = os.path.join(BASIN_DIR, basin_name, "Results", "validation", filename)
    if not os.path.exists(filepath): return pd.DataFrame()
    try: return pd.read_csv(filepath, sep=';')
    except: return pd.DataFrame()

# Shapefile Helpers
def _force_2d(geom):
    try: return shp_wkb.loads(shp_wkb.dumps(geom, output_dimension=2))
    except: return geom

def _repair_poly(geom):
    try:
        g = geom.buffer(0)
        return g if (g is not None and not g.is_empty) else geom
    except: return geom

@st.cache_data
def load_all_basins_geodata() -> gpd.GeoDataFrame:
    rows = []
    if not os.path.isdir(BASIN_DIR): return gpd.GeoDataFrame(columns=["basin", "geometry"], geometry="geometry", crs="EPSG:4326")
    for b in sorted([d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))]):
        shp = find_shp_file(b)
        if not shp or not os.path.exists(shp): continue
        try:
            with fiona.open(shp) as src:
                crs_obj = "EPSG:4326"
                geoms = []
                for feat in src:
                    if not feat or not feat.get("geometry"): continue
                    geom = shp_shape(feat["geometry"])
                    geom = _force_2d(geom)
                    geom = _repair_poly(geom)
                    if geom and not geom.is_empty: geoms.append(geom)
                if not geoms: continue
                gdf = gpd.GeoDataFrame({"basin": [b]*len(geoms)}, geometry=geoms, crs=crs_obj)
                try: gdf = gdf.to_crs("EPSG:4326")
                except: gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)
                gdf = gdf.explode(index_parts=False).reset_index(drop=True)
                rows.append(gdf[["basin", "geometry"]])
        except: continue
    if not rows: return gpd.GeoDataFrame(columns=["basin", "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")

ALL_BASINS_GDF = load_all_basins_geodata()

def basins_geojson(gdf: gpd.GeoDataFrame | None = None):
    gdf = ALL_BASINS_GDF if gdf is None else gdf
    if gdf is None or gdf.empty: return {"type": "FeatureCollection", "features": []}
    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": {"basin": row["basin"]}})
    return {"type": "FeatureCollection", "features": feats}

def _coarsen_to_1km(da: xr.DataArray, is_categorical=False) -> xr.DataArray:
    # Simplified coarsen
    return da

def load_and_process_data(basin_name: str, variable_type: str, year_start: int=None, year_end: int=None, aggregate_time: bool = True):
    fp = find_nc_file(basin_name, variable_type)
    if not fp: return None, None, "NetCDF file not found"
    try:
        ds = _open_xr_dataset(fp)
        ds = _standardize_latlon(ds)
        var = _pick_data_var(ds)
        if not var: return None, None, "No variable"
        da = ds[var]
        if "time" in ds.coords and (year_start is not None or year_end is not None):
            ys = int(year_start) if year_start is not None else pd.to_datetime(ds["time"].values).min().year
            ye = int(year_end) if year_end is not None else pd.to_datetime(ds["time"].values).max().year
            da = da.sel(time=slice(f"{ys}-01-01", f"{ye}-12-31"))
        if "time" in da.dims and aggregate_time and da.sizes.get("time", 0) > 1 and variable_type in ["P", "ET"]:
            da = da.mean(dim="time", skipna=True)
        elif variable_type == "LU" and "time" in da.dims:
            da = da.isel(time=-1)
        da = _coarsen_to_1km(da, is_categorical=(variable_type=="LU"))
        return da, var, os.path.basename(fp)
    except Exception as e: return None, None, str(e)

# Plotting Helpers
def _clean_nan_data(da: xr.DataArray):
    if da is None: return None, None, None
    valid_mask = np.isfinite(da.values)
    if not np.any(valid_mask): return None, None, None
    x = np.asarray(da["longitude"].values)
    y = np.asarray(da["latitude"].values)
    z_clean = da.values.copy()
    return z_clean, x, y

def _create_clean_heatmap(da: xr.DataArray, title: str, colorscale="Viridis", z_label="value"):
    z, x, y = _clean_nan_data(da)
    if z is None: return go.Figure()
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=colorscale, hoverinfo="x+y+z"))
    fig.update_layout(title=title, xaxis_title="Lon", yaxis_title="Lat", margin=dict(l=0, r=0, t=30, b=0))
    return fig

def add_shapefile_to_fig(fig: go.Figure, basin_name: str) -> go.Figure:
    shp_file = find_shp_file(basin_name)
    if not shp_file: return fig
    try:
        gdf = gpd.read_file(shp_file).to_crs("EPSG:4326")
        for geom in gdf.geometry:
            if geom.geom_type == "Polygon":
                x, y = geom.exterior.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(color="black", width=1), showlegend=False))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(color="black", width=1), showlegend=False))
    except: pass
    return fig

def make_basin_selector_map(selected_basin=None):
    gdf = ALL_BASINS_GDF if (not selected_basin or selected_basin == "all") else ALL_BASINS_GDF[ALL_BASINS_GDF["basin"] == selected_basin]
    if gdf.empty: return go.Figure()
    gj = basins_geojson(gdf)
    locations = [f["properties"]["basin"] for f in gj["features"]]
    z_vals = [1] * len(locations)

    # Colors
    fill_color = "rgba(43, 88, 122, 0.4)" # Updated Theme color alpha
    line_color = THEME_COLOR

    ch = go.Choroplethmapbox(
        geojson=gj, locations=locations, featureidkey="properties.basin", z=z_vals,
        colorscale=[[0, fill_color], [1, fill_color]],
        marker=dict(line=dict(width=2, color=line_color)),
        showscale=False, hovertemplate="<b>%{location}</b><extra></extra>"
    )
    fig = go.Figure(ch)

    # Center map
    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2

    fig.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lon=center_lon, lat=center_lat), zoom=6),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    return fig

# =========================
# PAGES
# =========================

def render_home():
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Rapid Water Accounting Dashboard - Jordan</div>
            <div class="hero-subtitle">Empowering sustainable water management through advanced remote sensing data and hydrological modeling.</div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📊 **Basin Analysis**\n\nInteractive maps and metrics for major basins in Jordan. Analyze inflows, outflows, and storage changes.")
    with c2:
        st.info("🌧️ **Climate Data**\n\nVisualize long-term precipitation and evapotranspiration trends derived from high-resolution satellite data.")
    with c3:
        st.info("📑 **WA+ Reporting**\n\nStandardized Water Accounting Plus (WA+) sheets and indicators to support evidence-based decision making.")

def render_about():
    st.title("Introduction")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.header("Introduction")
        with open("assets/intro.txt", "r") as f:
            st.markdown(f.read())
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.header("Objectives")
        with open("assets/objectives.txt", "r") as f:
            st.markdown(f.read())
        st.markdown('</div>', unsafe_allow_html=True)

def render_framework():
    st.title("Scientific Framework")
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown("### Customized WA+ Analytics for Jordan")

    st.markdown(r"""
    WA+ is a robust framework that harnesses the potential of publicly available remote sensing data to assess water resources and their consumption. Its reliance on such data is particularly beneficial in data scarce areas and transboundary basins. A significant benefit of WA+ lies in its incorporation of land use classification into water resource assessments, promoting a holistic approach to land and water management.

    The updated water balance equation for Jordan:

    $$ \Delta S/\Delta t = (P + Q_{in}) - (ET + CW_{sec} + Q_{WWT} + Q_{re} + Q_{natural}) $$

    Where:
    *   **P**: Precipitation
    *   **ET**: Evapotranspiration
    *   **Qin**: Total inflows
    *   **CWsec**: Non-irrigated water consumption (Domestic, Industrial, Livestock, Tourism)
    *   **QWWT**: Treated wastewater
    *   **Qre**: Groundwater recharge
    *   **Qnatural**: Naturalized streamflow
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def render_analysis():
    st.title("Water Accounting Analysis")

    # Sidebar or Top Selector for Basin
    basins = sorted([d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))]) if os.path.exists(BASIN_DIR) else []

    col_sel, col_map = st.columns([1, 2])

    with col_sel:
        selected_basin = st.selectbox("Select Basin", ["Select a Basin..."] + basins)

        if selected_basin and selected_basin != "Select a Basin...":
            # Show study area text
            st.markdown(f"**{selected_basin} Overview**")
            sa_text = read_basin_text(selected_basin, "study area.txt")
            if "No text" in sa_text: sa_text = read_basin_text(selected_basin, "studyarea.txt")
            st.caption(sa_text[:500] + "...")

    with col_map:
        fig_map = make_basin_selector_map(selected_basin if selected_basin != "Select a Basin..." else None)
        st.plotly_chart(fig_map, use_container_width=True)

    if not selected_basin or selected_basin == "Select a Basin...":
        st.warning("Please select a basin to view the analysis.")
        return

    # Tabs for Analysis
    tab1, tab2, tab3 = st.tabs(["Land Use", "Climate Inputs", "Results & Balance"])

    # Global Year Selection for this Basin
    # Get available years
    p_fp = find_nc_file(selected_basin, "P")
    try:
        ds = _open_xr_dataset(p_fp)
        min_y, max_y = pd.to_datetime(ds.time.values).min().year, pd.to_datetime(ds.time.values).max().year
    except:
        min_y, max_y = 2000, 2020

    with st.container():
        st.markdown("#### Analysis Period")
        c_y1, c_y2 = st.columns(2)
        start_year = c_y1.number_input("Start Year", min_value=1980, max_value=2030, value=min_y)
        end_year = c_y2.number_input("End Year", min_value=1980, max_value=2030, value=max_y)

    with tab1:
        st.subheader("Land Use Analysis")
        # Reuse logic
        da_lu, _, _ = load_and_process_data(selected_basin, "LU", 2020, 2020) # Latest
        if da_lu is not None:
             fig = px.imshow(da_lu.values, origin='lower', title="Land Use Map")
             fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
             st.plotly_chart(fig, use_container_width=True)

             # Table
             if selected_basin == "Amman Zarqa":
                  # Hardcoded example from dashboard.py
                  st.table(pd.DataFrame([
                      {"Class": "Natural", "Subclass": "Protected forests", "Area (km2)": 6.14, "P (mm)": 552.2},
                      {"Class": "Agricultural", "Subclass": "Rainfed crops", "Area (km2)": 208.90, "P (mm)": 285.7},
                      {"Class": "Urban", "Subclass": "Paved Surface", "Area (km2)": 345.97, "P (mm)": 268.2},
                  ]))
        else:
             st.info("No Land Use Data Available")

    with tab2:
        st.subheader("Climate Inputs (P & ET)")
        c_p, c_et = st.columns(2)

        # P
        da_p, _, _ = load_and_process_data(selected_basin, "P", start_year, end_year, aggregate_time=True)
        if da_p is not None:
             c_p.plotly_chart(_create_clean_heatmap(da_p, "Average Precipitation (mm)"), use_container_width=True)
        else: c_p.warning("No Precipitation Data")

        # ET
        da_et, _, _ = load_and_process_data(selected_basin, "ET", start_year, end_year, aggregate_time=True)
        if da_et is not None:
             c_et.plotly_chart(_create_clean_heatmap(da_et, "Average ET (mm)"), use_container_width=True)
        else: c_et.warning("No ET Data")

        # Validation
        st.markdown("#### Validation")
        val_p = get_validation_data(selected_basin, "P")
        if not val_p.empty:
            st.plotly_chart(px.scatter(val_p, x='Observed', y='Satellite', title="Rainfall Validation"), use_container_width=True)

    with tab3:
        st.subheader("Results: Water Balance & Accounting")

        # Metrics
        metrics = get_basin_overview_metrics(selected_basin, start_year, end_year)
        if metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Inflows", f"{metrics['total_inflows']:.0f} Mm3")
            m2.metric("Precipitation", f"{metrics['total_precipitation']:.0f} Mm3")
            m3.metric("Consumption", f"{metrics['total_water_consumption']:.0f} Mm3")
            m4.metric("Recharge", f"{metrics['recharge']:.0f} Mm3")

        # P-ET Map
        if da_p is not None and da_et is not None:
             try:
                 da_p_align, da_et_align = xr.align(da_p, da_et, join="inner")
                 da_pet = da_p_align - da_et_align
                 st.plotly_chart(_create_clean_heatmap(da_pet, "Water Balance (P - ET)", colorscale="RdBu"), use_container_width=True)
             except: pass

        # WA+ Data
        st.markdown("#### Water Accounting Indicators")
        df_wa = get_wa_data(selected_basin, start_year, end_year)
        if not df_wa.empty:
            st.bar_chart(df_wa, x="VARIABLE", y="VALUE", color="SUBCLASS")


# =========================
# MAIN APP
# =========================

def main():
    # Header with Logos
    c_logo1, c_title, c_logo2 = st.columns([1, 4, 1])
    with c_logo1:
        if os.path.exists("assets/iwmi.png"):
            st.image("assets/iwmi.png", width=150)
    with c_title:
        pass # Title is handled in Navbar or Hero
    with c_logo2:
        if os.path.exists("assets/cgiar.png"):
            st.image("assets/cgiar.png", width=150)

    # Navigation
    selected_page = option_menu(
        menu_title=None,
        options=["Home", "Introduction", "Framework", "WA+ Analysis"],
        icons=["house", "info-circle", "diagram-3", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": PRIMARY_COLOR},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "center", "margin": "0px", "--hover-color": "#4A7C9D", "color": "white"},
            "nav-link-selected": {"background-color": "#1F4E79"},
        }
    )

    # Page Routing
    if selected_page == "Home":
        render_home()
    elif selected_page == "Introduction":
        render_about()
    elif selected_page == "Framework":
        render_framework()
    elif selected_page == "WA+ Analysis":
        render_analysis()

    # Footer
    st.markdown(f"""
    <div class="footer">
        <p>© 2024 International Water Management Institute (IWMI). All rights reserved.</p>
        <p>Science for a water-secure world.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
