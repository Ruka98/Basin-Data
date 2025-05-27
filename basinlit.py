import streamlit as st
import ee
import datetime
import logging
import json
import geopandas as gpd
from shapely.geometry import mapping
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import fiona
from fiona import Env
import uuid

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Earth Engine
@st.cache_resource
def initialize_ee(json_file):
    try:
        if json_file is not None:
            service_account_info = json.load(json_file)
            credentials = ee.ServiceAccountCredentials(
                service_account_info['client_email'],
                key_data=json.dumps(service_account_info)
            )
            ee.Initialize(credentials)
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Earth Engine initialization failed: {e}")
        return False

def main():
    st.set_page_config(page_title="Basin Data Visualizer", layout="wide")
    st.title("Basin Data Visualizer")

    # Available variables and their data sources
    data_sources = {
        'Precipitation': ['CHIRPS'],
        'ETa': ['ETA_V6'],
        'ETref': ['WAPOR'],
        'LAI': ['MODIS15']
    }

    # Dataset configurations
    dataset_configs = {
        'Precipitation': {
            'CHIRPS': {
                'collection': 'UCSB-CHG/CHIRPS/DAILY',
                'band': 'precipitation',
                'scale': 5566,
                'aggregation': 'sum',
                'unit': 'mm',
                'label': 'Precipitation'
            }
        },
        'ETa': {
            'ETA_V6': {
                'collection': 'FAO/WAPOR/2/L1_AETI_D',
                'band': 'L1_AETI_D',
                'scale': 250,
                'aggregation': 'mean',
                'unit': 'mm',
                'label': 'Actual Evapotranspiration'
            }
        },
        'ETref': {
            'WAPOR': {
                'collection': 'FAO/WAPOR/2/L1_RET_E',
                'band': 'L1_RET_E',
                'scale': 250,
                'aggregation': 'mean',
                'unit': 'mm',
                'label': 'Reference Evapotranspiration'
            }
        },
        'LAI': {
            'MODIS15': {
                'collection': 'MODIS/006/MCD15A3H',
                'band': 'Lai',
                'scale': 500,
                'aggregation': 'mean',
                'unit': '',
                'label': 'Leaf Area Index'
            }
        }
    }

    # Sidebar for inputs
    st.sidebar.header("Configuration")

    # Service Account JSON Upload
    json_file = st.sidebar.file_uploader("Upload Service Account JSON", type=["json"])
    ee_initialized = initialize_ee(json_file)
    if not ee_initialized:
        st.sidebar.error("Please upload a valid service account JSON file.")
        return

    # Year Range Selection
    st.sidebar.subheader("Time Period")
    start_year = st.sidebar.number_input("Start Year (1981 onwards)", min_value=1981, max_value=2024, value=1981)
    end_year = st.sidebar.number_input("End Year (up to 2024)", min_value=1981, max_value=2024, value=2024)
    
    if start_year > end_year:
        st.sidebar.error("Start year must be less than or equal to end year.")
        return

    # Shapefile Upload
    st.sidebar.subheader("Shapefile Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload shapefile and its components (.shp, .shx, .dbf, .prj)",
        type=["shp", "shx", "dbf", "prj"],
        accept_multiple_files=True
    )

    # Validate and process shapefile components
    if not uploaded_files:
        st.sidebar.error("Please upload the .shp file and its accompanying files (.shx, .dbf, .prj).")
        return

    try:
        temp_dir = "temp_shapefiles"
        os.makedirs(temp_dir, exist_ok=True)

        # Find the .shp file and its base name
        shp_file = next((f for f in uploaded_files if f.name.lower().endswith('.shp')), None)
        if not shp_file:
            st.sidebar.error("No .shp file found in uploaded files. Please include the .shp file.")
            return

        shp_base_name = os.path.splitext(shp_file.name)[0]
        shp_path = os.path.join(temp_dir, shp_file.name)
        with open(shp_path, "wb") as f:
            f.write(shp_file.getvalue())

        # Check for required accompanying files (.shx, .dbf)
        required_extensions = ['.shx', '.dbf']
        uploaded_extensions = [os.path.splitext(f.name)[1].lower() for f in uploaded_files]
        missing_extensions = [ext for ext in required_extensions if ext not in uploaded_extensions]

        # Save all uploaded files
        additional_file_paths = {}
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            additional_file_paths[os.path.splitext(file.name)[1].lower()] = file_path

        # Validate required files and provide feedback
        if missing_extensions:
            missing_str = ", ".join(missing_extensions)
            if '.shx' in missing_extensions:
                st.sidebar.warning(f"Missing {missing_str}. Attempting to restore .shx with SHAPE_RESTORE_SHX=YES.")
            else:
                st.sidebar.error(f"Missing required files: {missing_str}. Please upload them.")
                return

        # Read shapefile with SHAPE_RESTORE_SHX enabled
        with Env(SHAPE_RESTORE_SHX='YES'):
            gdf = gpd.read_file(shp_path)
            gdf = gdf[gdf.geometry.is_valid]  # Keep only valid geometries
            if gdf.empty:
                st.sidebar.error("No valid geometries found in shapefile.")
                return
            gdf = gdf.to_crs(epsg=4326)  # Ensure CRS is WGS84 for Folium
            geometry = gdf.geometry.union_all()
            ee_geometry = ee.Geometry(mapping(geometry))
            logger.debug(f"Geometry bounds: {gdf.geometry.total_bounds}")
    except Exception as e:
        st.sidebar.error(f"Error reading shapefile: {str(e)}")
        return

    # Dataset Selection
    st.sidebar.subheader("Dataset Selection")
    selected_variable = st.sidebar.selectbox("Select Variable", list(data_sources.keys()))
    available_sources = data_sources[selected_variable]
    selected_source = st.sidebar.selectbox("Select Source", available_sources)

    # Initialize session state for selected datasets
    if 'selected_datasets' not in st.session_state:
        st.session_state.selected_datasets = []

    # Add dataset
    if st.sidebar.button("Add Dataset"):
        dataset = f"{selected_variable} ({selected_source})"
        if dataset not in st.session_state.selected_datasets:
            st.session_state.selected_datasets.append(dataset)
            st.sidebar.success(f"Added dataset: {dataset}")
        else:
            st.sidebar.warning("Dataset already selected!")

    # Display and remove selected datasets
    if st.session_state.selected_datasets:
        st.sidebar.subheader("Selected Datasets")
        for dataset in st.session_state.selected_datasets:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(dataset)
            if col2.button("Remove", key=f"remove_{dataset}_{str(uuid.uuid4())}"):
                st.session_state.selected_datasets.remove(dataset)
                st.sidebar.success(f"Removed dataset: {dataset}")
                # Removed st.rerun() to avoid resetting the map

    # Initialize session state for visualizations
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = {
            'map': None,
            'map_state': {'center': None, 'zoom': 8},
            'plots': {},
            'data_for_charts': None,
            'labels': None
        }

    # Clear Visualizations Button
    if st.session_state.visualizations['map'] or st.session_state.visualizations['plots']:
        if st.sidebar.button("Clear Visualizations"):
            st.session_state.visualizations = {
                'map': None,
                'map_state': {'center': None, 'zoom': 8},
                'plots': {},
                'data_for_charts': None,
                'labels': None
            }
            # Removed st.rerun() to avoid resetting the map

    # Visualize Data Button
    if st.sidebar.button("Visualize Basin Data"):
        if not st.session_state.selected_datasets:
            st.error("Please select at least one dataset.")
            return

        try:
            with st.spinner("Processing data..."):
                total_months = (end_year - start_year + 1) * 12
                progress_bar = st.progress(0)
                current_progress = 0

                data_for_charts = {}
                labels = []
                current_date = datetime.datetime(start_year, 1, 1)
                end_date = datetime.datetime(end_year + 1, 1, 1)
                while current_date < end_date:
                    labels.append(current_date.strftime('%Y-%m'))
                    current_date += relativedelta(months=1)

                for dataset in st.session_state.selected_datasets:
                    var, source = dataset.split(' (')
                    source = source[:-1]
                    config = dataset_configs[var][source]
                    dataset_coll = ee.ImageCollection(config['collection']) \
                        .filterDate(f"{start_year}-01-01", f"{end_year + 1}-01-01") \
                        .filterBounds(ee_geometry)

                    st.write(f"Processing {var} ({source}) data...")
                    monthly_values = []
                    current_date = datetime.datetime(start_year, 1, 1)
                    while current_date < end_date:
                        month_start = current_date.strftime('%Y-%m-01')
                        month_end = (current_date + relativedelta(months=1)).strftime('%Y-%m-01')
                        monthly_data = dataset_coll.filterDate(month_start, month_end)
                        size = monthly_data.size().getInfo()
                        value = 0
                        if size == 0:
                            st.warning(f"No data for {var} ({source}) for {month_start}. Using 0.")
                        else:
                            agg_func = monthly_data.sum() if config['aggregation'] == 'sum' else monthly_data.mean()
                            monthly_agg = agg_func.select(config['band'])
                            value_dict = monthly_agg.reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=ee_geometry,
                                scale=config['scale'],
                                maxPixels=1e10
                            ).getInfo()
                            value = value_dict.get(config['band'], 0) or 0
                        monthly_values.append(round(value, 2))
                        current_progress += 1
                        progress_bar.progress(current_progress / (total_months * len(st.session_state.selected_datasets)))
                        current_date += relativedelta(months=1)

                    data_for_charts[dataset] = {
                        'values': monthly_values,
                        'unit': config['unit'],
                        'label': config['label']
                    }

                # Generate Map
                if st.session_state.visualizations['map'] is None:
                    centroid = gdf.geometry.centroid.iloc[0]
                    logger.debug(f"Centroid: {centroid.y}, {centroid.x}")
                    m = folium.Map(
                        location=[centroid.y, centroid.x],
                        zoom_start=8
                    )
                    folium.GeoJson(gdf).add_to(m)
                    st.session_state.visualizations['map'] = m
                    st.session_state.visualizations['map_state'] = {'center': [centroid.y, centroid.x], 'zoom': 8}
                else:
                    m = st.session_state.visualizations['map']

                # Generate Plots
                plots = {}
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                for idx, dataset in enumerate(data_for_charts):
                    var, source = dataset.split(' (')
                    source = source[:-1]
                    unit = data_for_charts[dataset]['unit']
                    chart_title = f"Monthly {data_for_charts[dataset]['label']} Variation in Basin"
                    y_label = f"{data_for_charts[dataset]['label']} ({unit})" if unit else data_for_charts[dataset]['label']

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(labels, data_for_charts[dataset]['values'], color=colors[idx % len(colors)], 
                            linewidth=2, marker='o', markersize=4, label=data_for_charts[dataset]['label'])
                    ax.set_title(chart_title, fontsize=16)
                    ax.set_xlabel("Time (Year-Month)", fontsize=14)
                    ax.set_ylabel(y_label, fontsize=14)
                    ax.grid(True)
                    ax.legend()
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plots[dataset] = fig

                # Store visualizations in session state
                st.session_state.visualizations['plots'] = plots
                st.session_state.visualizations['data_for_charts'] = data_for_charts
                st.session_state.visualizations['labels'] = labels

                st.success("Visualization completed! Check the map and graphs below.")

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            st.error(f"Failed to visualize data: {str(e)}")

    # Display stored visualizations
    if st.session_state.visualizations['map']:
        st.subheader("Basin Map")
        try:
            map_output = st_folium(
                st.session_state.visualizations['map'],
                width=700,
                height=400,
                key=f"basin_map_{str(uuid.uuid4())}",
                returned_objects=["center", "zoom"]
            )
            logger.debug(f"Map output: {map_output}")
            if map_output and "center" in map_output and "zoom" in map_output:
                st.session_state.visualizations['map_state'] = {
                    'center': [map_output["center"]["lat"], map_output["center"]["lng"]],
                    'zoom': map_output["zoom"]
                }
        except Exception as e:
            st.error(f"Error rendering map: {str(e)}")
            logger.error(f"Map rendering failed: {e}")

    if st.session_state.visualizations['plots']:
        for dataset, fig in st.session_state.visualizations['plots'].items():
            st.subheader(f"Plot for {dataset}")
            st.pyplot(fig)
            plt.close(fig)

    # Download CSV Button
    if st.session_state.visualizations['data_for_charts']:
        df_dict = {'Time': st.session_state.visualizations['labels']}
        for dataset in st.session_state.visualizations['data_for_charts']:
            df_dict[dataset] = st.session_state.visualizations['data_for_charts'][dataset]['values']
        df = pd.DataFrame(df_dict)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="basin_data.csv",
            mime="text/csv",
            key="download_csv"
        )

if __name__ == "__main__":
    main()