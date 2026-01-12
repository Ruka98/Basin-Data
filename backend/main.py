
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import pandas as pd
import numpy as np
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
import json

from backend.utils import (
    BASIN_DIR, find_nc_file, _open_xr_dataset, load_and_process_data,
    read_basin_intro, get_basin_overview_metrics_for_range,
    add_shapefile_to_fig, basins_geojson, _create_clean_heatmap,
    _clean_nan_data, class_info, _empty_fig
)

app = FastAPI(title="Basin Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Basin(BaseModel):
    name: str

class YearRange(BaseModel):
    start_year: int
    end_year: int
    available_years: List[int]
    files_found: Dict[str, bool]

class OverviewMetrics(BaseModel):
    metrics: Dict[str, float]
    intro: str
    summary_items: List[str]

class ChartData(BaseModel):
    map_figure: Dict[str, Any]
    bar_figure: Dict[str, Any]
    explanation: str

# Endpoints

@app.get("/api/basins", response_model=List[str])
def get_basins():
    """List all available basins."""
    if not os.path.isdir(BASIN_DIR):
        return []
    basins = sorted([d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))])
    return basins

@app.get("/api/basins/map")
def get_basins_map_data():
    """Get GeoJSON for all basins."""
    return basins_geojson()

@app.get("/api/basins/{basin_name}/years", response_model=YearRange)
def get_basin_years(basin_name: str):
    """Get available years for a basin based on NetCDF files."""
    if not os.path.isdir(os.path.join(BASIN_DIR, basin_name)):
        raise HTTPException(status_code=404, detail="Basin not found")

    p_fp = find_nc_file(basin_name, "P")
    et_fp = find_nc_file(basin_name, "ET")
    lu_fp = find_nc_file(basin_name, "LU")

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
        # LU years check skipped for range to simplify, usually less critical for range
    except Exception as e:
        print(f"[WARN] Year scan error: {e}")

    common_min = max(p_min_yr, et_min_yr)
    common_max = min(p_max_yr, et_max_yr)
    if common_min > common_max:
        common_min = min(p_min_yr, et_min_yr)
        common_max = max(p_max_yr, et_max_yr)

    available_years = list(range(common_min, common_max + 1))

    return {
        "start_year": common_min,
        "end_year": common_max,
        "available_years": available_years,
        "files_found": {
            "P": p_fp is not None,
            "ET": et_fp is not None,
            "LU": lu_fp is not None
        }
    }

@app.get("/api/basins/{basin_name}/overview")
def get_basin_overview(basin_name: str, start_year: int, end_year: int):
    """Get overview metrics and intro text."""
    metrics = get_basin_overview_metrics_for_range(basin_name, start_year, end_year)
    intro = read_basin_intro(basin_name)

    if metrics is None:
        metrics = {}

    # Helper for summary generation (logic from dashboard.py)
    total_inflows = f"{metrics.get('total_inflows', 0):.0f}"
    precip_pct = f"{metrics.get('precipitation_percentage', 0):.0f}"
    imports = f"{metrics.get('surface_water_imports', 0):.0f}"
    total_consumption = f"{metrics.get('total_water_consumption', 0):.0f}"
    manmade_consumption = f"{metrics.get('manmade_consumption', 0):.0f}"
    treated_wastewater = f"{metrics.get('treated_wastewater', 0):.0f}"
    non_irrigated = f"{metrics.get('non_irrigated_consumption', 0):.0f}"
    recharge = f"{metrics.get('recharge', 0):.0f}"

    year_range_str = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"

    summary_items = [
        f"The total water inflows into the {basin_name} basin in {year_range_str} is {total_inflows} Mm3/ year.",
        f"Precipitation accounts for {precip_pct}% of the gross inflows and the remaining from imports for domestic purposes.",
        f"{imports} Mm3/ year of water is imported into the basin for domestic use.",
        f"The total landscape water consumption is {total_consumption} Mm3/ year.",
        f"The manmade water consumption is {manmade_consumption} Mm3/ year",
        f"About {treated_wastewater} Mm3/ year of treated wastewater that is discharged to streams.",
        f"The average sectorial non-irrigated water consumption is {non_irrigated} Mm3/ year.",
        f"On average {recharge} Mm3/ year recharged the basin."
    ]

    return {
        "metrics": metrics,
        "intro": intro,
        "summary_items": summary_items
    }

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

@app.get("/api/basins/{basin_name}/hydro/{variable}", response_model=ChartData)
def get_hydro_charts(basin_name: str, variable: str, start_year: int, end_year: int):
    """Get map and bar charts for P, ET, or P-ET."""
    if variable not in ["P", "ET", "P-ET"]:
        raise HTTPException(status_code=400, detail="Invalid variable. Must be P, ET, or P-ET")

    ys, ye = start_year, end_year
    if ys > ye:
        ys, ye = ye, ys

    if variable == "P-ET":
        da_p_ts, _, _ = load_and_process_data(basin_name, "P",  year_start=ys, year_end=ye, aggregate_time=False)
        da_et_ts, _, _ = load_and_process_data(basin_name, "ET", year_start=ys, year_end=ye, aggregate_time=False)

        if da_p_ts is None or da_et_ts is None:
             return {
                "map_figure": json.loads(_empty_fig(f"Data missing for {variable}").to_json()),
                "bar_figure": json.loads(_empty_fig().to_json()),
                "explanation": "Data unavailable."
            }

        da_p_aligned, da_et_aligned = xr.align(da_p_ts, da_et_ts, join="inner")
        if da_p_aligned.sizes.get("time", 0) == 0:
            return {
                "map_figure": json.loads(_empty_fig("No overlapping time steps").to_json()),
                "bar_figure": json.loads(_empty_fig().to_json()),
                "explanation": "No overlapping data."
            }

        da_ts = da_p_aligned - da_et_aligned
        da_map = da_ts.mean(dim="time", skipna=True)
        title = f"Mean Water Yield (P-ET) ({ys}–{ye})"
        colorscale = "RdBu"
    else:
        da_ts, _, msg = load_and_process_data(basin_name, variable, year_start=ys, year_end=ye, aggregate_time=False)
        if da_ts is None:
             return {
                "map_figure": json.loads(_empty_fig(f"Data missing for {variable}: {msg}").to_json()),
                "bar_figure": json.loads(_empty_fig().to_json()),
                "explanation": "Data unavailable."
            }
        da_map = da_ts.mean(dim="time", skipna=True)
        title = f"Mean {variable} ({ys}–{ye})"
        colorscale = "Blues" if variable == "P" else "YlOrRd"

    # Map
    fig_map = _create_clean_heatmap(da_map, title=title, colorscale=colorscale, z_label="mm")
    fig_map = add_shapefile_to_fig(fig_map, basin_name)

    # Bar Chart
    spatial_dims = [d for d in ["latitude", "longitude"] if d in da_ts.dims]
    spatial_mean_ts = da_ts.mean(dim=spatial_dims, skipna=True)
    explanation = ""

    try:
        monthly = spatial_mean_ts.groupby("time.month").mean(skipna=True).rename({"month": "Month"})
        months = [pd.to_datetime(m, format="%m").strftime("%b") for m in monthly["Month"].values]
        y_vals = np.asarray(monthly.values).flatten()

        if np.isfinite(y_vals).any():
            fig_bar = px.bar(x=months, y=y_vals, title=f"Mean Monthly {variable} ({ys}–{ye})",
                             labels={"x": "Month", "y": f"Mean Daily {variable} (mm)"})
            # Modern bar chart styling
            color = "#ef4444" if variable == "P-ET" and np.mean(y_vals) < 0 else "#3b82f6"
            if variable == "P-ET":
                 # Conditional coloring for P-ET
                 colors = ["#ef4444" if v < 0 else "#10b981" for v in y_vals]
                 fig_bar.update_traces(marker_color=colors, marker_line_color='#1e293b', marker_line_width=1, opacity=0.8)
            else:
                 fig_bar.update_traces(marker_color=color, marker_line_color='#1d4ed8', marker_line_width=1, opacity=0.8)

            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#1e293b")
            )
            explanation = _generate_explanation(variable, basin_name, ys, ye, y_vals, months)
        else:
             fig_bar = _empty_fig(f"No valid monthly data for {variable}.")
             explanation = "No valid data."
    except Exception as e:
        print(f"Error generating bar chart for {variable}: {e}")
        fig_bar = _empty_fig(f"Error generating bar chart.")
        explanation = "Error generating explanation."

    return {
        "map_figure": json.loads(fig_map.to_json()),
        "bar_figure": json.loads(fig_bar.to_json()),
        "explanation": explanation
    }

@app.get("/api/basins/{basin_name}/landuse", response_model=ChartData)
def get_landuse_charts(basin_name: str):
    """Get Land Use map and bar chart."""
    lu_fp = find_nc_file(basin_name, "LU")
    if not lu_fp:
        return {
                "map_figure": json.loads(_empty_fig("Land Use data not found").to_json()),
                "bar_figure": json.loads(_empty_fig().to_json()),
                "explanation": "Data missing."
            }

    try:
        with _open_xr_dataset(lu_fp) as ds:
            if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
                t = pd.to_datetime(ds["time"].values)
                latest_year = int(t.max().year)
            else:
                latest_year = 2020
    except Exception:
        latest_year = 2020

    da, _, msg = load_and_process_data(basin_name, "LU", year_start=latest_year, year_end=latest_year)
    if da is None:
         return {
                "map_figure": json.loads(_empty_fig(f"Data unavailable: {msg}").to_json()),
                "bar_figure": json.loads(_empty_fig().to_json()),
                "explanation": "Data unavailable."
            }

    vals = np.asarray(da.values)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
         return {
                "map_figure": json.loads(_empty_fig("No valid data").to_json()),
                "bar_figure": json.loads(_empty_fig().to_json()),
                "explanation": "No valid data."
            }

    # Map
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
    fig_map = add_shapefile_to_fig(fig_map, basin_name)

    # Bar Chart
    try:
        unique, counts = np.unique(finite_vals, return_counts=True)
        total_pixels = counts.sum()

        lu_stats = []
        for u, c in zip(unique, counts):
            cid = int(u)
            cname = class_info.get(cid, {"name": f"Class {cid}"})["name"]
            ccolor = class_info.get(cid, {"color": "gray"})["color"]
            pct = (c / total_pixels) * 100
            lu_stats.append({"class_id": cid, "class_name": cname, "percentage": pct, "color": ccolor})

        df_lu = pd.DataFrame(lu_stats)
        df_lu = df_lu.sort_values("percentage", ascending=False)
        top5 = df_lu.head(5)

        fig_bar = px.bar(
            top5,
            x="percentage",
            y="class_name",
            orientation='h',
            title=f"Top 5 Land Use Types ({actual_year})",
            labels={"percentage": "Coverage (%)", "class_name": "Land Use Type"},
            text="percentage"
        )

        fig_bar.update_traces(
            marker_color=top5["color"].tolist(),
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )

        fig_bar.update_layout(
            yaxis={'categoryorder':'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#1e293b"),
            margin=dict(l=10, r=20, t=40, b=10)
        )

        top_names = top5["class_name"].tolist()
        top_pcts = top5["percentage"].tolist()
        explanation_items = [f"**{name}** ({pct:.1f}%)" for name, pct in zip(top_names, top_pcts)]
        explanation_str = ", ".join(explanation_items)
        explanation = (f"**Land Use Analysis ({actual_year}):** "
                       f"The most dominant land use type in the basin is **{top_names[0]}**, covering **{top_pcts[0]:.1f}%** of the area. "
                       f"Other significant land use types include {', '.join(explanation_items[1:])}. "
                       f"This distribution reflects the basin's ecological and anthropogenic characteristics.")

    except Exception as e:
        print(f"Error LU Bar: {e}")
        fig_bar = _empty_fig("Error calculating statistics")
        explanation = "Error generating explanation."

    return {
        "map_figure": json.loads(fig_map.to_json()),
        "bar_figure": json.loads(fig_bar.to_json()),
        "explanation": explanation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
