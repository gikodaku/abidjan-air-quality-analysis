"""Map of where the data is sampled: CAMS grid cell, weather point and Abidjan districts."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

# Approximate centres of Abidjan's main communes, for orientation only
DISTRICTS = {
    "Plateau": (5.3237, -4.0197), "Cocody": (5.3601, -3.9820), "Yopougon": (5.3363, -4.0880),
    "Abobo": (5.4181, -4.0200), "Adjamé": (5.3667, -4.0333), "Attécoubé": (5.3370, -4.0410),
    "Treichville": (5.2925, -4.0110), "Marcory": (5.3030, -3.9830),
    "Koumassi": (5.2970, -3.9480), "Port-Bouët": (5.2560, -3.9260),
    "Bingerville": (5.3550, -3.8850), "Anyama": (5.4940, -4.0520),
}


def load_grid(path: Path, latitude: float, longitude: float) -> dict:
    """Grid points saved by `aaqa fetch`; falls back to the configured point."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        point = {"latitude": latitude, "longitude": longitude}
        return {"requested": point,
                "air_quality": {**point, "model": "Copernicus CAMS global",
                                "resolution_deg": 0.4},
                "weather": {**point, "model": "ERA5 reanalysis (Open-Meteo)"}}


def sampling_map(grid: dict, line_color: str, fill_color: str, label: str,
                 dark: bool = False, height: int = 380,
                 districts: dict[str, tuple[float, float]] | None = None) -> go.Figure:
    """CAMS cell coloured by air quality, with the sampled points and the districts
    (Abidjan's by default; pass an empty dict for another city)."""
    districts = DISTRICTS if districts is None else districts
    air, weather, requested = grid["air_quality"], grid["weather"], grid["requested"]
    half = air.get("resolution_deg", 0.4) / 2
    lat0, lon0 = air["latitude"], air["longitude"]
    box_lat = [lat0 - half, lat0 - half, lat0 + half, lat0 + half, lat0 - half]
    box_lon = [lon0 - half, lon0 + half, lon0 + half, lon0 - half, lon0 - half]
    ink = "#f0efec" if dark else "#13262b"
    muted = "#8fa3a6" if dark else "#7b8d91"

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=box_lat, lon=box_lon, mode="lines", fill="toself", fillcolor=fill_color,
        line={"color": line_color, "width": 2},
        name=f"Air-quality grid cell (CAMS, ~{2 * half * 111:.0f} km) · {label}",
        hoverinfo="name"))
    if districts:
        fig.add_trace(go.Scattermap(
            lat=[lat for lat, _ in districts.values()],
            lon=[lon for _, lon in districts.values()],
            mode="markers+text", text=list(districts), textposition="top center",
            textfont={"size": 11, "color": muted}, marker={"size": 6, "color": muted},
            name="Districts", hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scattermap(
        lat=[weather["latitude"]], lon=[weather["longitude"]], mode="markers",
        marker={"size": 13, "color": "#006079"}, name="Weather model point (ERA5)",
        hovertemplate=f"Weather point<br>{weather['latitude']:.3f}, "
                      f"{weather['longitude']:.3f}<extra></extra>"))
    fig.add_trace(go.Scattermap(
        lat=[lat0], lon=[lon0], mode="markers", marker={"size": 15, "color": line_color},
        name="CAMS cell centre (air quality value)",
        hovertemplate=f"CAMS cell centre<br>{lat0:.2f}, {lon0:.2f}<extra></extra>"))
    fig.add_trace(go.Scattermap(
        lat=[requested["latitude"]], lon=[requested["longitude"]], mode="markers",
        marker={"size": 11, "color": "#3FAB3D"}, name="Requested point (city centre)",
        hovertemplate="Requested point<extra></extra>"))
    fig.update_layout(
        map={"style": "carto-darkmatter" if dark else "carto-positron",
             "center": {"lat": lat0 - 0.04, "lon": lon0}, "zoom": 8.6},
        height=height, margin={"t": 0, "b": 0, "l": 0, "r": 0},
        paper_bgcolor="rgba(0,0,0,0)", font={"family": "Montserrat, system-ui", "color": ink},
        legend={"orientation": "h", "y": -0.02, "yanchor": "top", "x": 0,
                "font": {"size": 11, "color": ink}, "bgcolor": "rgba(0,0,0,0)"},
    )
    return fig
