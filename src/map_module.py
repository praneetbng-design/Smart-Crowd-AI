"""
Smart Crowd AI: Interactive Crowd Zone Map Module
Builds dynamic Folium map visualizations with color-coded safety risk zones,
capacity thresholds, heatmaps, and situational advisories.
"""

import folium
from folium import plugins
from typing import List, Dict, Any, Optional


COLOR_MAP = {
    "SAFE": "#10B981",
    "MODERATE": "#3B82F6",
    "WARNING": "#F59E0B",
    "CRITICAL": "#F97316",
    "STAMPEDE RISK": "#EF4444"
}


def build_crowd_map(
    zones: List[Dict[str, Any]],
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom_start: int = 16,
    enable_heatmap: bool = True
) -> folium.Map:
    """
    Constructs a dark-styled Folium map with zone markers, circles, and risk overlays.
    """
    if not zones:
        # Default center if no zones
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=15, tiles="CartoDB dark_matter")
        return m

    # Determine center
    if center_lat is None or center_lon is None:
        center_lat = sum(z["lat"] for z in zones) / len(zones)
        center_lon = sum(z["lon"] for z in zones) / len(zones)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB dark_matter",
        control_scale=True
    )

    # Add Fullscreen button
    plugins.Fullscreen(
        position="topright",
        title="Expand Map",
        title_cancel="Exit Fullscreen",
        force_separate_button=True
    ).add_to(m)

    heat_data = []

    # Draw each zone
    for z in zones:
        lat = z["lat"]
        lon = z["lon"]
        name = z.get("zone_name", "Monitored Zone")
        count = z.get("current_count", 0)
        max_cap = z.get("max_capacity", 100)
        area = z.get("area_sqm", 200.0)
        density = round(count / (area if area > 0 else 100.0), 2)
        risk = z.get("current_risk", "SAFE")
        color = COLOR_MAP.get(risk, "#10B981")
        occ_pct = round((count / max_cap * 100.0) if max_cap > 0 else 0.0, 1)

        # Append to heat data (weight proportional to density)
        heat_weight = min(1.0, max(0.1, density / 4.0))
        heat_data.append([lat, lon, heat_weight])

        # Radius scaled by area
        circle_radius = max(35, min(90, int((area ** 0.5) * 3)))

        # HTML Popup with dark card styling
        popup_html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, sans-serif; min-width: 210px; padding: 6px; color: #1e293b;">
            <div style="font-size: 14px; font-weight: bold; border-bottom: 2px solid {color}; padding-bottom: 4px; margin-bottom: 6px;">
                {name}
            </div>
            <div style="font-size: 12px; line-height: 1.5;">
                <b>Venue:</b> {z.get('venue_name', 'Venue')}<br/>
                <b>Current Crowd:</b> <span style="font-size: 13px; font-weight: bold; color: {color};">{count}</span> / {max_cap} ({occ_pct}%)<br/>
                <b>Area:</b> {area} m²<br/>
                <b>Density:</b> <b>{density}</b> persons/m²<br/>
                <b>Safety Status:</b> <span style="display: inline-block; padding: 2px 6px; border-radius: 4px; background: {color}; color: white; font-weight: bold; font-size: 11px;">{risk}</span>
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #64748b; background: #f8fafc; padding: 4px 6px; border-radius: 4px;">
                Warning limit: {z.get('warning_threshold', 1.8)} p/m²<br/>
                Critical limit: {z.get('critical_threshold', 2.8)} p/m²
            </div>
        </div>
        """

        # Add zone circle
        folium.Circle(
            location=[lat, lon],
            radius=circle_radius,
            color=color,
            weight=3,
            fill=True,
            fill_color=color,
            fill_opacity=0.35 if risk in ("SAFE", "MODERATE") else 0.55,
            tooltip=f"{name}: {count} people ({risk})",
            popup=folium.Popup(popup_html, max_width=320)
        ).add_to(m)

        # Add icon marker
        icon_name = "exclamation-triangle" if risk in ("CRITICAL", "STAMPEDE RISK") else ("users" if risk == "WARNING" else "check-circle")
        marker_color = "red" if risk in ("CRITICAL", "STAMPEDE RISK") else ("orange" if risk == "WARNING" else "green")

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color=marker_color, icon=icon_name, prefix="fa"),
            tooltip=f"{name} [{risk}]"
        ).add_to(m)

    # Add heatmap layer if enabled and points exist
    if enable_heatmap and len(heat_data) > 0:
        plugins.HeatMap(
            heat_data,
            name="Crowd Density Heatmap",
            min_opacity=0.3,
            max_zoom=18,
            radius=45,
            blur=25,
            gradient={0.2: "#10B981", 0.5: "#F59E0B", 0.8: "#F97316", 1.0: "#EF4444"}
        ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    return m
