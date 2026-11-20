"""
Smart Crowd AI: Real-Time Crowd Monitoring & Stampede Prevention System
Main Streamlit Application Dashboard.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import time
from datetime import datetime
from streamlit_folium import st_folium

# Import internal modules
from src.database import (
    init_db, insert_detection, insert_alert, get_recent_detections,
    get_recent_alerts, acknowledge_alert, acknowledge_all_alerts,
    get_venue_zones, get_all_venue_names, update_zone_metrics,
    add_custom_zone, get_summary_stats, clear_history, seed_sample_audit_logs
)
from src.detector import CrowdDetector
from src.risk_engine import CrowdRiskEngine
from src.map_module import build_crowd_map
from src.simulation import (
    SCENARIOS, generate_crowd_simulation_tick,
    create_synthetic_crowd_image, ensure_sample_assets
)
from src.ui_components import (
    DARK_THEME_CSS, render_metric_card, render_risk_badge,
    render_origin_badge, render_audio_visual_alarm, play_alert_chime,
    create_risk_gauge_chart, create_density_timeline_chart,
    create_zone_occupancy_bar_chart
)

# Set page config
st.set_page_config(
    page_title="Smart Crowd AI | Stampede Prevention System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject dark theme CSS
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# Initialize database and sample assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
init_db()
ensure_sample_assets(ASSETS_DIR)


@st.cache_resource
def load_detector():
    """Caches the detector instance to prevent reloading weights on every interaction."""
    return CrowdDetector(model_name="yolov8n.pt", confidence=0.35)


detector = load_detector()

# Initialize session state variables
if "active_alerts" not in st.session_state:
    st.session_state.active_alerts = []
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "sim_ticks" not in st.session_state:
    st.session_state.sim_ticks = 0
if "last_processed_result" not in st.session_state:
    st.session_state.last_processed_result = None

# ==========================================
# SIDEBAR CONTROLS & SETTINGS
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 28px;">🛡️</span>
        <div>
            <h2 style="margin: 0; font-size: 19px; color: #f8fafc; font-weight: 800;">Smart Crowd AI</h2>
            <div style="font-size: 11px; color: #38bdf8; font-weight: 600;">STAMPEDE PREVENTION SUITE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active AI Engine Badge
    st.markdown(f"""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; margin-bottom: 15px;">
        <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">Active Vision Backend</div>
        <div style="font-size: 12px; color: #34d399; font-weight: 700;">{detector.active_engine}</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation menu
    app_mode = st.radio(
        "NAVIGATION",
        [
            "📊 Crowd Command Dashboard",
            "🖼️ Image AI Scanner",
            "📹 Video Stream Analyzer",
            "🗺️ Interactive Zone Map",
            "🚨 Real-Time Alerts & Safety",
            "🧪 Live Surge Demo Simulator",
            "📜 Audit & Detection History"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Risk Threshold Settings")
    warning_density_thresh = st.slider("Warning Density (p/m²)", 1.0, 3.0, 2.0, 0.1)
    critical_density_thresh = st.slider("Critical Density (p/m²)", 2.5, 4.5, 3.5, 0.1)
    stampede_density_thresh = st.slider("Stampede Crush Danger (p/m²)", 4.0, 7.0, 5.0, 0.1)

    enable_audio_alerts = st.checkbox("🔊 Offline Audio Chime Alarms", value=True)

    st.markdown("---")
    # Quick DB summary in sidebar
    stats = get_summary_stats()
    st.markdown(f"""
    <div style="font-size: 11px; color: #94a3b8; line-height: 1.6;">
        <b>Monitored Zones:</b> {stats['total_zones']}<br/>
        <b>Total Venue Occupancy:</b> {stats['total_venue_population']} / {stats['total_venue_capacity']}<br/>
        <b>Active Unacknowledged Alerts:</b> <span style="color: {'#ef4444' if stats['active_alerts'] > 0 else '#10b981'}; font-weight: bold;">{stats['active_alerts']}</span>
    </div>
    """, unsafe_allow_html=True)

# Instantiate the risk engine with user-configured thresholds
risk_engine = CrowdRiskEngine(
    warning_density=warning_density_thresh,
    critical_density=critical_density_thresh,
    stampede_density=stampede_density_thresh
)


# ==========================================
# PAGE 1: CROWD COMMAND DASHBOARD
# ==========================================
if app_mode == "📊 Crowd Command Dashboard":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px;">
        <div>
            <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #f8fafc;">Live Crowd Monitoring & Stampede Prevention</h1>
            <p style="margin: 0; font-size: 13px; color: #94a3b8;">Real-time spatial telemetry, Fruin Level of Service (LOS) analytics, and early-warning alerts</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check for active alerts and render audio-visual alert banner
    unack_alerts = get_recent_alerts(limit=1, unacknowledged_only=True)
    if not unack_alerts.empty:
        top_alert = unack_alerts.iloc[0]
        st.markdown(
            render_audio_visual_alarm(
                risk_level=top_alert["alert_level"],
                zone_name=top_alert["zone_name"],
                message=top_alert["message"],
                advisory=top_alert["advisory"]
            ),
            unsafe_allow_html=True
        )
        if enable_audio_alerts:
            play_alert_chime(top_alert["alert_level"], enable_audio=True)

    # Top KPI Metrics Cards
    venues = get_all_venue_names()
    selected_venue = st.selectbox("Select Venue Telemetry Stream", ["All Venues"] + venues, index=0)
    zones = get_venue_zones(selected_venue)

    tot_crowd = sum(z["current_count"] for z in zones)
    tot_cap = sum(z["max_capacity"] for z in zones)
    overall_occ = round((tot_crowd / tot_cap * 100) if tot_cap > 0 else 0, 1)

    # Determine highest risk zone
    highest_risk_zone = max(zones, key=lambda z: z["current_count"] / max(1, z["area_sqm"])) if zones else None
    highest_density = round(highest_risk_zone["current_count"] / highest_risk_zone["area_sqm"], 2) if highest_risk_zone else 0.0
    highest_risk_status = highest_risk_zone.get("current_risk", "SAFE") if highest_risk_zone else "SAFE"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_metric_card("Monitored Population", f"{tot_crowd:,}", f"Capacity: {tot_cap:,} ({overall_occ}%)", "blue"), unsafe_allow_html=True)
    with col2:
        status_color = "red" if highest_risk_status in ("CRITICAL", "STAMPEDE RISK") else ("yellow" if highest_risk_status == "WARNING" else "green")
        st.markdown(render_metric_card("Highest Zone Status", highest_risk_status, f"Zone: {highest_risk_zone['zone_name'] if highest_risk_zone else 'N/A'}", status_color), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("Peak Local Density", f"{highest_density} p/m²", f"Fruin LOS: {highest_risk_zone.get('current_risk', 'SAFE') if highest_risk_zone else 'A'}", "yellow"), unsafe_allow_html=True)
    with col4:
        alert_count = stats["active_alerts"]
        alert_col = "red" if alert_count > 0 else "green"
        st.markdown(render_metric_card("Active Unresolved Alerts", str(alert_count), "Zero unhandled warnings" if alert_count == 0 else "Immediate action required", alert_col), unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Main Visual Analytics Grid
    left_col, right_col = st.columns([1.6, 1.4])

    with left_col:
        st.markdown("""
        <div class="smart-card">
            <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 10px;">
                📈 Crowd Density Timeline & Threshold Violations (Recent Scans)
            </div>
        """, unsafe_allow_html=True)
        recent_dets = get_recent_detections(limit=30)
        st.plotly_chart(create_density_timeline_chart(recent_dets, warning_thresh=warning_density_thresh, crit_thresh=critical_density_thresh), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("""
        <div class="smart-card">
            <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 10px;">
                🎯 Stampede Hazard Index & Capacity Distribution
            </div>
        """, unsafe_allow_html=True)
        # Evaluate representative score
        rep_eval = risk_engine.evaluate_risk(
            person_count=highest_risk_zone["current_count"] if highest_risk_zone else 25,
            area_sqm=highest_risk_zone["area_sqm"] if highest_risk_zone else 100,
            max_capacity=highest_risk_zone["max_capacity"] if highest_risk_zone else 150
        )
        st.plotly_chart(create_risk_gauge_chart(rep_eval["risk_score"], rep_eval["risk_level"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Lower Grid: Zone Occupancy Bars & Quick Action Panel
    col_bar, col_actions = st.columns([1.8, 1.2])

    with col_bar:
        st.markdown("""
        <div class="smart-card">
            <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">
                📊 Zone Capacity Saturation Breakdown
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_zone_occupancy_bar_chart(zones), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_actions:
        st.markdown("""
        <div class="smart-card">
            <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">
                ⚡ Live Actions & Rapid Ingress Simulation
            </div>
            <p style="font-size: 12px; color: #94a3b8;">
                Simulate an immediate surge into any monitored sector to test real-time alerts:
            </p>
        """, unsafe_allow_html=True)

        zone_options = {z["zone_name"]: z for z in zones}
        if zone_options:
            target_zone_name = st.selectbox("Select Target Sector", list(zone_options.keys()))
            surge_addition = st.slider("Simulate Surge Influx (+persons)", 20, 400, 100, step=20)

            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("🚨 Inject Surge", use_container_width=True):
                    tz = zone_options[target_zone_name]
                    new_count = tz["current_count"] + surge_addition
                    eval_res = risk_engine.evaluate_risk(
                        person_count=new_count,
                        area_sqm=tz["area_sqm"],
                        max_capacity=tz["max_capacity"],
                        surge_rate_per_sec=surge_addition / 10.0
                    )
                    update_zone_metrics(tz["id"], new_count, eval_res["risk_level"])

                    # Log detection & alert if triggered
                    insert_detection(
                        source_type="Simulation",
                        person_count=new_count,
                        density=eval_res["density"],
                        risk_level=eval_res["risk_level"],
                        risk_score=eval_res["risk_score"],
                        source_name="Manual Sector Influx",
                        venue_zone=target_zone_name,
                        is_simulated=True
                    )

                    if eval_res["trigger_alert"]:
                        insert_alert(
                            zone_name=target_zone_name,
                            alert_level=eval_res["risk_level"],
                            trigger_count=new_count,
                            density=eval_res["density"],
                            message=f"Surge of +{surge_addition} people pushed density to {eval_res['density']} p/m².",
                            advisory=eval_res["advisory"]
                        )
                    st.success(f"Injected +{surge_addition} people into {target_zone_name}! Status: {eval_res['risk_level']}")
                    st.rerun()

            with c_btn2:
                if st.button("🔄 Reset Sector", use_container_width=True):
                    tz = zone_options[target_zone_name]
                    safe_count = int(tz["max_capacity"] * 0.35)
                    eval_res = risk_engine.evaluate_risk(
                        person_count=safe_count,
                        area_sqm=tz["area_sqm"],
                        max_capacity=tz["max_capacity"]
                    )
                    update_zone_metrics(tz["id"], safe_count, eval_res["risk_level"])
                    st.info(f"Sector {target_zone_name} reset to nominal safe occupancy.")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 2: IMAGE AI SCANNER
# ==========================================
elif app_mode == "🖼️ Image AI Scanner":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #f8fafc;">AI Crowd Density & Pedestrian Vision Scanner</h1>
        <p style="margin: 0; font-size: 13px; color: #94a3b8;">Upload photos or select realistic crowd presets for YOLOv8 pedestrian detection and Gaussian heatmap analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_view = st.columns([1.1, 1.9])

    with col_input:
        st.markdown("""
        <div class="smart-card">
            <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">
                📥 Input Media Selection
            </div>
        """, unsafe_allow_html=True)

        input_choice = st.radio("Choose Source", ["Preloaded Sample Crowd Presets", "Upload Custom Image"], horizontal=True)
        img_bgr = None
        source_name = "Sample Preset"
        is_sample = False

        if input_choice == "Preloaded Sample Crowd Presets":
            preset_options = {
                "🟢 Safe Normal Flow (Concourse)": "sample_safe_crowd.jpg",
                "⚡ Warning Queue Buildup (Subway)": "sample_warning_crowd.jpg",
                "⚠️ Critical Exit Chokepoint (Bottleneck)": "sample_critical_surge.jpg",
                "🚨 Stampede Hazard (Capacity Crush)": "sample_stampede_risk.jpg"
            }
            chosen_preset = st.selectbox("Select Scenario Preset", list(preset_options.keys()))
            sample_path = os.path.join(ASSETS_DIR, preset_options[chosen_preset])
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)
                source_name = chosen_preset
                is_sample = True
        else:
            uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                source_name = uploaded_file.name
                is_sample = False

        st.markdown("---")
        st.markdown("#### 📐 Sector Geometry & Model Parameters")
        all_zones = get_venue_zones()
        zone_names = [z["zone_name"] for z in all_zones]
        selected_zone_name = st.selectbox("Assign to Monitored Zone", zone_names, index=0)

        # Retrieve matched zone properties
        matched_zone = next((z for z in all_zones if z["zone_name"] == selected_zone_name), None)
        default_area = matched_zone["area_sqm"] if matched_zone else 150.0
        default_cap = matched_zone["max_capacity"] if matched_zone else 300

        area_sqm = st.number_input("Zone Monitored Area (m²)", min_value=10.0, max_value=10000.0, value=float(default_area), step=10.0)
        max_capacity = st.number_input("Maximum Rated Capacity", min_value=10, max_value=50000, value=int(default_cap), step=25)

        engine_mode = st.selectbox(
            "Vision Engine Mode",
            ["Auto (YOLOv8 + Intelligent Fallback)", "OpenCV HOG Pedestrian Classifier", "Synthetic / Blob Detection"],
            index=0
        )

        conf_thresh = st.slider("Detection Confidence Threshold", 0.15, 0.85, 0.35, 0.05)

        st.markdown("#### 🎨 Visualization Overlays")
        show_boxes = st.checkbox("Bounding Boxes with Confidence", value=True)
        show_centroids = st.checkbox("Pedestrian Centroid Coordinates", value=True)
        show_heatmap = st.checkbox("Gaussian Crowd Density Heatmap", value=True)
        heatmap_opacity = st.slider("Heatmap Blend Opacity", 0.1, 0.9, 0.45, 0.05)

        scan_button = st.button("🚀 Run AI Crowd Analysis", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_view:
        if img_bgr is not None:
            if scan_button or st.session_state.last_processed_result is not None:
                if scan_button:
                    # Run Detection
                    override = None
                    if "HOG" in engine_mode:
                        override = "OpenCV HOG Pedestrian Classifier"
                    elif "Synthetic" in engine_mode:
                        override = "Synthetic / Blob Detection"

                    det_res = detector.detect(img_bgr, engine_override=override, conf_threshold=conf_thresh)

                    # Evaluate Risk
                    risk_info = risk_engine.evaluate_risk(
                        person_count=det_res["person_count"],
                        area_sqm=area_sqm,
                        max_capacity=max_capacity,
                        cluster_factor=det_res["cluster_factor"]
                    )

                    # Annotate image
                    annotated_bgr = detector.annotate(
                        img_bgr,
                        boxes=det_res["boxes"],
                        centroids=det_res["centroids"],
                        risk_info=risk_info,
                        show_boxes=show_boxes,
                        show_centroids=show_centroids,
                        show_heatmap=show_heatmap,
                        heatmap_opacity=heatmap_opacity
                    )

                    # Save to database
                    det_id = insert_detection(
                        source_type="Image Upload",
                        person_count=det_res["person_count"],
                        density=risk_info["density"],
                        risk_level=risk_info["risk_level"],
                        risk_score=risk_info["risk_score"],
                        source_name=source_name,
                        venue_zone=selected_zone_name,
                        is_simulated=det_res["is_simulated"],
                        processing_time_ms=det_res["inference_time_ms"]
                    )

                    # Log alert if safety thresholds crossed
                    if risk_info["trigger_alert"]:
                        insert_alert(
                            zone_name=selected_zone_name,
                            alert_level=risk_info["risk_level"],
                            trigger_count=det_res["person_count"],
                            density=risk_info["density"],
                            message=f"High density scan ({risk_info['density']} p/m²) detected in {selected_zone_name}.",
                            advisory=risk_info["advisory"]
                        )

                    # Update zone status
                    if matched_zone:
                        update_zone_metrics(matched_zone["id"], det_res["person_count"], risk_info["risk_level"])

                    # Store in session
                    st.session_state.last_processed_result = {
                        "annotated_bgr": annotated_bgr,
                        "orig_bgr": img_bgr,
                        "det_res": det_res,
                        "risk_info": risk_info,
                        "source_name": source_name,
                        "det_id": det_id
                    }

                # Display Results
                res_data = st.session_state.last_processed_result
                risk = res_data["risk_info"]
                det = res_data["det_res"]

                # Result Metrics Bar
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(render_metric_card("Detected People", str(det["person_count"]), f"Inference: {det['inference_time_ms']} ms", "blue"), unsafe_allow_html=True)
                with m2:
                    st.markdown(render_metric_card("Surface Density", f"{risk['density']} p/m²", f"Occupancy: {risk['occupancy_pct']}%", "yellow"), unsafe_allow_html=True)
                with m3:
                    st.markdown(render_metric_card("Fruin Standard", risk["fruin_los"].split()[0], risk["fruin_los"], "green"), unsafe_allow_html=True)
                with m4:
                    rc = "red" if risk["risk_level"] in ("CRITICAL", "STAMPEDE RISK") else ("yellow" if risk["risk_level"] == "WARNING" else "green")
                    st.markdown(render_metric_card("Safety Status", risk["risk_level"], f"Index: {risk['risk_score']}/100", rc), unsafe_allow_html=True)

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                # Badges row
                st.markdown(f"""
                <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 12px;">
                    {render_risk_badge(risk["risk_level"])}
                    {render_origin_badge(det["is_simulated"], det["engine_used"])}
                    <span style="font-size: 11px; color: #94a3b8; margin-left: auto;">Audit Record ID: #{res_data.get('det_id', 'N/A')}</span>
                </div>
                """, unsafe_allow_html=True)

                # Actionable Directive Box
                st.markdown(f"""
                <div style="background: #111827; border-left: 4px solid {risk['color']}; border-radius: 6px; padding: 12px 16px; margin-bottom: 15px;">
                    <div style="font-size: 12px; font-weight: 700; color: {risk['color']}; text-transform: uppercase;">Prescriptive Safety Directive:</div>
                    <div style="font-size: 13px; color: #f1f5f9; margin-top: 4px;">{risk['advisory']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Tabs for Annotated vs Original
                tab_ann, tab_orig = st.tabs(["🎯 Annotated AI Vision View", "📷 Original Clean Feed"])
                with tab_ann:
                    annotated_rgb = cv2.cvtColor(res_data["annotated_bgr"], cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)
                with tab_orig:
                    orig_rgb = cv2.cvtColor(res_data["orig_bgr"], cv2.COLOR_BGR2RGB)
                    st.image(orig_rgb, use_container_width=True)

            else:
                # Preview original image
                orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, caption=f"Ready for analysis: {source_name}", use_container_width=True)
                st.info("Click '🚀 Run AI Crowd Analysis' in the left control panel to process this scene.")
        else:
            st.warning("Please upload an image or choose one of the preloaded crowd presets on the left.")


# ==========================================
# PAGE 3: VIDEO STREAM ANALYZER
# ==========================================
elif app_mode == "📹 Video Stream Analyzer":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #f8fafc;">Real-Time Video Stream Crowd Tracker</h1>
        <p style="margin: 0; font-size: 13px; color: #94a3b8;">Frame-by-frame temporal tracking, surge velocity estimation, and automated video annotations</p>
    </div>
    """, unsafe_allow_html=True)

    col_v_ctrl, col_v_out = st.columns([1.1, 1.9])

    with col_v_ctrl:
        st.markdown("""
        <div class="smart-card">
            <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">
                📹 Video Source Selection
            </div>
        """, unsafe_allow_html=True)

        vid_choice = st.radio("Source", ["Preloaded Ingress Surge Video", "Upload Custom Video"], horizontal=True)
        video_path = None

        if vid_choice == "Preloaded Ingress Surge Video":
            preset_vpath = os.path.join(ASSETS_DIR, "sample_crowd_surge.mp4")
            if os.path.exists(preset_vpath):
                video_path = preset_vpath
                st.success("Loaded Preloaded Surge Video (45-frame sequence demonstrating bottleneck rush).")
        else:
            uploaded_video = st.file_uploader("Upload Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                video_path = tfile.name

        st.markdown("---")
        st.markdown("#### ⚡ Processing Optimization")
        frame_stride = st.slider("Frame Stride (Analyze every Nth frame)", 1, 5, 2, help="Higher stride improves processing speed for long videos.")
        max_frames_to_process = st.slider("Maximum Frames to Process", 10, 150, 45, step=5)
        vid_zone = st.selectbox("Monitored Video Sector", [z["zone_name"] for z in get_venue_zones()], index=1)
        vid_area = st.number_input("Sector Area (m²)", min_value=10.0, value=80.0, step=10.0)

        process_vid_btn = st.button("▶️ Process Video Stream", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_v_out:
        if video_path and os.path.exists(video_path):
            if process_vid_btn:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

                st.markdown(f"**Analyzing Video:** {total_frames} total frames at {fps:.1f} FPS")

                prog_bar = st.progress(0.0)
                status_placeholder = st.empty()
                frame_placeholder = st.empty()

                frame_counts = []
                frame_densities = []
                frame_timestamps = []
                frame_risks = []

                processed_count = 0
                frame_idx = 0
                peak_count = 0
                peak_risk = "SAFE"

                while cap.isOpened() and processed_count < max_frames_to_process:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_stride == 0:
                        # Detect
                        det_res = detector.detect(frame, conf_threshold=0.35)
                        p_cnt = det_res["person_count"]
                        peak_count = max(peak_count, p_cnt)

                        # Evaluate
                        risk_info = risk_engine.evaluate_risk(
                            person_count=p_cnt,
                            area_sqm=vid_area,
                            surge_rate_per_sec=max(0.0, (p_cnt - (frame_counts[-1] if frame_counts else p_cnt)) * (fps / frame_stride))
                        )

                        if risk_info["risk_score"] > (85.0 if peak_risk == "CRITICAL" else 65.0):
                            peak_risk = risk_info["risk_level"]

                        # Annotate
                        ann_frame = detector.annotate(
                            frame,
                            boxes=det_res["boxes"],
                            centroids=det_res["centroids"],
                            risk_info=risk_info,
                            show_boxes=True,
                            show_centroids=True,
                            show_heatmap=True
                        )

                        # Track data
                        time_sec = round(frame_idx / fps, 1)
                        frame_counts.append(p_cnt)
                        frame_densities.append(risk_info["density"])
                        frame_timestamps.append(f"{time_sec}s")
                        frame_risks.append(risk_info["risk_level"])

                        # Live preview frame
                        frame_rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, caption=f"Frame #{frame_idx} | Crowd Count: {p_cnt} | Density: {risk_info['density']} p/m²", use_container_width=True)

                        processed_count += 1
                        prog_bar.progress(min(1.0, processed_count / max_frames_to_process))
                        status_placeholder.text(f"Processed {processed_count}/{max_frames_to_process} frames...")

                    frame_idx += 1

                cap.release()
                prog_bar.progress(1.0)
                status_placeholder.success("Video analysis completed successfully!")

                # Log video summary detection to SQLite
                avg_density = round(sum(frame_densities) / len(frame_densities), 2) if frame_densities else 0.0
                insert_detection(
                    source_type="Video Stream",
                    person_count=peak_count,
                    density=avg_density,
                    risk_level=peak_risk,
                    risk_score=max(30.0, avg_density * 18.0),
                    source_name="Video Analysis Stream",
                    venue_zone=vid_zone,
                    is_simulated=False,
                    details={"frames_analyzed": processed_count, "peak_count": peak_count}
                )

                # Render Video Trend Chart
                st.markdown("#### 📊 Temporal Crowd Surge Curve")
                v_df = pd.DataFrame({
                    "timestamp": frame_timestamps,
                    "person_count": frame_counts,
                    "density": frame_densities,
                    "risk": frame_risks
                })
                fig = create_density_timeline_chart(v_df, warning_thresh=warning_density_thresh, crit_thresh=critical_density_thresh)
                st.plotly_chart(fig, use_container_width=True)

                if peak_risk in ("CRITICAL", "STAMPEDE RISK"):
                    insert_alert(
                        zone_name=vid_zone,
                        alert_level=peak_risk,
                        trigger_count=peak_count,
                        density=max(frame_densities),
                        message=f"Video analysis recorded surge spike reaching {peak_count} people ({max(frame_densities)} p/m²).",
                        advisory="Initiate video-verified crowd dispersion and slow entry turnstiles."
                    )
                    st.error(f"🚨 Hazard Detected: Video analysis recorded {peak_risk} with peak density of {max(frame_densities)} persons/m².")
            else:
                st.info("Click '▶️ Process Video Stream' in the left panel to begin temporal analysis.")
        else:
            st.warning("Please upload a video or select the preloaded surge sequence.")


# ==========================================
# PAGE 4: INTERACTIVE ZONE MAP
# ==========================================
elif app_mode == "🗺️ Interactive Zone Map":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #f8fafc;">Interactive Multi-Sector Crowd Map</h1>
        <p style="margin: 0; font-size: 13px; color: #94a3b8;">Live spatial geolocation of safe, warning, and critical density zones with heatmap overlays</p>
    </div>
    """, unsafe_allow_html=True)

    venues = get_all_venue_names()
    c_map_filter, c_map_heat = st.columns([2, 1])
    with c_map_filter:
        map_venue = st.selectbox("Venue Filter", ["All Venues"] + venues, index=1)
    with c_map_heat:
        enable_map_heatmap = st.checkbox("🔥 Enable Global Density Heatmap Overlay", value=True)

    zones = get_venue_zones(map_venue)

    # Render Folium Map
    crowd_map = build_crowd_map(zones, enable_heatmap=enable_map_heatmap)
    st_folium(crowd_map, width="100%", height=520)

    st.markdown("---")
    col_t_left, col_t_right = st.columns([1.8, 1.2])

    with col_t_left:
        st.markdown("#### 📍 Monitored Sectors Telemetry Table")
        table_rows = []
        for z in zones:
            density = round(z["current_count"] / z["area_sqm"], 2)
            table_rows.append({
                "Zone Sector": z["zone_name"],
                "Venue": z["venue_name"],
                "Count": z["current_count"],
                "Capacity": z["max_capacity"],
                "Occupancy": f"{round((z['current_count']/z['max_capacity'])*100, 1)}%",
                "Density (p/m²)": density,
                "Status": z["current_risk"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    with col_t_right:
        st.markdown("#### ➕ Add New Custom Zone")
        with st.expander("Register New Monitored Sector", expanded=False):
            with st.form("add_zone_form"):
                new_vname = st.text_input("Venue Name", value=map_venue if map_venue != "All Venues" else "Central Mega Arena")
                new_zname = st.text_input("Zone / Sector Name", placeholder="Gate 5 West Promenade")
                new_area = st.number_input("Surface Area (m²)", min_value=20.0, value=250.0)
                new_cap = st.number_input("Max Capacity", min_value=10, value=500)
                c_lat, c_lon = st.columns(2)
                with c_lat:
                    new_lat = st.number_input("Latitude", value=40.7128, format="%.4f")
                with c_lon:
                    new_lon = st.number_input("Longitude", value=-74.0060, format="%.4f")

                submit_zone = st.form_submit_button("Register Sector")
                if submit_zone:
                    if new_zname.strip():
                        add_custom_zone(
                            venue_name=new_vname,
                            zone_name=new_zname,
                            area_sqm=new_area,
                            max_capacity=new_cap,
                            warning_threshold=warning_density_thresh,
                            critical_threshold=critical_density_thresh,
                            lat=new_lat,
                            lon=new_lon
                        )
                        st.success(f"Sector '{new_zname}' registered successfully!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid sector name.")


# ==========================================
# PAGE 5: REAL-TIME ALERTS & SAFETY
# ==========================================
elif app_mode == "🚨 Real-Time Alerts & Safety":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #f8fafc;">Automatic Risk Alerts & Emergency Incident Log</h1>
        <p style="margin: 0; font-size: 13px; color: #94a3b8;">Incident acknowledgment workflow, emergency evacuation protocols, and prescriptive response directives</p>
    </div>
    """, unsafe_allow_html=True)

    col_btn_ack, col_btn_clear = st.columns([1.5, 3.5])
    with col_btn_ack:
        if st.button("✅ Acknowledge All Active Alerts", use_container_width=True):
            acknowledge_all_alerts()
            st.success("All alerts marked as acknowledged.")
            st.rerun()

    alerts_df = get_recent_alerts(limit=50)

    if alerts_df.empty:
        st.markdown("""
        <div style="background: #111827; border: 1px dashed #334155; border-radius: 10px; padding: 30px; text-align: center; color: #94a3b8;">
            <div style="font-size: 32px; margin-bottom: 8px;">🛡️</div>
            <div style="font-size: 16px; font-weight: 700; color: #f8fafc;">No Active Incident Alerts</div>
            <div style="font-size: 13px; margin-top: 4px;">All sectors are functioning within safe density thresholds.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display incident table with actions
        for idx, row in alerts_df.iterrows():
            is_unack = row["acknowledged"] == 0
            border_col = "#ef4444" if row["alert_level"] in ("CRITICAL", "STAMPEDE RISK") else "#f59e0b"
            bg_card = "#1a1215" if is_unack and row["alert_level"] in ("CRITICAL", "STAMPEDE RISK") else "#131b2e"

            c_card, c_act = st.columns([4, 1])
            with c_card:
                st.markdown(f"""
                <div style="background: {bg_card}; border: 1px solid {border_col}; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 14px; font-weight: 800; color: {border_col};">
                            {'🚨' if row['alert_level'] in ('CRITICAL', 'STAMPEDE RISK') else '⚠️'} {row['alert_level']} — Sector: {row['zone_name']}
                        </span>
                        <span style="font-size: 11px; color: #94a3b8;">{row['timestamp']}</span>
                    </div>
                    <div style="font-size: 13px; color: #f1f5f9; margin-bottom: 6px;">
                        <b>Telemetry Trigger:</b> {row['trigger_count']} persons detected | Density: <b>{row['density']} persons/m²</b>
                    </div>
                    <div style="font-size: 12px; color: #e2e8f0; background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: 4px; border-left: 3px solid {border_col};">
                        <b>Command Action:</b> {row['advisory']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_act:
                if is_unack:
                    if st.button(f"Acknowledge #{row['id']}", key=f"ack_{row['id']}", use_container_width=True):
                        acknowledge_alert(row["id"])
                        st.rerun()
                else:
                    st.markdown("<span style='font-size: 12px; color: #10b981; font-weight: 700;'>Resolved ✔</span>", unsafe_allow_html=True)

    # Standard Operating Procedures Card
    st.markdown("---")
    st.markdown("#### 📋 Crowd Safety Standard Operating Procedures (SOP)")
    st.markdown("""
    <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; font-size: 12px; color: #cbd5e1; line-height: 1.7;">
        <b>Level 1 - Safe Operation (&lt; 2.0 p/m²):</b> Maintain standard queue barriers, unimpeded ingress/egress, nominal surveillance.<br/>
        <b>Level 2 - Metering Warning (2.0 - 3.5 p/m²):</b> Deploy marshals to perimeter, throttle turnstile entry to 50%, activate queue zig-zag buffers.<br/>
        <b>Level 3 - Critical Chokepoint (3.5 - 5.0 p/m²):</b> Halt incoming flows at outer perimeter, open secondary relief gates, sound pre-recorded dispersal directives.<br/>
        <b>Level 4 - Stampede Hazard (&gt; 5.0 p/m²):</b> Drop all physical turnstiles, enact emergency egress routes, scramble rapid response emergency medical units.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 6: LIVE SURGE DEMO SIMULATOR
# ==========================================
elif app_mode == "🧪 Live Surge Demo Simulator":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #f8fafc;">Live Crowd Surge Simulator (Zero-Hardware Demo Mode)</h1>
        <p style="margin: 0; font-size: 13px; color: #94a3b8;">Demonstrate real-time stampede prevention, automatic alert triggering, and visual overlays without requiring physical cameras</p>
    </div>
    """, unsafe_allow_html=True)

    col_sim_cfg, col_sim_view = st.columns([1.1, 1.9])

    with col_sim_cfg:
        st.markdown("""
        <div class="smart-card">
            <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">
                🧪 Simulation Configuration
            </div>
        """, unsafe_allow_html=True)

        selected_scenario = st.selectbox("Select Scenario Preset", list(SCENARIOS.keys()), index=2)
        scen_info = SCENARIOS[selected_scenario]

        st.markdown(f"""
        <div style="font-size: 12px; color: #cbd5e1; background: #1a2234; padding: 10px; border-radius: 6px; margin-bottom: 12px;">
            <b>Scenario Profile:</b> {scen_info['description']}<br/>
            <b>Expected Baseline:</b> ~{scen_info['base_count']} people | <b>Target Status:</b> {scen_info['expected_risk']}
        </div>
        """, unsafe_allow_html=True)

        sim_area = st.slider("Simulated Sector Area (m²)", 30.0, 200.0, 60.0, 5.0)
        num_ticks = st.slider("Surge Evolution Steps", 3, 20, 8, step=1)
        step_delay = st.slider("Step Interval (seconds)", 0.2, 2.0, 0.6, 0.1)

        sim_run_btn = st.button("🚀 Launch Real-Time Surge Sequence", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sim_view:
        sim_container = st.empty()
        status_box = st.empty()
        chart_box = st.empty()

        if sim_run_btn:
            tick_counts = []
            tick_densities = []
            tick_labels = []

            for step in range(num_ticks):
                # Generate dynamic simulation state
                sim_tick = generate_crowd_simulation_tick(selected_scenario, elapsed_ticks=step, area_sqm=sim_area)
                count = sim_tick["person_count"]

                # Generate synthetic visual frame
                synthetic_frame = create_synthetic_crowd_image(num_people=count, width=800, height=450)

                # Run detector on frame
                det_res = detector.detect(synthetic_frame, conf_threshold=0.35)

                # Evaluate risk
                risk_info = risk_engine.evaluate_risk(
                    person_count=det_res["person_count"],
                    area_sqm=sim_area,
                    surge_rate_per_sec=sim_tick["surge_rate"],
                    cluster_factor=sim_tick["cluster_factor"]
                )

                # Annotate
                annotated = detector.annotate(
                    synthetic_frame,
                    boxes=det_res["boxes"],
                    centroids=det_res["centroids"],
                    risk_info=risk_info,
                    show_boxes=True,
                    show_centroids=True,
                    show_heatmap=True
                )

                # Record in database
                insert_detection(
                    source_type="Simulation",
                    person_count=det_res["person_count"],
                    density=risk_info["density"],
                    risk_level=risk_info["risk_level"],
                    risk_score=risk_info["risk_score"],
                    source_name=f"Simulator: {selected_scenario}",
                    venue_zone="Demo Sector A",
                    is_simulated=True
                )

                if risk_info["trigger_alert"]:
                    insert_alert(
                        zone_name="Demo Sector A",
                        alert_level=risk_info["risk_level"],
                        trigger_count=det_res["person_count"],
                        density=risk_info["density"],
                        message=f"Surge reached {det_res['person_count']} people in {selected_scenario}.",
                        advisory=risk_info["advisory"]
                    )

                # Track for chart
                tick_counts.append(det_res["person_count"])
                tick_densities.append(risk_info["density"])
                tick_labels.append(f"T+{step}")

                # Render live frame
                ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                sim_container.image(ann_rgb, caption=f"Live Simulation Step {step + 1}/{num_ticks} | {render_risk_badge(risk_info['risk_level'])}", use_container_width=True)

                status_box.markdown(f"""
                <div style="display: flex; gap: 15px; margin-top: 8px;">
                    {render_origin_badge(True)}
                    {render_risk_badge(risk_info['risk_level'])}
                    <span style="font-size: 13px; font-weight: bold; color: #f8fafc;">Density: {risk_info['density']} p/m² | Score: {risk_info['risk_score']}/100</span>
                </div>
                """, unsafe_allow_html=True)

                # Update live mini-chart
                t_df = pd.DataFrame({
                    "timestamp": tick_labels,
                    "person_count": tick_counts,
                    "density": tick_densities
                })
                chart_box.plotly_chart(
                    create_density_timeline_chart(t_df, warning_thresh=warning_density_thresh, crit_thresh=critical_density_thresh),
                    use_container_width=True
                )

                time.sleep(step_delay)

            st.success("Simulation sequence completed! All telemetry saved to audit history.")
        else:
            # Show initial placeholder
            sample_preview = create_synthetic_crowd_image(num_people=45, width=800, height=450)
            st.image(cv2.cvtColor(sample_preview, cv2.COLOR_BGR2RGB), caption="Simulation Engine Ready. Click Launch to simulate crowd evolution.", use_container_width=True)


# ==========================================
# PAGE 7: AUDIT & DETECTION HISTORY
# ==========================================
elif app_mode == "📜 Audit & Detection History":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #f8fafc;">Persistent SQLite Audit & Detection Logs</h1>
        <p style="margin: 0; font-size: 13px; color: #94a3b8;">Filter, inspect, and export all recorded crowd detections, risk scores, and alert triggers</p>
    </div>
    """, unsafe_allow_html=True)

    # Filter row
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        filter_source = st.selectbox("Filter by Source Type", ["All", "Image Upload", "Video Stream", "Simulation"], index=0)
    with f_col2:
        filter_risk = st.selectbox("Filter by Safety Risk Level", ["All", "SAFE", "MODERATE", "WARNING", "CRITICAL", "STAMPEDE RISK"], index=0)
    with f_col3:
        record_limit = st.selectbox("Max Records to Display", [25, 50, 100, 250], index=1)

    history_df = get_recent_detections(limit=record_limit, filter_source=filter_source, filter_risk=filter_risk)

    # Action controls row: Export, Restore, and Guarded Reset
    c_exp, c_restore, c_reset = st.columns([1.5, 1.2, 1.3])
    with c_exp:
        if not history_df.empty:
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Export Telemetry Log (CSV)",
                data=csv_data,
                file_name=f"smart_crowd_ai_audit_{timestamp_str}.csv",
                mime="text/csv",
                use_container_width=True
            )
    with c_restore:
        if st.button("🔄 Restore Demo Telemetry", use_container_width=True):
            seed_sample_audit_logs()
            st.success("Sample telemetry data restored successfully!")
            st.rerun()

    with c_reset:
        with st.popover("🗑️ Clear Logs"):
            st.markdown("<b style='color:#ef4444;'>Danger Zone</b>", unsafe_allow_html=True)
            st.caption("This will permanently remove all detection and alert history.")
            confirm_clear = st.checkbox("Confirm deletion")
            if st.button("Permanently Clear All", type="primary", disabled=not confirm_clear, use_container_width=True):
                clear_history()
                st.success("History database successfully reset.")
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if history_df.empty:
        st.info("No records matched your filter criteria.")
    else:
        # Format columns for crisp display
        display_df = history_df.copy()
        display_df["Origin"] = display_df["is_simulated"].apply(lambda s: "🧪 Simulation" if s == 1 else "⚡ Real AI")
        display_df = display_df[[
            "id", "timestamp", "source_type", "venue_zone", "person_count",
            "density", "risk_level", "risk_score", "Origin", "processing_time_ms"
        ]]
        display_df.columns = [
            "ID", "Timestamp", "Source", "Sector Zone", "Count",
            "Density (p/m²)", "Risk Level", "Score", "Telemetry Mode", "Inference (ms)"
        ]

        st.dataframe(display_df, use_container_width=True)
