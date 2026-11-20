"""
Smart Crowd AI: Simulation and Synthetic Crowd Generator
Provides demo simulation scenarios, synthetic crowd image generators,
and realistic surge patterns to demonstrate stampede alerts and safety protocols.
"""

import cv2
import numpy as np
import os
import random
import time
from typing import Dict, Any, List, Tuple


SCENARIOS = {
    "Normal Free Flow": {
        "description": "Steady pedestrian movement within safe operating thresholds. Ingress and egress balanced.",
        "base_count": 28,
        "variance": 6,
        "surge_rate": 0.2,
        "cluster_factor": 0.25,
        "expected_risk": "SAFE"
    },
    "Metro Evening Rush Hour": {
        "description": "High influx from train platforms creating moderate queueing near escalators and fare gates.",
        "base_count": 78,
        "variance": 12,
        "surge_rate": 2.4,
        "cluster_factor": 0.55,
        "expected_risk": "WARNING"
    },
    "Concert Gate Ingress Bottleneck": {
        "description": "Rapid crowd surge attempting to enter via Gate 2. Tight bottleneck forming at security check.",
        "base_count": 145,
        "variance": 20,
        "surge_rate": 5.8,
        "cluster_factor": 0.78,
        "expected_risk": "CRITICAL"
    },
    "Severe Stampede & Chokepoint Crush": {
        "description": "CRITICAL EMERGENCY: Exits partially obstructed, crowd density exceeding 5.0 persons/m². High compressive force.",
        "base_count": 260,
        "variance": 25,
        "surge_rate": 11.5,
        "cluster_factor": 0.92,
        "expected_risk": "STAMPEDE RISK"
    }
}


def generate_crowd_simulation_tick(
    scenario_name: str,
    elapsed_ticks: int = 0,
    area_sqm: float = 80.0
) -> Dict[str, Any]:
    """Generates a dynamic data point for a time-step in a live simulation."""
    config = SCENARIOS.get(scenario_name, SCENARIOS["Normal Free Flow"])
    base = config["base_count"]
    var = config["variance"]
    surge_rate = config["surge_rate"]
    cluster = config["cluster_factor"]

    # Add dynamic wave effect
    wave = int(np.sin(elapsed_ticks * 0.4) * (var * 0.7))
    noise = random.randint(-var // 2, var // 2)

    # In surge scenarios, count rises with ticks
    growth = 0
    if "Bottleneck" in scenario_name or "Stampede" in scenario_name:
        growth = min(80, elapsed_ticks * 4)

    sim_count = max(5, base + wave + noise + growth)

    return {
        "scenario": scenario_name,
        "person_count": sim_count,
        "area_sqm": area_sqm,
        "surge_rate": round(surge_rate + (random.random() * 0.8), 2),
        "cluster_factor": round(min(0.98, cluster + (random.random() * 0.05)), 2),
        "is_simulated": True
    }


def create_synthetic_crowd_image(
    num_people: int = 40,
    width: int = 960,
    height: int = 540,
    title_banner: str = "SYNTHETIC CROWD SIMULATION"
) -> np.ndarray:
    """
    Creates a realistic synthetic overhead / concourse crowd scene with floor tiles,
    barriers, and pedestrian silhouettes.
    """
    # Create textured background (concourse floor)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (38, 42, 54) # Dark floor

    # Draw grid floor tiles
    tile_size = 40
    for x in range(0, width, tile_size):
        cv2.line(canvas, (x, 0), (x, height), (48, 52, 66), 1)
    for y in range(0, height, tile_size):
        cv2.line(canvas, (0, y), (width, y), (48, 52, 66), 1)

    # Draw entrance gate barriers
    cv2.rectangle(canvas, (40, 20), (width - 40, 45), (70, 75, 90), -1)
    cv2.putText(canvas, "GATE 4 - MAIN INGRESS & SCREENING CHANNEL", (60, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 175, 200), 1)

    # Draw side guide rails
    cv2.rectangle(canvas, (20, 60), (35, height - 20), (90, 100, 120), -1)
    cv2.rectangle(canvas, (width - 35, 60), (width - 20, height - 20), (90, 100, 120), -1)

    # Seed random coordinates with realistic clustering
    random.seed(42 + num_people)
    np.random.seed(42 + num_people)

    # Cluster more heavily towards the middle if density is high
    spread_factor = max(0.15, 0.45 - (min(num_people, 200) / 500.0))

    clothing_colors = [
        (180, 120, 50),   # Blue
        (60, 140, 60),    # Green
        (160, 60, 60),    # Maroon
        (180, 180, 180),  # Light grey
        (40, 120, 180),   # Orange/amber
        (120, 50, 150),   # Purple
        (90, 90, 90),     # Dark grey
        (220, 190, 140),  # Khaki
        (50, 50, 180)     # Red
    ]

    for i in range(num_people):
        # Position with gaussian distribution around center flow
        cx = int(np.random.normal(width * 0.5, width * spread_factor))
        cy = int(np.random.normal(height * 0.55, height * 0.25))

        # Clamp inside boundaries
        cx = max(60, min(width - 60, cx))
        cy = max(75, min(height - 45, cy))

        # Perspective scaling: people closer to bottom are slightly larger
        scale = 0.75 + (cy / float(height)) * 0.55
        head_radius = int(8 * scale)
        body_w = int(22 * scale)
        body_h = int(34 * scale)

        color = clothing_colors[i % len(clothing_colors)]

        # Soft drop shadow
        cv2.ellipse(canvas, (cx, cy + int(body_h * 0.6)), (int(body_w * 0.8), int(head_radius * 0.7)), 0, 0, 360, (20, 22, 28), -1)

        # Body (torso)
        cv2.ellipse(canvas, (cx, cy + int(body_h * 0.2)), (int(body_w * 0.5), int(body_h * 0.45)), 0, 0, 360, color, -1)

        # Head
        cv2.circle(canvas, (cx, cy - int(head_radius * 1.1)), head_radius, (210, 190, 175), -1)
        # Hair/hat
        cv2.ellipse(canvas, (cx, cy - int(head_radius * 1.3)), (head_radius, int(head_radius * 0.7)), 0, 180, 360, (30, 25, 20), -1)

    # Draw simulation watermark badge
    cv2.rectangle(canvas, (width - 290, height - 38), (width - 15, height - 12), (15, 23, 42), -1)
    cv2.rectangle(canvas, (width - 290, height - 38), (width - 15, height - 12), (59, 130, 246), 1)
    cv2.putText(canvas, f"[SIMULATED SCENARIO] {num_people} Persons", (width - 280, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (147, 197, 253), 1)

    return canvas


def ensure_sample_assets(target_dir: str):
    """Generates standard sample images and video for instant 1-click testing."""
    os.makedirs(target_dir, exist_ok=True)

    samples = [
        ("sample_safe_crowd.jpg", 22, "SAFE NORMAL FLOW - CONCOURSE"),
        ("sample_warning_crowd.jpg", 68, "WARNING LEVEL - SUBWAY INGRESS"),
        ("sample_critical_surge.jpg", 155, "CRITICAL CONGESTION - EXIT CHOKEPOINT"),
        ("sample_stampede_risk.jpg", 240, "STAMPEDE RISK - SEVERE CAPACITY CRUSH")
    ]

    for fname, count, title in samples:
        fpath = os.path.join(target_dir, fname)
        if not os.path.exists(fpath):
            img = create_synthetic_crowd_image(num_people=count, title_banner=title)
            cv2.imwrite(fpath, img)

    # Also generate a 60-frame sample MP4 video if not present
    video_path = os.path.join(target_dir, "sample_crowd_surge.mp4")
    if not os.path.exists(video_path):
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(video_path, fourcc, 15.0, (800, 450))
            for frame_idx in range(45):
                # People count gradually surges from 20 to 110
                current_p = int(20 + (frame_idx / 45.0) * 90)
                frame = create_synthetic_crowd_image(num_people=current_p, width=800, height=450)
                # Frame timestamp text
                cv2.putText(frame, f"REC 00:00:{frame_idx:02d} | FPS: 15", (25, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                vw.write(frame)
            vw.release()
        except Exception as e:
            print(f"[Simulation] Video generation notice: {e}")
