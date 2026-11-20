"""
Smart Crowd AI: Risk Scoring and Alert Engine
Implements multi-factor crowd density analysis based on John J. Fruin's Level of Service (LOS)
and stampede dynamics modeling.
"""

from typing import Dict, Any, Tuple, Optional


class CrowdRiskEngine:
    """Calculates risk levels, scores, and actionable advisories for crowd densities."""

    def __init__(
        self,
        warning_density: float = 2.0,      # people per sqm
        critical_density: float = 3.5,     # people per sqm
        stampede_density: float = 5.0,     # people per sqm
        capacity_warning_pct: float = 75.0,
        capacity_critical_pct: float = 90.0
    ):
        self.warning_density = warning_density
        self.critical_density = critical_density
        self.stampede_density = stampede_density
        self.capacity_warning_pct = capacity_warning_pct
        self.capacity_critical_pct = capacity_critical_pct

    def evaluate_risk(
        self,
        person_count: int,
        area_sqm: float = 200.0,
        max_capacity: Optional[int] = None,
        surge_rate_per_sec: float = 0.0,
        cluster_factor: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluates risk score (0-100), risk category, and recommended emergency actions.

        Parameters:
        - person_count: Detected number of people
        - area_sqm: Surface area in square meters (default 200 sqm if unspecified)
        - max_capacity: Rated capacity limit of the space
        - surge_rate_per_sec: Rate of crowd increase (positive = rapid influx)
        - cluster_factor: Spatial clustering metric (0.0=uniform, 1.0=dense choke point)
        """
        if area_sqm <= 0:
            area_sqm = 100.0

        # Calculate base crowd density (people per square meter)
        density = round(person_count / area_sqm, 2)

        # 1. Density Score component (0 to 50 points)
        if density <= 0.5:
            density_score = (density / 0.5) * 15.0
        elif density <= self.warning_density:
            density_score = 15.0 + ((density - 0.5) / (self.warning_density - 0.5)) * 25.0
        elif density <= self.critical_density:
            density_score = 40.0 + ((density - self.warning_density) / (self.critical_density - self.warning_density)) * 25.0
        elif density <= self.stampede_density:
            density_score = 65.0 + ((density - self.critical_density) / (self.stampede_density - self.critical_density)) * 25.0
        else:
            density_score = 90.0 + min(10.0, (density - self.stampede_density) * 5.0)

        # 2. Capacity Occupancy component (0 to 25 points)
        if max_capacity and max_capacity > 0:
            occupancy_pct = (person_count / max_capacity) * 100.0
            if occupancy_pct < self.capacity_warning_pct:
                capacity_score = (occupancy_pct / self.capacity_warning_pct) * 10.0
            elif occupancy_pct < self.capacity_critical_pct:
                capacity_score = 10.0 + ((occupancy_pct - self.capacity_warning_pct) / (self.capacity_critical_pct - self.capacity_warning_pct)) * 10.0
            else:
                capacity_score = 20.0 + min(5.0, ((occupancy_pct - self.capacity_critical_pct) / 20.0) * 5.0)
        else:
            occupancy_pct = (person_count / (area_sqm * 2.0)) * 100.0
            capacity_score = min(25.0, (occupancy_pct / 100.0) * 25.0)

        # 3. Surge Rate component (0 to 15 points)
        # Influx of > 5 people/sec indicates sudden rush or bottleneck
        surge_score = min(15.0, max(0.0, surge_rate_per_sec * 2.5))

        # 4. Clustering / Bottleneck factor (0 to 10 points)
        cluster_score = cluster_factor * 10.0

        # Aggregate composite score
        composite_score = round(min(100.0, max(0.0, density_score + capacity_score + surge_score + cluster_score)), 1)

        # Determine level & action advisory
        if composite_score >= 85.0 or density >= self.stampede_density:
            risk_level = "STAMPEDE RISK"
            color = "#EF4444"
            badge_class = "danger"
            advisory = (
                "EMERGENCY ALERT: Critical crush danger! Drop all turnstile barriers immediately, "
                "open emergency relief exits 2A/2B, halt incoming flow, and broadcast evacuation routing."
            )
            trigger_alert = True
        elif composite_score >= 65.0 or density >= self.critical_density:
            risk_level = "CRITICAL"
            color = "#F97316"
            badge_class = "critical"
            advisory = (
                "WARNING: High density bottleneck detected! Deploy perimeter crowd marshals, "
                "meter incoming ingress gates to 30% flow, and activate one-way pedestrian pathways."
            )
            trigger_alert = True
        elif composite_score >= 40.0 or density >= self.warning_density:
            risk_level = "WARNING"
            color = "#F59E0B"
            badge_class = "warning"
            advisory = (
                "CAUTION: Elevated crowd buildup. Monitor gate transit rate closely and prepare "
                "secondary buffer queue lines."
            )
            trigger_alert = True
        elif composite_score >= 20.0:
            risk_level = "MODERATE"
            color = "#3B82F6"
            badge_class = "moderate"
            advisory = "Normal crowd presence with steady circulation. Maintain standard monitoring."
            trigger_alert = False
        else:
            risk_level = "SAFE"
            color = "#10B981"
            badge_class = "safe"
            advisory = "Crowd levels well within optimal safety thresholds. Free movement unimpeded."
            trigger_alert = False

        # Fruin's Level of Service (LOS) grade
        if density < 0.5:
            fruin_los = "LOS A (Free Flow)"
        elif density < 1.0:
            fruin_los = "LOS B (Reasonably Free)"
        elif density < 1.8:
            fruin_los = "LOS C (Noticeably Restricted)"
        elif density < 2.5:
            fruin_los = "LOS D (Severely Restricted)"
        elif density < 3.8:
            fruin_los = "LOS E (Capacity Threshold)"
        else:
            fruin_los = "LOS F (Breakdown / Crush Hazard)"

        return {
            "density": density,
            "person_count": person_count,
            "occupancy_pct": round(occupancy_pct, 1),
            "risk_score": composite_score,
            "risk_level": risk_level,
            "color": color,
            "badge_class": badge_class,
            "advisory": advisory,
            "fruin_los": fruin_los,
            "trigger_alert": trigger_alert
        }
