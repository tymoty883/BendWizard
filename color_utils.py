"""
Stress visualization color mapping utilities.

Maps numeric stress/curvature values to RGBA colors for visualization.
Uses color gradient: Red (low) → Orange → Yellow → Green (high)
"""

from typing import Tuple
import constants


class ColorUtils:
    """Map numeric values to visualization colors for stress/curvature mapping."""
    
    @staticmethod
    def get_gradient_color(value: float) -> Tuple[float, float, float, float]:
        """
        Map a numeric value to RGBA color using stress-based gradient.
        
        Color scale represents bore radius (in meters):
        - Red (< 600m): Critical - very small bore
        - Orange (600-800m): High stress zone
        - Yellow (800-1000m): Medium stress zone
        - Green (1000-1600m): Low stress zone
        - Dark Green (> 1600m): Safe - large bore
        
        Args:
            value: Numeric value to map (e.g., bore radius in meters)
            
        Returns:
            RGBA tuple with components in range [0, 1]
            Alpha channel fixed at 0.5 for transparency
        """
        if value >= constants.STRESS_COLOR_THRESHOLD_HIGH:
            # Dark green: safe, large bore
            return constants.STRESS_COLOR_VERY_HIGH
        elif value >= constants.STRESS_COLOR_THRESHOLD_MEDIUM:
            # Light green: low stress
            return constants.STRESS_COLOR_HIGH
        elif value >= constants.STRESS_COLOR_THRESHOLD_LOW:
            # Yellow: medium stress
            return constants.STRESS_COLOR_MEDIUM
        elif value >= constants.STRESS_COLOR_THRESHOLD_VERY_LOW:
            # Orange: high stress
            return constants.STRESS_COLOR_LOW
        else:
            # Red: critical - very small bore
            return constants.STRESS_COLOR_VERY_LOW 