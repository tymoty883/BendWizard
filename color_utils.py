from typing import Optional, Sequence, Tuple
import constants


class ColorUtils:
    """Map numeric values to visualization colors for stress/curvature mapping."""
    
    @staticmethod
    def get_gradient_color(
        value: float,
        scale_min: Optional[float] = None,
        scale_max: Optional[float] = None,
        average: Optional[float] = None,
        quantile_thresholds: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Color scale represents bore curvature radius (in meters).

        If ``scale_min`` and ``scale_max`` are provided, thresholds are distributed
        evenly between those values while keeping the same 5 colors.
        Otherwise, legacy fixed thresholds from constants are used.
        
        Args:
            value: Numeric value to map (e.g., bore radius in meters)
            scale_min: Optional lower bound for dynamic scaling
            scale_max: Optional upper bound for dynamic scaling
            average: Optional average used to classify outliers (> 2x average)
            quantile_thresholds: Optional 4-element sequence with q20/q40/q60/q80
            
        Returns:
            RGBA tuple with components in range [0, 1]
            Alpha channel fixed at 0.5 for transparency
        """

        if quantile_thresholds is not None and len(quantile_thresholds) == 4:
            threshold_very_low = float(quantile_thresholds[0])
            threshold_low = float(quantile_thresholds[1])
            threshold_medium = float(quantile_thresholds[2])
            threshold_high = float(quantile_thresholds[3])

        elif scale_min is not None and scale_max is not None:
            if scale_max <= scale_min:
                # Degenerate range: return a stable middle color.
                return constants.STRESS_COLOR_MEDIUM

            step = (scale_max - scale_min) / 5.0
            threshold_very_low = scale_min + step
            threshold_low = scale_min + 2.0 * step
            threshold_medium = scale_min + 3.0 * step
            threshold_high = scale_min + 4.0 * step
        else:
            threshold_very_low = constants.STRESS_COLOR_THRESHOLD_VERY_LOW
            threshold_low = constants.STRESS_COLOR_THRESHOLD_LOW
            threshold_medium = constants.STRESS_COLOR_THRESHOLD_MEDIUM
            threshold_high = constants.STRESS_COLOR_THRESHOLD_HIGH

        if value >= threshold_high:
            return constants.STRESS_COLOR_VERY_HIGH
        elif value >= threshold_medium:
            return constants.STRESS_COLOR_HIGH
        elif value >= threshold_low:
            return constants.STRESS_COLOR_MEDIUM
        elif value >= threshold_very_low:
            return constants.STRESS_COLOR_LOW
        else:
            return constants.STRESS_COLOR_VERY_LOW 