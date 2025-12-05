class ColorUtils:
    @staticmethod
    def get_gradient_color(val):
        # Map val in [-1, 1] to color: red if <0, yellow if near 0, green if >0
        if val < 0:
            return (1.0, 0.2, 0.2, 1.0)
        elif val < 0.05:
            return (1.0, 1.0, 0.3, 1.0)
        else:
            return (0.2, 1.0, 0.2, 1.0)
