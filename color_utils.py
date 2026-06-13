class ColorUtils:
    @staticmethod
    def get_gradient_color(val):
        if val >= 1600:
            r, g, b = 0, 0.39, 0
        elif val >= 1000:
            r, g, b = 0, 1, 0
        elif val >= 800:
            r, g, b = 1, 1, 0
        elif val >= 600:
            r, g, b = 1, 0.65, 0
        else:
            r, g, b = 1, 0, 0
        return (r, g, b, 0.5) 