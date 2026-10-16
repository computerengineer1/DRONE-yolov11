"""
UI Theme — Color palette, fonts, and styling constants for CustomTkinter.
"""

# ──────────────────────────── Color Palette ──────────────────────────
# Dark tactical background shades
BG_DARK = "#0a0a12"
BG_PANEL = "#0f0f1a"
BG_CARD = "#14142a"
BG_HOVER = "#1a1a35"

# Neon accent colors
NEON_CYAN = "#00ffff"
NEON_BLUE = "#00b4ff"
TACTICAL_RED = "#ff2244"
TACTICAL_YELLOW = "#ffcc00"
TACTICAL_GREEN = "#00ff66"

# Text colors
TEXT_PRIMARY = "#e0e0e0"
TEXT_SECONDARY = "#888899"
TEXT_MUTED = "#555566"

# Border / line colors
BORDER_COLOR = "#222244"
BORDER_ACCENT = "#00ffff"

# ──────────────────────────── Fonts ──────────────────────────────────
FONT_FAMILY = "Consolas"
FONT_TITLE = (FONT_FAMILY, 16, "bold")
FONT_HEADING = (FONT_FAMILY, 13, "bold")
FONT_BODY = (FONT_FAMILY, 11)
FONT_SMALL = (FONT_FAMILY, 10)
FONT_MONO = (FONT_FAMILY, 9)

# ──────────────────────────── Button Styles ─────────────────────────
BTN_PRIMARY = {
    "fg_color": "#002244",
    "hover_color": "#003366",
    "border_color": NEON_CYAN,
    "border_width": 1,
    "text_color": NEON_CYAN,
    "corner_radius": 6,
    "font": (FONT_FAMILY, 12, "bold"),
    "height": 42,
}

BTN_DANGER = {
    "fg_color": "#220011",
    "hover_color": "#440022",
    "border_color": TACTICAL_RED,
    "border_width": 1,
    "text_color": TACTICAL_RED,
    "corner_radius": 6,
    "font": (FONT_FAMILY, 12, "bold"),
    "height": 42,
}

BTN_SECONDARY = {
    "fg_color": "#111122",
    "hover_color": "#1a1a35",
    "border_color": BORDER_COLOR,
    "border_width": 1,
    "text_color": TEXT_SECONDARY,
    "corner_radius": 6,
    "font": (FONT_FAMILY, 11),
    "height": 36,
}
