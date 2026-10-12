"""
Reusable Apple-inspired UI components, styles, and Lucide SVG icons.
Designed according to Apple Human Interface Guidelines (HIG):
- Restrained color with semantic hierarchy
- Clean typography (Inter, JetBrains Mono)
- Crisp Lucide SVG icons (zero emojis)
- Tactile elevated cards with subtle 1px border
- Smooth form controls and inputs
- Meaningful whitespace and visual rhythm
"""

import streamlit as st

# Standard Lucide Icons as crisp inline SVGs
LUCIDE_ICONS = {
    "laptop": '<path d="M20 16V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9m16 0H4m16 0 1.28 2.55a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45L4 16"/>',
    "tag": '<path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"/><circle cx="7.5" cy="7.5" r=".5" fill="currentColor"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "trending-down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.9a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 12.5-9.17 4.16a2 2 0 0 1-1.66 0L2 12.5"/><path d="m22 17.5-9.17 4.16a2 2 0 0 1-1.66 0L2 17.5"/>',
    "grid": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 12h18"/><path d="M12 3v18"/>',
    "flask-conical": '<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "check-circle": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "indian-rupee": '<path d="M6 3h12"/><path d="M6 8h12"/><path d="m6 13 8.5 8"/><path d="M6 13h3a4 4 0 0 0 0-8"/>',
    "bar-chart-2": '<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "boxes": '<path d="M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z"/><path d="m7 16.5-4.74-2.85"/><path d="m7 16.5 5-3"/><path d="M7 16.5v5.17"/><path d="M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z"/><path d="m17 16.5-5-3"/><path d="m17 16.5 4.74-2.85"/><path d="M17 16.5v5.17"/><path d="M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z"/><path d="M12 8 7.26 5.15"/><path d="m12 8 4.74-2.85"/><path d="M12 13.5V8"/>'
}

def lucide_icon(name: str, size: int = 18, color: str = "currentColor") -> str:
    """Returns an inline Lucide SVG string."""
    svg_paths = LUCIDE_ICONS.get(name, LUCIDE_ICONS["activity"])
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-{name}" style="vertical-align: middle; display: inline-block;">{svg_paths}</svg>'


def apply_custom_css():
    """Injects refined global CSS for Apple-grade typography, smooth forms, and cards."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
        color: #F8FAFC;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', "SF Mono", Menlo, Consolas, monospace !important;
    }

    /* Page container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1200px;
    }

    /* Column & Container Alignment */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }

    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }

    div[data-testid="column"] > div {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        height: 100% !important;
    }

    /* Container Cards matching Apple HIG */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #161F30 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 1.25rem 1.4rem !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25) !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(255, 255, 255, 0.14) !important;
    }

    /* Card Containers */
    .apple-card {
        background: #161F30;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        box-sizing: border-box;
    }

    .apple-card:hover {
        border-color: rgba(255, 255, 255, 0.14);
    }

    /* Metric Cards */
    .apple-metric-box {
        background: #161F30;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 140px;
        height: 100%;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        box-sizing: border-box;
        transition: border-color 0.2s ease;
    }

    .apple-metric-box:hover {
        border-color: rgba(255, 255, 255, 0.14);
    }

    .apple-metric-label {
        font-size: 0.8125rem;
        font-weight: 500;
        color: #94A3B8;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    .apple-metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .apple-metric-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.4rem;
    }

    .apple-metric-delta-pos {
        display: inline-flex;
        align-items: center;
        font-size: 0.775rem;
        font-weight: 600;
        color: #34D399;
        margin-top: 0.25rem;
        gap: 0.25rem;
    }

    .apple-metric-delta-neg {
        display: inline-flex;
        align-items: center;
        font-size: 0.775rem;
        font-weight: 600;
        color: #F87171;
        margin-top: 0.25rem;
        gap: 0.25rem;
    }

    /* Category Tag (clean 4px border-radius, no oval pills) */
    .apple-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        gap: 0.35rem;
    }

    .apple-badge-blue {
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    .apple-badge-green {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .apple-badge-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.25);
    }

    .apple-badge-amber {
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    /* Smooth Form Controls & Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: rgba(255, 255, 255, 0.25) !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }

    /* Number & text inputs */
    div[data-baseweb="input"] > div {
        background-color: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }

    div[data-baseweb="input"] > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border: 1px solid transparent !important;
    }

    .stButton > button[kind="primary"] {
        background: #2563EB !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1D4ED8 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.45) !important;
    }

    .stButton > button[kind="secondary"] {
        background: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #F8FAFC !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #334155 !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Checkbox & Radio */
    div[data-testid="stCheckbox"] label, div[data-testid="stRadio"] label {
        font-size: 0.9rem !important;
        color: #E2E8F0 !important;
    }

    /* Tab navigation polish */
    button[data-baseweb="tab"] {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding-top: 0.75rem !important;
        padding-bottom: 0.75rem !important;
        transition: all 0.15s ease !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        font-weight: 600 !important;
        color: #60A5FA !important;
    }

    /* Empty states */
    .apple-empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        background: #161F30;
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        margin: 1.5rem 0;
    }

    .apple-empty-state-icon {
        margin-bottom: 0.75rem;
        color: #64748B;
    }

    .apple-empty-state-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 0.25rem;
    }

    .apple-empty-state-desc {
        font-size: 0.875rem;
        color: #94A3B8;
        max-width: 480px;
        margin: 0 auto;
    }

    /* Callouts */
    .apple-callout {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        font-size: 0.875rem;
        line-height: 1.5;
    }

    .apple-callout-info {
        background: rgba(30, 58, 138, 0.25);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #93C5FD;
    }

    .apple-callout-success {
        background: rgba(6, 78, 59, 0.25);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6EE7B7;
    }

    .apple-callout-warning {
        background: rgba(120, 53, 15, 0.25);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #FCD34D;
    }
    </style>
    """, unsafe_allow_html=True)


def render_page_header(
    title: str,
    subtitle: str,
    icon_name: str = "laptop",
    **kwargs
):
    """Renders a refined Apple HIG page header with crisp typography."""
    apply_custom_css()
    st.title(title)
    icon_html = f'{lucide_icon(icon_name, size=18, color="#60A5FA")} ' if icon_name else ""
    sub_html = (
        f'<div style="display: flex; align-items: center; gap: 0.5rem; margin-top: -0.75rem; margin-bottom: 1.5rem; font-size: 1.05rem; color: #94A3B8; font-weight: 400; line-height: 1.45;">'
        f'{icon_html}<span>{subtitle}</span>'
        f'</div>'
    )
    st.markdown(sub_html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    subtext: str = None,
    delta: str = None,
    delta_positive: bool = True,
    icon_name: str = None
):
    """Renders a standardized elevated metric card with zero markdown indentation leaks."""
    icon_html = f'{lucide_icon(icon_name, size=15, color="#60A5FA")} ' if icon_name else ""
    delta_html = ""
    if delta:
        cls = "apple-metric-delta-pos" if delta_positive else "apple-metric-delta-neg"
        delta_icon = lucide_icon("trending-up" if delta_positive else "trending-down", size=13)
        delta_html = f'<div class="{cls}">{delta_icon}{delta}</div>'

    sub_html = f'<div class="apple-metric-subtext">{subtext}</div>' if subtext else ""

    card_html = (
        f'<div class="apple-metric-box">'
        f'<div>'
        f'<div class="apple-metric-label">{icon_html}{label}</div>'
        f'<div class="apple-metric-value">{value}</div>'
        f'{delta_html}'
        f'</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = None, icon_name: str = None):
    """Renders a clean section boundary with generous negative space and a Lucide icon."""
    icon_html = f'{lucide_icon(icon_name, size=18, color="#94A3B8")} ' if icon_name else ""
    sub_html = f'<p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #94A3B8;">{subtitle}</p>' if subtitle else ""
    section_html = (
        f'<div style="margin: 2.25rem 0 1rem 0;">'
        f'<h3 style="margin: 0; font-size: 1.35rem; font-weight: 600; color: #F8FAFC; letter-spacing: -0.015em; display: flex; align-items: center; gap: 0.5rem;">'
        f'{icon_html}{title}'
        f'</h3>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(section_html, unsafe_allow_html=True)


def render_empty_state(title: str, description: str, icon_name: str = "search"):
    """Renders a friendly and actionable empty state with a Lucide icon."""
    empty_html = (
        f'<div class="apple-empty-state">'
        f'<div class="apple-empty-state-icon">{lucide_icon(icon_name, size=32, color="#64748B")}</div>'
        f'<div class="apple-empty-state-title">{title}</div>'
        f'<div class="apple-empty-state-desc">{description}</div>'
        f'</div>'
    )
    st.markdown(empty_html, unsafe_allow_html=True)


def render_spec_card(
    title: str,
    items: list,
    icon_name: str = None,
    icon_color: str = "#60A5FA"
):
    """Renders a uniform, balanced Apple HIG specification card with key-value rows."""
    icon_html = f'{lucide_icon(icon_name, size=18, color=icon_color)} ' if icon_name else ""
    rows_html = ""
    for label, val in items:
        rows_html += (
            f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.65rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-size: 0.875rem;">'
            f'<span style="color: #94A3B8; font-weight: 500;">{label}</span>'
            f'<span style="color: #F8FAFC; font-weight: 600; text-align: right;">{val}</span>'
            f'</div>'
        )
    card_html = (
        f'<div class="apple-card" style="margin-bottom: 0; height: 100%; display: flex; flex-direction: column;">'
        f'<div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">'
        f'{icon_html}{title}'
        f'</div>'
        f'<div style="display: flex; flex-direction: column; flex: 1 1 auto; justify-content: space-between;">'
        f'{rows_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_callout(text: str = None, title: str = None, variant: str = "info", icon_name: str = "info"):
    """No-op: Callout boxes removed per user design direction."""
    pass
