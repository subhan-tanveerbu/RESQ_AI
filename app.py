import streamlit as st
from dashboard.components import page_header

st.set_page_config(page_title="RESQ AI | Rescue Intelligence", page_icon="🚨", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.04em; }
.stApp { background: #F5F3ED; color: #202522; }
[data-testid="stSidebar"] { background: #202522; }
[data-testid="stSidebar"] * { color: #F5F3ED !important; }
.stMetric { background: #FFFFFF; padding: 14px; border: 1px solid #E5E3DC; }
.stButton button { border-radius: 5px; font-weight: 600; }
div[data-testid="stAlert"] { border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

pages = {
    "OPERATIONS": [
        st.Page("dashboard/mission_control.py", title="Mission Control", icon=":material/radar:", default=True),
        st.Page("dashboard/live_simulation.py", title="Live Simulation", icon=":material/map:")
    ],
    "INTELLIGENCE": [
        st.Page("dashboard/training_lab.py", title="RL Training Lab", icon=":material/model_training:"),
        st.Page("dashboard/performance.py", title="Agent Performance", icon=":material/monitoring:")
    ],
    "SYSTEM": [
        st.Page("dashboard/environment_config.py", title="Environment Config", icon=":material/tune:"),
        st.Page("dashboard/about.py", title="Project / About", icon=":material/info:")
    ]
}
pg = st.navigation(pages)
with st.sidebar:
    st.markdown("# RESQ / AI")
    st.caption("AUTONOMOUS RESCUE INTELLIGENCE")
    st.divider()
    st.caption("BAHRIA UNIVERSITY · BS AI")
    st.caption("SECTOR 07 · EARTHQUAKE RESPONSE")
pg.run()
