import streamlit as st
from config import RescueConfig
from environment.rescue_env import RescueEnv
from dashboard.components import page_header, metric_strip

page_header("RESQ AI / OPERATIONS", "Mission Control", "Autonomous disaster response overview.")
if "env" not in st.session_state:
    st.session_state.env = RescueEnv(RescueConfig())
env = st.session_state.env
info = env.info()
st.markdown("## SECTOR 07 — AFTERSHOCK ZONE")
st.caption("EARTHQUAKE RESPONSE SIMULATION · LIVE ENVIRONMENT STATE")
metric_strip(info)
left, right = st.columns([1.7,1])
with left:
    st.subheader("Mission overview")
    st.write("A rescue robot is deployed into a partially observable urban district. It must locate victims, manage resources, avoid hazards, and evacuate survivors.")
    st.progress(min(1.0, info["step"]/env.config.max_steps), text="Mission time elapsed")
    st.markdown("**Current status**")
    st.success("Robot operational" if info["health"] > 0 else "Robot disabled")
    st.write(f"Mission step: **{info['step']} / {env.config.max_steps}**")
    st.write(f"Mission result: **{info['reason'] or 'In progress'}**")
with right:
    st.subheader("Rescue priorities")
    for label, count in [("Critical rescued",info["critical_rescued"]),("Victims carried",info["carried"]),("Victims lost",info["lost"])]:
        st.metric(label,count)
    if st.button("Create new mission", type="primary", use_container_width=True):
        st.session_state.env = RescueEnv(RescueConfig(seed=env.config.seed+1))
        st.rerun()
st.info("Use Live Simulation to control the environment, or RL Training Lab to train the learning agent.")
