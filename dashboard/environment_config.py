import streamlit as st
from pathlib import Path
from config import RescueConfig
from environment.rescue_env import RescueEnv
from dashboard.components import page_header

ROOT = Path(__file__).resolve().parents[1]
page_header("RESQ AI / SYSTEM", "Environment Configuration", "Configure a new scenario. Existing experiment files remain unchanged.")
cfg = RescueConfig()
a,b = st.columns(2)
with a:
    size = st.slider("Map size",8,20,12)
    victims = st.slider("Victim count",1,15,6)
    rubble = st.slider("Rubble count",0,50,12)
    difficulty = st.selectbox("Difficulty",["easy","normal","hard"],index=1)
with b:
    fire = st.slider("Fire zones",0,10,2)
    gas = st.slider("Gas zones",0,10,2)
    steps = st.slider("Mission steps",50,500,250,25)
    seed = st.number_input("Random seed",min_value=0,value=7)
if st.button("Save scenario configuration",type="primary"):
    cfg = RescueConfig(map_size=size,victim_count=victims,rubble_count=rubble,fire_count=fire,
                       gas_count=gas,max_steps=steps,difficulty=difficulty,seed=int(seed)).validate()
    path = ROOT/"data/scenarios/active_config.json"
    cfg.save(path)
    st.session_state.env = RescueEnv(cfg)
    st.success(f"Configuration saved to {path.relative_to(ROOT)} and a new mission created.")
st.info("Configuration changes apply to newly created missions. Previously saved training and evaluation CSV files are not overwritten.")
