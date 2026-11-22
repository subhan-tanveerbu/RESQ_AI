import streamlit as st

def page_header(kicker, title, subtitle):
    st.caption(kicker.upper())
    st.title(title)
    st.write(subtitle)
    st.divider()

def metric_strip(info):
    cols = st.columns(5)
    items = [("RESCUED",info.get("rescued",0)),("REMAINING",info.get("waiting",0)),
             ("ENERGY",info.get("energy",0)),("HEALTH",info.get("health",0)),
             ("STEP",info.get("step",0))]
    for c,(label,value) in zip(cols,items):
        c.metric(label,value)
