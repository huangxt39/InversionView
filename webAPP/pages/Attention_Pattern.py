import streamlit as st
import random
import os
import json
from utils import *


def click_random(idx):
    st.session_state.sel_example_idx = idx

chosen_dir = st.sidebar.selectbox("task", ["3 Digit Addition", "Char Counting"], index=0)

if chosen_dir == "3 Digit Addition":
    root_path = "./cached_attention"
elif chosen_dir == "Char Counting":
    root_path = "./cached_counting_attention"
examples = os.listdir(root_path)

if ("sel_example_idx" not in st.session_state) or (len(examples) <= st.session_state.sel_example_idx):
    st.session_state.sel_example_idx = random.randint(0, len(examples)-1)
st.session_state.sel_example = st.sidebar.selectbox("select an example", examples, index=st.session_state.sel_example_idx)

st.sidebar.button("random example", type="primary", on_click=click_random, args=(random.randint(0, len(examples)-1),))


sel_example = st.session_state.sel_example
sel_example_dir = os.path.join(root_path, sel_example)


all_layer = []
all_head = []

for item in os.listdir(sel_example_dir):
    layer_idx, head_idx = item.rstrip(".json").split("-")
    all_layer.append(layer_idx)
    all_head.append(head_idx)

all_layer = sorted(list(set(all_layer)), key=lambda x: int(x))
all_head = sorted(list(set(all_head)), key=lambda x: int(x))

with st.sidebar:

    sel_layer = st.radio("layer", all_layer, horizontal=True)
    sel_head = st.radio("head", all_head, horizontal=True)
    des_on = st.toggle("show description", value=False)


generation_path = os.path.join(sel_example_dir, f"{sel_layer}-{sel_head}.json")
with open(generation_path, "r") as f:
    obj = json.load(f)

if des_on:
    st.subheader("Attention weights")
    st.caption("the query corresponds to the token in ( ). Hover to see the excact weight")
for i in range(len(obj["str_tokens"])):
    str_tokens = mark_position(obj["str_tokens"], i, "&nbsp;")
    row = f'<div style="margin-bottom: 5px">' + "".join([span_maker_attn_light(*temp) for temp in zip(str_tokens, obj["attn_weights"][i])]) + '</div>'
    st.markdown(row, unsafe_allow_html=True)