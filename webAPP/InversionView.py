import streamlit as st
import random
import os
import json

from utils import *
st.set_page_config(layout="wide")

def click_random(examples):
    st.session_state.sel_example_idx = random.randint(0, len(examples)-1)

chosen_dir = st.sidebar.selectbox("task", ["3 Digit Addition", "Char Counting", "IOI", "Factual Recall"], index=0)

if chosen_dir == "3 Digit Addition":
    root_path = "../training_outputs/cached_addition_generation"
elif chosen_dir == "Char Counting":
    root_path = "../training_outputs/cached_counting_generation"
elif chosen_dir == "IOI":
    root_path = "../training_outputs/cached_ioi_generation"
elif chosen_dir == "Factual Recall":
    root_path = "../training_outputs/cached_fact_generation"
examples = os.listdir(root_path)

if ("sel_example_idx" not in st.session_state) or (len(examples) <= st.session_state.sel_example_idx):
    st.session_state.sel_example_idx = random.randint(0, len(examples)-1)
st.session_state.sel_example = st.sidebar.selectbox("select an example", examples, index=st.session_state.sel_example_idx)

st.sidebar.button("random example", type="primary", on_click=click_random, args=(examples, chosen_dir, root_path))


if "saved_examples" not in st.session_state:
    st.session_state.saved_examples = []

sel_example_dir = os.path.join(root_path, st.session_state.sel_example)

all_probed_act = []
all_q_pos = []
all_temperature = []

for item in os.listdir(sel_example_dir):
    probed_act, q_pos, temperature, _ = item.rstrip(".json").split("-")
    all_probed_act.append(probed_act)
    all_q_pos.append(q_pos)
    all_temperature.append(temperature)

# all_probed_act = sorted(list(set(all_probed_act)))
all_short_name = sorted([site_name_to_short_name(n) for n in set(all_probed_act)])
all_q_pos = sorted(list(set(all_q_pos)), key=lambda x: int(x))
all_temperature = sorted(list(set(all_temperature))) #, key=lambda x: float(x))


with st.sidebar:

    if "sel_short_name" in st.session_state and st.session_state.sel_short_name in all_short_name:
        index = all_short_name.index(st.session_state.sel_short_name)
    else:
        index = None
    sel_short_name = st.selectbox("activation site", all_short_name, index=index, placeholder="select an activation site", key="selected_act_site")
    sel_probed_act = short_name_to_site_name(sel_short_name)

    if chosen_dir == "3 Digit Addition":
        captions = ["\+", "=", "A1", "A2", "A3"]
    elif chosen_dir == "Char Counting":
        captions = ["\|", "query char", ":"]
    elif chosen_dir == "IOI":
        captions = ["S1+1", "S2", "END"]
    elif chosen_dir == "Factual Recall":
        captions = ["SUBJ", "END"]
        
    sel_cap_pos = st.radio("select a position", captions, index=1, horizontal=True)
    sel_q_pos = all_q_pos[captions.index(sel_cap_pos)]

    sel_temperatures = st.multiselect("select a sampling temperature", all_temperature, ["Auto"], placeholder="results are combined when selecting multiple options")
    if len(sel_temperatures) == 0:
        st.warning("select at least one temperature")
        st.stop()

    sel_metric = st.radio("metric", ["normalized euclidean distance", "1 - cosine similarity"], index=(1 if chosen_dir in ["IOI", "Factual Recall"] else 0), horizontal=True)
    metric_mapping = {"normalized euclidean distance": "dist", "1 - cosine similarity": "sim"}
    sel_metric = metric_mapping[sel_metric]

    color_on = st.toggle("show color", value=(chosen_dir in ["3 Digit Addition", "Factual Recall"]))
    des_on = st.toggle("show description", value=True)
    v_mode_on = st.toggle("visual mode", value=True)

    display_num = st.slider("display number", 1, 100*len(sel_temperatures), 10)

generation_paths = [os.path.join(sel_example_dir, f"{sel_probed_act}-{sel_q_pos}-{tp}-{sel_metric}.json") for tp in sel_temperatures]

def show_generation(generation_paths, color_on, v_mode_on, des_on, display_num, fixed_pos, chosen_dir, sel_generation_idx=None):
    if (not all([os.path.exists(path) for path in generation_paths])) or (len(generation_paths) == 0):
        st.warning("file does not exist, select other parameters on the left side bar")
        st.stop()

    gen_objs = []
    for path in generation_paths:
        with open(path, "r", encoding="utf-8") as f:
            gen_objs.append(json.load(f))
    merged_obj = merge_gen_obj(gen_objs)

    if "hook_result" in sel_probed_act:
        if des_on:
            st.subheader("Attention weights")
            st.caption("the query corresponds to the token in ( ). :rainbow[Hover] to see the excact weight")
        str_tokens = mark_position(merged_obj["query_input"], merged_obj["query_position"], "&nbsp;" if chosen_dir not in ["IOI", "Factual Recall"] else "")
        row = f'<div>' + "".join([span_maker_attn(*temp) for temp in zip(str_tokens, merged_obj["attn_weights"])]) + '</div>'
        st.markdown(row, unsafe_allow_html=True)
        st.text("")

    if des_on:
        st.subheader("Generation result")
        st.caption("In the query input, the activation corresponds to token in ( ) is taken as the query activation")
        st.caption("In generated inputs, the metric value for distance/similarity is calculated. The query activation is compared with activations correspond to all tokens in a input. :rainbow[Hover] each token to see its metric value. The metric value for the most similar activation is shown on the left, and its token is in ( ).")
        st.caption("the color indicates what the decoder can know when it can see the query activation compared to when it cannot. Blue color means the probability increase and red color means it decreases")
    st.text("Query Input")

    str_tokens = mark_position(merged_obj["query_input"], merged_obj["query_position"], "&nbsp;" if chosen_dir not in ["IOI", "Factual Recall"] else "")
    metric_sim = generation_paths[0][-8:-5] == "sim"
    value_self = bar_maker(0.0) + f'<span> {0.0:.3f} ; </span>' if v_mode_on else f'<span> {0.0:.3f} ; </span>'
    row = f'<div style="margin-top: -10px; margin-bottom: 12px;">' + value_self + "".join([span_maker(*temp) for temp in zip(str_tokens, [0.0] * len(str_tokens) , [None] * len(str_tokens))]) + '</div>'
    st.markdown(row, unsafe_allow_html=True)
    st.text('Generated Samples') # st.divider()
    if sel_generation_idx is None:
        if ("last_display_param" not in st.session_state) or ((des_on, display_num, len(merged_obj["generation"])) != st.session_state.last_display_param):
            temp_idx = list(range(len(merged_obj["generation"])))
            random.shuffle(temp_idx)
            sel_generation_idx = sorted(temp_idx[:display_num])
        else:
            sel_generation_idx = st.session_state.last_sel_generation_idx
    sel_generation = [merged_obj["generation"][i] for i in sel_generation_idx]
    no_vert_line = True
    epsilon = 0.1 if chosen_dir != "Factual Recall" else 0.25
    for str_tokens, prob_diff, metric_value, best_index, best_value in sel_generation:
        margin_top = ""
        # print(len(str_tokens), len(prob_diff), len(metric_value))
        if chosen_dir == "Char Counting":
            correct_count = str_tokens.count(str_tokens[-4]) - 1
            addition_info = f'<span style="color: gray;"> #: {correct_count} ; </span>'
        else:
            addition_info = ""

        if fixed_pos is None:
            shown_index = best_index
            shown_value = 1-best_value if metric_sim else best_value
            show_vert_line = shown_value > epsilon and no_vert_line and v_mode_on
            shown_value = bar_maker(shown_value) + f"<span> {shown_value:.3f} ; </span>" if v_mode_on else f"<span> {shown_value:.3f} ; </span>"
        else:
            shown_index = fixed_pos if fixed_pos >= 0 else (len(str_tokens)+fixed_pos)
            shown_value = metric_value[shown_index] if 0 <= shown_index < len(metric_value) else float("nan")
            shown_value = 1-shown_value if metric_sim else shown_value
            show_vert_line = shown_value > epsilon and no_vert_line and v_mode_on
            shown_value = bar_maker(shown_value, rounded=True) + f"<span> <i>{shown_value:.3f}</i> ; </span>" if v_mode_on else f"<span> <i>{shown_value:.3f}</i> ; </span>"
        str_tokens = mark_position(str_tokens, shown_index, "&nbsp;" if chosen_dir not in ["IOI", "Factual Recall"] else "")
        if not color_on:
            prob_diff = [0.0] * len(prob_diff)

        if show_vert_line:
            st.markdown('<hr style="width:15em; height: 2px; margin-top: 0px; margin-bottom: 0px; color: #808080; background-color: #808080;" >', unsafe_allow_html=True)   # <span style="margin-top: -10px; font-size: 12px;">0.1</span></hr>
            no_vert_line = False
            margin_top = "margin-top: -10px;"
        row = f'<div style="margin-bottom: 4px; {margin_top}">' + shown_value + addition_info + "".join([span_maker(*temp, flip_value=metric_sim) for temp in zip(str_tokens, prob_diff, metric_value)]) + '</div>'

        st.markdown(row, unsafe_allow_html=True)

    st.session_state.last_display_param = (des_on, display_num, len(merged_obj["generation"]))
    st.session_state.last_sel_generation_idx = sel_generation_idx


with st.sidebar:
    st.write("save generation temporarily and compare")
    if st.button("save"):
        st.session_state.saved_examples.append((generation_paths, sel_probed_act, st.session_state.last_sel_generation_idx))
        st.toast("generation saved!")
    if st.button("rm last") and (len(st.session_state.saved_examples) > 0):
        del st.session_state.saved_examples[-1]
        st.toast("last generation removed!")
    if st.button("rm first") and (len(st.session_state.saved_examples) > 0):
        del st.session_state.saved_examples[0]
        st.toast("first generation removed!")
    if st.button("clear"):
        st.session_state.saved_examples.clear()
        st.toast("all generation removed!")

    st.text(f"num saved:{len(st.session_state.saved_examples)}")

    comparing = st.button("compare", type="primary")

    fixed_pos = st.number_input("show metric value for fixed pos", value=None, step=1, placeholder="can be negative")

if comparing:
    if len(st.session_state.saved_examples) == 0:
        st.error("empty list, you should first save")
    else:
        cols = st.columns(len(st.session_state.saved_examples))
        for i, col in enumerate(cols):
            with col:
                generation_paths, sel_probed_act, sel_generation_idx = st.session_state.saved_examples[i]
                # if des_on:
                #     st.subheader("Activation site")
                # st.write(sel_probed_act)
                st.markdown("Activation Site: "+"$$"+site_name_to_latex_name(sel_probed_act)+"$$")
                show_generation(generation_paths, color_on, v_mode_on, des_on, display_num, fixed_pos, chosen_dir, sel_generation_idx)
else:
    show_generation(generation_paths, color_on, v_mode_on, des_on, display_num, fixed_pos, chosen_dir)
