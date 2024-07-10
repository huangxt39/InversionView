from collections.abc import Iterable
import re

def span_maker(token: str, color_value: float, hover_value: float | None, flip_value: bool = False):
    if color_value > 0:
        bg_color = "40,116,166"
    else:
        bg_color = "238,75,43"
    if abs(color_value) > 0.6:
        txt_color = "white"
    else:
        txt_color = "black"
    if hover_value is not None:
        hover_text = 'title="%.3f"'%(1-hover_value) if flip_value else 'title="%.3f"'%hover_value
    else:
        hover_text = ''
    return '<span style="background-color:rgba(%s,%.2f); color: %s" %s>'%(bg_color, abs(color_value), txt_color, hover_text) + token + '</span>' 

def span_maker_attn(token: str, color_value: float):
    assert color_value >= 0 and color_value <= 1.0, color_value
    
    # bg_color = "238,75,43"
    # bg_color = "160, 8, 8"
    bg_color = "238,0,30"
    if abs(color_value) > 0.6:
        txt_color = "white"
    else:
        txt_color = "black"

    return '<span style="background-color:rgba(%s,%.2f); color: %s; font-size: 20px; font-weight: bold;" title="%.3f">'%(bg_color, color_value, txt_color, color_value) + token + '</span>' 

def span_maker_attn_light(token: str, color_value: float):
    assert color_value >= 0 and color_value <= 1.0, color_value
    
    # bg_color = "238,75,43"
    # bg_color = "160, 8, 8"
    bg_color = "238,0,30"
    if abs(color_value) > 0.6:
        txt_color = "white"
    else:
        txt_color = "black"

    return '<span style="background-color:rgba(%s,%.2f); color: %s; font-size: 16px;" title="%.3f">'%(bg_color, color_value, txt_color, color_value) + token + '</span>' 

def bar_maker(value, max_value=0.5, max_width=30, rounded=False):
    # max_value=0.8, max_width=46
    bar_width = min(value / max_value * max_width, max_width)
    height = 9

    color = f"rgba(255, 170, 51, {1 - min(value, 1.5) / 2.0})"
    rxry = 'rx="4" ry="4"' if rounded else ''
    return f'<svg width="{max_width}" height="{height+4}" style="margin-right: 5px;"><rect width="{bar_width}" height="{height}" x="{max_width-bar_width}" y="0" {rxry} fill="{color}" /></svg>'


def mark_position(str_tokens: list[str], pos: int, wraper: str = ""):
    return [f"{wraper}({token}){wraper}" if i == pos else f"{wraper}{token}{wraper}" for i, token in enumerate(str_tokens)]

def merge_gen_obj(gen_objs: list[dict]):
    """
    query_input: list[str]
    query_position: int
    probed_act: str
    attn_weights: list[float]
    temperature: float
    metric: str
    generation: list[tuple[...]]
    """

    merged_obj = {}
    for gen_obj in gen_objs:
        for k, v in gen_obj.items():
            if k == "temperature":
                continue
            if k in merged_obj:
                if k != "generation":
                    if isinstance(v, Iterable):
                        assert tuple(merged_obj[k]) == tuple(v)
                    else:
                        assert merged_obj[k] == v
                else:
                    merged_obj[k].extend(v)
            else:
                merged_obj[k] = v

    # merged_obj["generation"].sort(key=lambda x: x[-1])
    merged_obj["generation"].sort(key=lambda x: -x[-1] if merged_obj["metric"] == "sim" else x[-1])
    return merged_obj

def site_name_to_latex_name(probed_act):
    block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))
    if "hook_result" in probed_act:
        head_idx = int(re.search(r"hook_result\.(\d+)", probed_act).group(1))
        return f"a^{{{block_idx},{head_idx}}}"
    elif "hook_resid" in probed_act:
        layer_idx = re.search(r"hook_resid_([a-z]+)", probed_act).group(1)
        return f"x^{{{block_idx},{layer_idx}}}"
    elif "mlp_out" in probed_act:
        return f"m^{{{block_idx}}}"
    else:
        raise NotImplementedError()


def short_name_to_site_name(act_name):
    # a0.1   x0.mid  m0
    if act_name is None:
        return None
    elif act_name.startswith("a"):
        block_idx, head_idx = act_name.lstrip("a").split(".")
        return f"blocks.{block_idx}.attn.hook_result.{head_idx}"
    elif act_name.startswith("x"):
        block_idx, layer_idx = act_name.lstrip("x").split(".")
        return f"blocks.{block_idx}.hook_resid_{layer_idx}"
    elif act_name.startswith("m"):
        block_idx = act_name.lstrip("m")
        return f"blocks.{block_idx}.hook_mlp_out"
    else:
        raise NotImplementedError()


def site_name_to_short_name(probed_act):
    block_idx = int(re.search(r"blocks\.(\d+)\.", probed_act).group(1))
    if "hook_result" in probed_act:
        head_idx = int(re.search(r"hook_result\.(\d+)", probed_act).group(1))
        return f"a{block_idx}.{head_idx}"
    elif "hook_resid" in probed_act:
        layer_idx = re.search(r"hook_resid_([a-z]+)", probed_act).group(1)
        return f"x{block_idx}.{layer_idx}"
    elif "mlp_out" in probed_act:
        return f"m{block_idx}"
    else:
        raise NotImplementedError()