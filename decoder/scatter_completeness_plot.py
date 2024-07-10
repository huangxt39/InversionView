import torch
import matplotlib.pyplot as plt
import re

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

save_path = "data_and_model/scatter_data_new_noise1.pt"


data_obj = torch.load(save_path)
print(len(data_obj))
data_obj = data_obj[-8:]

fig, ax = plt.subplots(2, 4, figsize=(12, 4))

for i in range(2):
    for j in range(4):
        obj = data_obj[i*4+j]
        ax[i,j].scatter(obj["dist"].numpy(), obj["prob"].numpy(), s=3)
        ax[i,j].axvline(x=0.1, color="r", linewidth=0.75)
        ax[i,j].tick_params(labelsize=8)
        ax[i,j].set_ylim(top=0)
        ax[i,j].set_title("$" + site_name_to_latex_name(obj["probed_act"]) + "$" + "  " + obj["query_input"], fontsize=10)


fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Distance", fontsize=16)
plt.ylabel("log Probability", fontsize=16)
plt.tight_layout()
# plt.show()
fig.savefig("data_and_model/completeness_new_noise1.png", dpi=400)