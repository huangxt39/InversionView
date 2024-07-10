from openai import OpenAI
import os
import random
import json
import torch
from typing import Callable
from prompts.prompt4 import messages as prompt

def mark_visible_range(str_tokens, query_pos):
    str_tokens = str_tokens[:query_pos+1]
    del str_tokens[0]
    notation = ["(F1)", "(F2)", "(F3)", "", "(S1)", "(S2)", "(S3)", "", "(A1)", "(A2)", "(A3)", "(A4)"]
    str_tokens = [t+n for t, n in zip(str_tokens, notation)]
    string = " ".join(str_tokens)
    # str_tokens.insert(query_pos+1, ")")
    # string = " ".join(str_tokens)
    # string = string.lstrip("B").rstrip("E")
    # string = "(" + string
    return string

def make_user_msg(act_site: str, query_pos: int, metric: str = "dist", filter_func: Callable = lambda x: True):
    gen_num = 20 # 25
    subset_threshold = 0.1
    if os.path.exists("./cached_generation/.DS_Store"):
        os.remove("./cached_generation/.DS_Store")
    examples = os.listdir("./cached_generation")
    examples = list(filter(filter_func, examples))
    random.shuffle(examples)
    examples = examples[:5]
    user_msg = ""

    # reproduce prompt:
    # act_site = "blocks.0.attn.hook_result.0"
    # query_pos = 8
    # examples = ["B859+137=996", "B308+118=426"] 

    # act_site = "blocks.0.hook_resid_pre"
    # query_pos = 9
    # examples = ["B837+936=1773", "B270+348=618"] 

    # act_site = "blocks.1.attn.hook_result.0"
    # query_pos = 9
    # examples = ["B379+701=1080", "B281+846=1127", "B747+949=1696"]

    for j, example in enumerate(examples):
        # user_msg += f"<example> index: {j} \n"
        user_msg += f'<example index="{j}">\n'
        file_path = os.path.join("./cached_generation", example, f"{act_site}-{query_pos}-Auto-{metric}.json")
        with open(file_path) as f:
            obj = json.load(f)
        assert query_pos == obj["query_position"]
        string = mark_visible_range(obj["query_input"].copy(), query_pos)
        user_msg += ("<q> " + string + " </q>\n")
        
        scores = list(map(lambda x: x[-1], obj["generation"]))
        scores = torch.tensor(scores)

        good_mask = scores <= subset_threshold
        # weight = 1.0
        # for i in range(25):
        #     indices = torch.multinomial(torch.where(good_mask, weight, 1.0), gen_num)
        #     r = good_mask[indices].sum().item() / gen_num
        #     if r < 0.6:
        #         weight *= 1.2
        #     elif r > 0.8:
        #         weight /= 1.1
        #     else:
        #         break
        if good_mask.sum().item() > 0:
            indices = torch.multinomial(torch.where(good_mask, 1.0, 0.0), min(gen_num, good_mask.sum().item()))

            indices = indices.tolist()
            for i in indices:
                str_tokens, prob_diff, metric_value, best_index, best_value = obj["generation"][i]
                string = mark_visible_range(str_tokens.copy(), best_index)
                if best_value <= subset_threshold:
                    user_msg += ("<p> " + string + " </p>\n")
                else:
                    user_msg += ("<n> " + string + " </n>\n")
                    
        user_msg += "</example>\n\n"

    return user_msg

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)        

    client = OpenAI()

    user_msg = make_user_msg("blocks.0.attn.hook_result.3", 8) #, filter_func=lambda x: x[9]=="1")
    print(user_msg)

    messages = prompt.copy()
    messages.append({"role": "user", "content": user_msg})


    completion = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=messages,
    )

    assert completion.choices[0].finish_reason == "stop"
    assistant_msg = completion.choices[0].message.content
    print(assistant_msg)


# <example index="0">
# <q> 7(F1) 1(F2) 1(F3) + 9(S1) 9(S2) 4(S3) = </q>
# <p> 6(F1) 1(F2) 1(F3) + 8(S1) 9(S2) 0(S3) = </p>
# <p> 8(F1) 9(F2) 9(F3) + 8(S1) 1(S2) 7(S3) = </p>
# <p> 2(F1) 1(F2) 8(F3) + 7(S1) 9(S2) 9(S3) = </p>
# <p> 5(F1) 1(F2) 4(F3) + 3(S1) 9(S2) 5(S3) = </p>
# <p> 8(F1) 9(F2) 3(F3) + 8(S1) 1(S2) 0(S3) = </p>
# <p> 6(F1) 1(F2) 8(F3) + 8(S1) 9(S2) 4(S3) = </p>
# <p> 2(F1) 9(F2) 8(F3) + 8(S1) 1(S2) 5(S3) = </p>
# <p> 4(F1) 9(F2) 1(F3) + 9(S1) 1(S2) 4(S3) = </p>
# <p> 2(F1) 1(F2) 8(F3) + 9(S1) 9(S2) 4(S3) = </p>
# <p> 8(F1) 9(F2) 0(F3) + 8(S1) 1(S2) 2(S3) = </p>
# <p> 1(F1) 1(F2) 2(F3) + 6(S1) 9(S2) 6(S3) = </p>
# <p> 5(F1) 1(F2) 6(F3) + 7(S1) 9(S2) 1(S3) = </p>
# <p> 3(F1) 9(F2) 6(F3) + 7(S1) 1(S2) 4(S3) = </p>
# <p> 2(F1) 1(F2) 2(F3) + 7(S1) 9(S2) 0(S3) = </p>
# <p> 9(F1) 1(F2) 9(F3) + 6(S1) 9(S2) 8(S3) = </p>
# <p> 3(F1) 1(F2) 7(F3) + 3(S1) 9(S2) 1(S3) = </p>
# <p> 4(F1) 1(F2) 8(F3) + 9(S1) 9(S2) 4(S3) = </p>
# <p> 4(F1) 1(F2) 5(F3) + 9(S1) 9(S2) 4(S3) = </p>
# <p> 5(F1) 1(F2) 5(F3) + 4(S1) 9(S2) 4(S3) = </p>
# <p> 3(F1) 1(F2) 5(F3) + 7(S1) 9(S2) 0(S3) = </p>
# </example>

# <example index="1">
# <q> 4(F1) 5(F2) 5(F3) + 4(S1) 1(S2) 1(S3) = </q>
# <p> 9(F1) 1(F2) 0(F3) + 1(S1) 1(S2) 1(S3) = </p>
# <p> 9(F1) 5(F2) 1(F3) + 7(S1) 1(S2) 4(S3) = </p>
# <p> 3(F1) 5(F2) 5(F3) + 1(S1) 1(S2) 3(S3) = </p>
# <p> 5(F1) 5(F2) 0(F3) + 3(S1) 1(S2) 1(S3) = </p>
# <p> 3(F1) 5(F2) 0(F3) + 8(S1) 1(S2) 7(S3) = </p>
# <p> 7(F1) 1(F2) 6(F3) + 9(S1) 5(S2) 7(S3) = </p>
# <p> 9(F1) 5(F2) 2(F3) + 6(S1) 1(S2) 4(S3) = </p>
# <p> 9(F1) 1(F2) 0(F3) = 1(S1) 3(S2) 7(S3) = </p>
# <p> 3(F1) 5(F2) 7(F3) + 3(S1) 1(S2) 9(S3) = </p>
# <p> 2(F1) 1(F2) 8(F3) + 8(S1) 5(S2) 9(S3) = </p>
# <p> 2(F1) 5(F2) 8(F3) + 9(S1) 1(S2) 3(S3) = </p>
# <p> 1(F1) 5(F2) 5(F3) + 9(S1) 1(S2) 2(S3) = </p>
# <p> 2(F1) 5(F2) 6(F3) + 3(S1) 1(S2) 9(S3) = </p>
# <p> 3(F1) 5(F2) 6(F3) + 4(S1) 1(S2) 2(S3) = </p>
# <p> 2(F1) 5(F2) 1(F3) + 3(S1) 1(S2) 4(S3) = </p>
# <p> B(F1) 2(F2) 5(F3) + 5(S1) 1(S2) 7(S3) = </p>
# <p> 7(F1) 5(F2) 7(F3) + 9(S1) 1(S2) 7(S3) = </p>
# <p> 6(F1) 5(F2) 7(F3) + 6(S1) 1(S2) 3(S3) = </p>
# <p> 3(F1) 3(F2) 7(F3) + 3(S1) 1(S2) 8(S3) = </p>
# <p> 2(F1) 1(F2) 1(F3) + 4(S1) 5(S2) 9(S3) = </p>
# </example>

# <example index="2">
# <q> 7(F1) 4(F2) 5(F3) + 7(S1) 0(S2) 7(S3) = </q>
# <p> 5(F1) 2(F2) 2(F3) + 7(S1) 0(S2) 2(S3) = </p>
# <p> 3(F1) 2(F2) 3(F3) + 4(S1) 0(S2) 9(S3) = </p>
# <p> 3(F1) 2(F2) 7(F3) + 3(S1) 0(S2) 8(S3) = </p>
# <p> 2(F1) 0(F2) 9(F3) + 8(S1) 3(S2) 7(S3) = </p>
# <p> 5(F1) 0(F2) 3(F3) + 9(S1) 1(S2) 2(S3) = </p>
# <p> 3(F1) 4(F2) 8(F3) + 4(S1) 0(S2) 6(S3) = </p>
# <p> 1(F1) 0(F2) 1(F3) + 7(S1) 3(S2) 0(S3) = </p>
# <p> 5(F1) 0(F2) 9(F3) + 6(S1) 4(S2) 9(S3) = </p>
# <p> 4(F1) 3(F2) 1(F3) + 3(S1) 0(S2) 4(S3) = </p>
# <p> 9(F1) 3(F2) 5(F3) + 6(S1) 0(S2) 3(S3) = </p>
# <p> 3(F1) 4(F2) 7(F3) + 9(S1) 0(S2) 7(S3) = </p>
# <p> 5(F1) 0(F2) 4(F3) + 3(S1) 3(S2) 3(S3) = </p>
# <p> 5(F1) 5(F2) 9(F3) + 1(S1) 0(S2) 3(S3) = </p>
# <p> 4(F1) 3(F2) 3(F3) + 2(S1) 0(S2) 1(S3) = </p>
# <p> 3(F1) 1(F2) 2(F3) + 4(S1) 0(S2) 3(S3) = </p>
# <p> 4(F1) 3(F2) 8(F3) + 1(S1) 0(S2) 5(S3) = </p>
# <p> 6(F1) 5(F2) 1(F3) + 5(S1) 0(S2) 4(S3) = </p>
# <p> 6(F1) 4(F2) 9(F3) + 9(S1) 0(S2) 0(S3) = </p>
# <p> 9(F1) 0(F2) 2(F3) + 6(S1) 0(S2) 7(S3) = </p>
# <p> 5(F1) 0(F2) 8(F3) + 4(S1) 2(S2) 9(S3) = </p>
# </example>

# <example index="3">
# <q> 2(F1) 7(F2) 0(F3) + 3(S1) 4(S2) 8(S3) = </q>
# <p> 8(F1) 4(F2) 0(F3) + 4(S1) 7(S2) 0(S3) = </p>
# <p> 9(F1) 4(F2) 4(F3) + 5(S1) 7(S2) 0(S3) = </p>
# <p> 6(F1) 4(F2) 0(F3) + 7(S1) 7(S2) 3(S3) = </p>
# <p> 1(F1) 4(F2) 9(F3) + 9(S1) 7(S2) 0(S3) = </p>
# <p> 1(F1) 7(F2) 2(F3) + 9(S1) 4(S2) 4(S3) = </p>
# <p> 7(F1) 7(F2) 3(F3) + 2(S1) 4(S2) 8(S3) = </p>
# <p> 2(F1) 4(F2) 1(F3) + 1(S1) 7(S2) 2(S3) = </p>
# <p> 8(F1) 4(F2) 9(F3) + 5(S1) 7(S2) 4(S3) = </p>
# <p> 2(F1) 7(F2) 8(F3) + 2(S1) 4(S2) 1(S3) = </p>
# <p> 7(F1) 7(F2) 1(F3) + 4(S1) 4(S2) 0(S3) = </p>
# <p> 8(F1) 7(F2) 7(F3) + 9(S1) 4(S2) 9(S3) = </p>
# <p> 6(F1) 7(F2) 3(F3) + 3(S1) 4(S2) 4(S3) = </p>
# <p> 1(F1) 4(F2) 9(F3) + 3(S1) 7(S2) 9(S3) = </p>
# <p> 4(F1) 4(F2) 6(F3) + 9(S1) 7(S2) 6(S3) = </p>
# <p> 5(F1) 7(F2) 9(F3) + 7(S1) 4(S2) 4(S3) = </p>
# <p> 6(F1) 7(F2) 7(F3) + 5(S1) 4(S2) 2(S3) = </p>
# <p> 8(F1) 4(F2) 5(F3) + 6(S1) 7(S2) 7(S3) = </p>
# <p> 9(F1) 7(F2) 0(F3) + 6(S1) 4(S2) 4(S3) = </p>
# <p> 4(F1) 4(F2) 2(F3) + 5(S1) 7(S2) 1(S3) = </p>
# <p> 7(F1) 4(F2) 3(F3) + 4(S1) 7(S2) 1(S3) = </p>
# </example>

# <example index="4">
# <q> 9(F1) 1(F2) 5(F3) + 7(S1) 0(S2) 5(S3) = </q>
# <p> 8(F1) 1(F2) 9(F3) + 1(S1) 0(S2) 9(S3) = </p>
# <p> 2(F1) 1(F2) 9(F3) + 4(S1) 0(S2) 7(S3) = </p>
# <p> 3(F1) 3(F2) 4(F3) + 5(S1) 0(S2) 5(S3) = </p>
# <p> 5(F1) 0(F2) 1(F3) + 6(S1) 2(S2) 9(S3) = </p>
# <p> 6(F1) 2(F2) 1(F3) + 3(S1) 0(S2) 1(S3) = </p>
# <p> 2(F1) 1(F2) 4(F3) + 4(S1) 0(S2) 1(S3) = </p>
# <p> 6(F1) 1(F2) 6(F3) + 5(S1) 0(S2) 0(S3) = </p>
# <p> 1(F1) 1(F2) 1(F3) + 7(S1) 0(S2) 9(S3) = </p>
# <p> 3(F1) 0(F2) 8(F3) + 3(S1) 1(S2) 7(S3) = </p>
# <p> 1(F1) 1(F2) 6(F3) + 4(S1) 0(S2) 9(S3) = </p>
# <p> 2(F1) 1(F2) 6(F3) + 3(S1) 0(S2) 4(S3) = </p>
# <p> 9(F1) 3(F2) 4(F3) + 2(S1) 0(S2) 5(S3) = </p>
# <p> 9(F1) 1(F2) 1(F3) + 1(S1) 0(S2) 2(S3) = </p>
# <p> 3(F1) 1(F2) 2(F3) + 1(S1) 0(S2) 0(S3) = </p>
# <p> 9(F1) 1(F2) 1(F3) + 7(S1) 0(S2) 4(S3) = </p>
# <p> 4(F1) 0(F2) 5(F3) + 3(S1) 3(S2) 3(S3) = </p>
# <p> 8(F1) 0(F2) 4(F3) + 5(S1) 1(S2) 5(S3) = </p>
# <p> 4(F1) 1(F2) 4(F3) + 1(S1) 0(S2) 4(S3) = </p>
# <p> 4(F1) 1(F2) 0(F3) + 8(S1) 0(S2) 9(S3) = </p>
# <p> 6(F1) 2(F2) 5(F3) + 7(S1) 0(S2) 5(S3) = </p>
# </example>


# <info index="0"> As for F1 and S1: no mutual information. As for F2 and S2: F2 of the query string ("1") is always the same in generated strings. As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is F2 of the query string. </info>
# <info index="1"> As for F1 and S1: no mutual information. As for F2 and S2: F2 of the query string ("5") is always the same in generated strings. As for F3 and S3: F3 of the query string ("5") is always the same in generated strings. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is F2 and F3 of the query string. </info>
# <info index="2"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("0") is always the same in generated strings. As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is S2 of the query string. </info>
# <info index="3"> As for F1 and S1: no mutual information. As for F2 and S2: F2 of the query string ("7") is always the same in generated strings. As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is F2 of the query string. </info>
# <info index="4"> As for F1 and S1: no mutual information. As for F2 and S2: F2 of the query string ("1") is always the same in generated strings. As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is F2 of the query string. </info>
# <summary> F2 of the query string is consistent across all examples. Additionally, F3 is consistent in one example. </summary>



# <example index="0">
# <q> 7(F1) 1(F2) 1(F3) + 9(S1) 9(S2) 4(S3) = </q>
# <p> 6(F1) 1(F2) 1(F3) + 8(S1) 9(S2) 0(S3) = </p>
# <p> 8(F1) 9(F2) 9(F3) + 8(S1) 1(S2) 7(S3) = </p>
# <p> 2(F1) 1(F2) 8(F3) + 7(S1) 9(S2) 9(S3) = </p>
# <p> 5(F1) 1(F2) 4(F3) + 3(S1) 9(S2) 5(S3) = </p>
# <p> 8(F1) 9(F2) 3(F3) + 8(S1) 1(S2) 0(S3) = </p>
# <p> 6(F1) 1(F2) 8(F3) + 8(S1) 9(S2) 4(S3) = </p>
# <p> 2(F1) 9(F2) 8(F3) + 8(S1) 1(S2) 5(S3) = </p>
# <p> 4(F1) 9(F2) 1(F3) + 9(S1) 1(S2) 4(S3) = </p>
# <p> 2(F1) 1(F2) 8(F3) + 9(S1) 9(S2) 4(S3) = </p>
# <p> 8(F1) 9(F2) 0(F3) + 8(S1) 1(S2) 2(S3) = </p>
# <p> 1(F1) 1(F2) 2(F3) + 6(S1) 9(S2) 6(S3) = </p>
# <p> 5(F1) 1(F2) 6(F3) + 7(S1) 9(S2) 1(S3) = </p>
# <p> 3(F1) 9(F2) 6(F3) + 7(S1) 1(S2) 4(S3) = </p>
# <p> 2(F1) 1(F2) 2(F3) + 7(S1) 9(S2) 0(S3) = </p>
# <p> 9(F1) 1(F2) 9(F3) + 6(S1) 9(S2) 8(S3) = </p>
# <p> 3(F1) 1(F2) 7(F3) + 3(S1) 9(S2) 1(S3) = </p>
# <p> 4(F1) 1(F2) 8(F3) + 9(S1) 9(S2) 4(S3) = </p>
# <p> 4(F1) 1(F2) 5(F3) + 9(S1) 9(S2) 4(S3) = </p>
# <p> 5(F1) 1(F2) 5(F3) + 4(S1) 9(S2) 4(S3) = </p>
# <p> 3(F1) 1(F2) 5(F3) + 7(S1) 9(S2) 0(S3) = </p>
# </example>

# <example index="1">
# <q> 4(F1) 5(F2) 5(F3) + 4(S1) 1(S2) 1(S3) = </q>
# <p> 9(F1) 1(F2) 0(F3) + 1(S1) 1(S2) 1(S3) = </p>
# <p> 9(F1) 5(F2) 1(F3) + 7(S1) 1(S2) 4(S3) = </p>
# <p> 3(F1) 5(F2) 5(F3) + 1(S1) 1(S2) 3(S3) = </p>
# <p> 5(F1) 5(F2) 0(F3) + 3(S1) 1(S2) 1(S3) = </p>
# <p> 3(F1) 5(F2) 0(F3) + 8(S1) 1(S2) 7(S3) = </p>
# <p> 7(F1) 1(F2) 6(F3) + 9(S1) 5(S2) 7(S3) = </p>
# <p> 9(F1) 5(F2) 2(F3) + 6(S1) 1(S2) 4(S3) = </p>
# <p> 9(F1) 1(F2) 0(F3) = 1(S1) 3(S2) 7(S3) = </p>
# <p> 3(F1) 5(F2) 7(F3) + 3(S1) 1(S2) 9(S3) = </p>
# <p> 2(F1) 1(F2) 8(F3) + 8(S1) 5(S2) 9(S3) = </p>
# <p> 2(F1) 5(F2) 8(F3) + 9(S1) 1(S2) 3(S3) = </p>
# <p> 1(F1) 5(F2) 5(F3) + 9(S1) 1(S2) 2(S3) = </p>
# <p> 2(F1) 5(F2) 6(F3) + 3(S1) 1(S2) 9(S3) = </p>
# <p> 3(F1) 5(F2) 6(F3) + 4(S1) 1(S2) 2(S3) = </p>
# <p> 2(F1) 5(F2) 1(F3) + 3(S1) 1(S2) 4(S3) = </p>
# <p> B(F1) 2(F2) 5(F3) + 5(S1) 1(S2) 7(S3) = </p>
# <p> 7(F1) 5(F2) 7(F3) + 9(S1) 1(S2) 7(S3) = </p>
# <p> 6(F1) 5(F2) 7(F3) + 6(S1) 1(S2) 3(S3) = </p>
# <p> 3(F1) 3(F2) 7(F3) + 3(S1) 1(S2) 8(S3) = </p>
# <p> 2(F1) 1(F2) 1(F3) + 4(S1) 5(S2) 9(S3) = </p>
# </example>

# <example index="2">
# <q> 7(F1) 4(F2) 5(F3) + 7(S1) 0(S2) 7(S3) = </q>
# <p> 5(F1) 2(F2) 2(F3) + 7(S1) 0(S2) 2(S3) = </p>
# <p> 3(F1) 2(F2) 3(F3) + 4(S1) 0(S2) 9(S3) = </p>
# <p> 3(F1) 2(F2) 7(F3) + 3(S1) 0(S2) 8(S3) = </p>
# <p> 2(F1) 0(F2) 9(F3) + 8(S1) 3(S2) 7(S3) = </p>
# <p> 5(F1) 0(F2) 3(F3) + 9(S1) 1(S2) 2(S3) = </p>
# <p> 3(F1) 4(F2) 8(F3) + 4(S1) 0(S2) 6(S3) = </p>
# <p> 1(F1) 0(F2) 1(F3) + 7(S1) 3(S2) 0(S3) = </p>
# <p> 5(F1) 0(F2) 9(F3) + 6(S1) 4(S2) 9(S3) = </p>
# <p> 4(F1) 3(F2) 1(F3) + 3(S1) 0(S2) 4(S3) = </p>
# <p> 9(F1) 3(F2) 5(F3) + 6(S1) 0(S2) 3(S3) = </p>
# <p> 3(F1) 4(F2) 7(F3) + 9(S1) 0(S2) 7(S3) = </p>
# <p> 5(F1) 0(F2) 4(F3) + 3(S1) 3(S2) 3(S3) = </p>
# <p> 5(F1) 5(F2) 9(F3) + 1(S1) 0(S2) 3(S3) = </p>
# <p> 4(F1) 3(F2) 3(F3) + 2(S1) 0(S2) 1(S3) = </p>
# <p> 3(F1) 1(F2) 2(F3) + 4(S1) 0(S2) 3(S3) = </p>
# <p> 4(F1) 3(F2) 8(F3) + 1(S1) 0(S2) 5(S3) = </p>
# <p> 6(F1) 5(F2) 1(F3) + 5(S1) 0(S2) 4(S3) = </p>
# <p> 6(F1) 4(F2) 9(F3) + 9(S1) 0(S2) 0(S3) = </p>
# <p> 9(F1) 0(F2) 2(F3) + 6(S1) 0(S2) 7(S3) = </p>
# <p> 5(F1) 0(F2) 8(F3) + 4(S1) 2(S2) 9(S3) = </p>
# </example>

# <example index="3">
# <q> 2(F1) 7(F2) 0(F3) + 3(S1) 4(S2) 8(S3) = </q>
# <p> 8(F1) 4(F2) 0(F3) + 4(S1) 7(S2) 0(S3) = </p>
# <p> 9(F1) 4(F2) 4(F3) + 5(S1) 7(S2) 0(S3) = </p>
# <p> 6(F1) 4(F2) 0(F3) + 7(S1) 7(S2) 3(S3) = </p>
# <p> 1(F1) 4(F2) 9(F3) + 9(S1) 7(S2) 0(S3) = </p>
# <p> 1(F1) 7(F2) 2(F3) + 9(S1) 4(S2) 4(S3) = </p>
# <p> 7(F1) 7(F2) 3(F3) + 2(S1) 4(S2) 8(S3) = </p>
# <p> 2(F1) 4(F2) 1(F3) + 1(S1) 7(S2) 2(S3) = </p>
# <p> 8(F1) 4(F2) 9(F3) + 5(S1) 7(S2) 4(S3) = </p>
# <p> 2(F1) 7(F2) 8(F3) + 2(S1) 4(S2) 1(S3) = </p>
# <p> 7(F1) 7(F2) 1(F3) + 4(S1) 4(S2) 0(S3) = </p>
# <p> 8(F1) 7(F2) 7(F3) + 9(S1) 4(S2) 9(S3) = </p>
# <p> 6(F1) 7(F2) 3(F3) + 3(S1) 4(S2) 4(S3) = </p>
# <p> 1(F1) 4(F2) 9(F3) + 3(S1) 7(S2) 9(S3) = </p>
# <p> 4(F1) 4(F2) 6(F3) + 9(S1) 7(S2) 6(S3) = </p>
# <p> 5(F1) 7(F2) 9(F3) + 7(S1) 4(S2) 4(S3) = </p>
# <p> 6(F1) 7(F2) 7(F3) + 5(S1) 4(S2) 2(S3) = </p>
# <p> 8(F1) 4(F2) 5(F3) + 6(S1) 7(S2) 7(S3) = </p>
# <p> 9(F1) 7(F2) 0(F3) + 6(S1) 4(S2) 4(S3) = </p>
# <p> 4(F1) 4(F2) 2(F3) + 5(S1) 7(S2) 1(S3) = </p>
# <p> 7(F1) 4(F2) 3(F3) + 4(S1) 7(S2) 1(S3) = </p>
# </example>

# <example index="4">
# <q> 9(F1) 1(F2) 5(F3) + 7(S1) 0(S2) 5(S3) = </q>
# <p> 8(F1) 1(F2) 9(F3) + 1(S1) 0(S2) 9(S3) = </p>
# <p> 2(F1) 1(F2) 9(F3) + 4(S1) 0(S2) 7(S3) = </p>
# <p> 3(F1) 3(F2) 4(F3) + 5(S1) 0(S2) 5(S3) = </p>
# <p> 5(F1) 0(F2) 1(F3) + 6(S1) 2(S2) 9(S3) = </p>
# <p> 6(F1) 2(F2) 1(F3) + 3(S1) 0(S2) 1(S3) = </p>
# <p> 2(F1) 1(F2) 4(F3) + 4(S1) 0(S2) 1(S3) = </p>
# <p> 6(F1) 1(F2) 6(F3) + 5(S1) 0(S2) 0(S3) = </p>
# <p> 1(F1) 1(F2) 1(F3) + 7(S1) 0(S2) 9(S3) = </p>
# <p> 3(F1) 0(F2) 8(F3) + 3(S1) 1(S2) 7(S3) = </p>
# <p> 1(F1) 1(F2) 6(F3) + 4(S1) 0(S2) 9(S3) = </p>
# <p> 2(F1) 1(F2) 6(F3) + 3(S1) 0(S2) 4(S3) = </p>
# <p> 9(F1) 3(F2) 4(F3) + 2(S1) 0(S2) 5(S3) = </p>
# <p> 9(F1) 1(F2) 1(F3) + 1(S1) 0(S2) 2(S3) = </p>
# <p> 3(F1) 1(F2) 2(F3) + 1(S1) 0(S2) 0(S3) = </p>
# <p> 9(F1) 1(F2) 1(F3) + 7(S1) 0(S2) 4(S3) = </p>
# <p> 4(F1) 0(F2) 5(F3) + 3(S1) 3(S2) 3(S3) = </p>
# <p> 8(F1) 0(F2) 4(F3) + 5(S1) 1(S2) 5(S3) = </p>
# <p> 4(F1) 1(F2) 4(F3) + 1(S1) 0(S2) 4(S3) = </p>
# <p> 4(F1) 1(F2) 0(F3) + 8(S1) 0(S2) 9(S3) = </p>
# <p> 6(F1) 2(F2) 5(F3) + 7(S1) 0(S2) 5(S3) = </p>
# </example>


# <info index="0"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("9") always shows up in generated strings, either as S2 or F2. As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: S2 of the query string. </info>
# <info index="1"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("1") always shows up in generated strings, either as S2 or F2. As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: S2 of the query string. </info>
# <info index="2"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("0") always shows up in generated strings, either as S2 or F2. As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: S2 of the query string. </info>
# <info index="3"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("4") always shows up in generated strings, either as S2 or F2. As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: S2 of the query string. </info>
# <info index="4"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("0") always shows up in generated strings, either as S2 or F2. As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: S2 of the query string. </info>
# <summary> S2 of the query string is always either F2 or S2 in all generated strings. </summary>