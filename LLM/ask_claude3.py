import anthropic
from ask_gpt4 import make_user_msg
from prompts.prompt4 import messages as prompt
import random
import torch
import json
from key import claude_key


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    client = anthropic.Anthropic(api_key=claude_key)

    system_msg = prompt.pop(0)["content"]

    result_key = "x1.post, A4/E" #, A1!=1"
    user_msg = make_user_msg("blocks.1.hook_resid_post", 11) #, filter_func=lambda x: x[9]!="1")
    print(user_msg)

    # confirm = input("confirm:(y)/n")
    # if len(confirm) != 0 and confirm != "y":
    #     exit()

    messages = prompt.copy()
    messages.append({"role": "user", "content": user_msg})

    # claude-3-opus-20240229
    # claude-3-sonnet-20240229
    # claude-3-haiku-20240307
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=2048,
        system=system_msg,
        # temperature=0.5,
        messages=messages
    )
    model_msg = message.content[0].text

    print(model_msg)

    with open("./claude3_ans.json", "r") as f:
        existing_results = json.load(f)
    existing_results[result_key] = {"user_msg": user_msg, "model_msg": model_msg}
    with open("./claude3_ans.json", "w") as f:
        json.dump(existing_results, f)


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


# Here are the key observations for each example:

# <info index="0">
# As for F2 and S2: S2 of the query string ("9") appears as either F2 or S2 in all generated strings. The other digit in the same place (F2 or S2) is always "1", which is close to F2 of the query string ("1"). 
# As for F3 and S3: F3 of the query string ("1") appears in most generated strings as either F3 or S3. The other digit is often "4", which is close to S3 of the query string ("4").
# No significant patterns found for F1/S1 or other aspects.
# Therefore, the main common information is the appearance of S2 and approximate F2 in generated strings (order can vary), as well as the appearance of F3 and approximate S3 in generated strings (order can vary).
# </info>

# <info index="1">
# As for F2 and S2: S2 of the query string ("1") appears as either F2 or S2 in nearly all generated strings. 
# As for F3 and S3: No significant patterns.
# No significant patterns found for F1/S1 or other aspects. 
# Therefore, the main common information is the appearance of S2 of the query string as either F2 or S2 in the generated strings.
# </info>

# <info index="2">
# As for F2 and S2: S2 of the query string ("0") appears as S2 in all generated strings.
# As for F3 and S3: No significant patterns.
# No significant patterns found for F1/S1 or other aspects.
# Therefore, the main common information is S2 of the query string appears as S2 in all generated strings.
# </info>

# <info index="3">
# As for F2 and S2: F2 of the query string ("7") appears in most generated strings as either F2 or S2. The other digit is often "4", which is the same as S2 of the query string.
# As for F3 and S3: No significant patterns.
# No significant patterns found for F1/S1 or other aspects.
# Therefore, the main common information is the appearance of F2 of the query string and S2 of the query string as F2 and S2 in generated strings (order can vary).
# </info>

# <info index="4">
# As for F2 and S2: S2 of the query string ("0") appears as S2 in nearly all generated strings.
# As for F3 and S3: No significant patterns.
# No significant patterns found for F1/S1 or other aspects.
# Therefore, the main common information is S2 of the query string appears as S2 in nearly all generated strings.
# </info>

# <summary>
# The most commonly observed pattern is that S2 of the query string often appears as S2 (or sometimes F2) in the generated strings. 
# Less frequently, F2 of the query string may appear alongside S2 as the F2 and S2 (order can vary) of generated strings.
# Occasionally, F3 and S3 of the query may approximately appear as F3 and S3 (order can vary) in generated strings, but this is not consistent.
# No significant patterns were detected for F1/S1 or the answer part of the equations.
# </summary>




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


# Here are the common patterns I found in each example:

# <info index="0">
# As for F1 and S1: no mutual information.
# As for F2 and S2: In the query string, F2 is 1, and S2 is 9. In all generated strings, either F2 is 1 and S2 is 9, or F2 is 9 and S2 is 1. So (F2, S2) from the query, (1, 9), always appears in generated strings, though the order may be swapped.
# As for F3 and S3: no mutual information.
# As for any other common pattern: not found.
# Therefore, the information that query and all generated strings have in common is: F2 and S2 (their order can vary).
# </info>

# <info index="1">
# As for F1 and S1: no mutual information.
# As for F2 and S2: In the query string, F2 is 5, and S2 is 1. In most generated strings, either F2 is 5 and S2 is 1, or F2 is 1 and S2 is 5. So (F2, S2) from the query, (5, 1), often appears in generated strings, though the order may be swapped.
# As for F3 and S3: no mutual information.
# As for any other common pattern: not found.
# Therefore, the information that query and most generated strings have in common is: F2 and S2 (their order can vary).
# </info>

# <info index="2">
# As for F1 and S1: no mutual information.
# As for F2 and S2: In the query string, F2 is 4, and S2 is 0. In most generated strings, either F2 is 4 (or close to 4) and S2 is 0, or F2 is 0 and S2 is 4 (or close to 4). So an approximate version of (F2, S2) from the query, (4, 0), often appears in generated strings, though the order may be swapped.  
# As for F3 and S3: no mutual information.
# As for any other common pattern: not found.
# Therefore, the information that query and most generated strings have in common is: approximate values of F2 and S2 (their order can vary).
# </info>

# <info index="3">
# As for F1 and S1: no mutual information.
# As for F2 and S2: In the query string, F2 is 7, and S2 is 4. In all generated strings, either F2 is 7 and S2 is 4, or F2 is 4 and S2 is 7. So (F2, S2) from the query, (7, 4), always appears in generated strings, though the order may be swapped.
# As for F3 and S3: no mutual information. 
# As for any other common pattern: not found.
# Therefore, the information that query and all generated strings have in common is: F2 and S2 (their order can vary).
# </info>

# <info index="4">
# As for F1 and S1: no mutual information.
# As for F2 and S2: In the query string, F2 is 1, and S2 is 0. In most generated strings, either F2 is 1 and S2 is 0, or F2 is 0 and S2 is 1. So (F2, S2) from the query, (1, 0), often appears in generated strings, though the order may be swapped.
# As for F3 and S3: no mutual information.
# As for any other common pattern: not found.
# Therefore, the information that query and most generated strings have in common is: F2 and S2 (their order can vary).
# </info>

# <summary>
# Across the examples, the most consistent pattern is that F2 and S2 values from the query string tend to appear in the generated strings, although sometimes their order is swapped and sometimes the match is approximate rather than exact. Other digit positions like F1, S1, F3, S3 do not seem to have mutual information between the query and generated strings.
# </summary>


# <info index="0"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("9") always appears as one of F2 or S2 in generated strings. Also in generated strings, the other digit of the same place (F2 and S2) is either "8" or "1", both are close to F2 of the query string ("1"). As for F3 and S3: F3 of the query string ("1") always appears in the place of F3 and S3 (can be either of them) of generated strings. The other digit in the same value place sometimes approximates S3 of the query string ("4"). As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: F2 and S2 (fuzzy representation, their order can vary); F3</info>
# <info index="1"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("1") always appears in the place of F2 and S2 (can be either of them) of generated strings. The other digit in the same value place sometimes approximates F2 of the query string ("5"). As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: S2</info>
# <info index="2"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("0") always appears as one of F2 or S2 in generated strings. Also in generated strings, the other digit of the same place (F2 and S2) is often "4", "3", or "5", all are close to F2 of the query string ("4"). As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: F2 and S2 (fuzzy representation, their order can vary). </info>
# <info index="3"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("4") always appears in the place of F2 and S2 (can be either of them) of generated strings. The other digit in the same value place often approximates F2 of the query string ("7"). As for F3 and S3: no mutual information. As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: F2 and S2 (fuzzy representation, their order can vary). </info>
# <info index="4"> As for F1 and S1: no mutual information. As for F2 and S2: S2 of the query string ("0") always appears as one of F2 or S2 in generated strings. Also in generated strings, the other digit of the same place (F2 and S2) is often "1", "2", or "3", all are close to F2 of the query string ("1"). As for F3 and S3: F3 of the query string ("5") always appears in the place of F3 and S3 (can be either of them) of generated strings. The other digit in the same value place sometimes approximates S3 of the query string ("5"). As for any other common pattern: not found. Therefore, the information that query and all generated strings have in common is: F2 and S2 (fuzzy representation, their order can vary); F3</info>
# <summary>
# F2 and S2 (fuzzy, interchangeable); F3 (sometimes fuzzy); S2 (sometimes fuzzy)
# </summary>