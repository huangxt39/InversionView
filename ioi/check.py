from ioi_dataset import IOIDataset, NAMES
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

for n in NAMES:
    assert "Ġ"+n in tokenizer.vocab, n
print("pass")

template = "Then, [B] and [A] went to the supermarket. [B] gave a drink to [A]"
for n in NAMES:
    text = template.replace("[A]", n).replace("[B]", "Christopher")
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    # print(tokenizer.convert_ids_to_tokens(ids))
    print(len(ids))
    assert len(ids) == 16, f"!!!!{tokenizer.convert_ids_to_tokens(ids)}"
print("pass")