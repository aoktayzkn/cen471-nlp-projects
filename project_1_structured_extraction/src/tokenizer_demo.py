from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def show(text):
    ids = tok.encode(text, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    print(f"{text!r}: {len(ids)} tokens, {len(text)} chars")
    print("   ", pieces)

show("kitaplarımızdan")          # the word asked in the question
show("from our books")           # English equivalent
print()
show("The movie was great, but the ending was disappointing.")
show("Film harikaydı ama sonu hayal kırıcıydı.")