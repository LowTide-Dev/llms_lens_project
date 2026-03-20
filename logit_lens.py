from transformer_lens import HookedTransformer

model: HookedTransformer = HookedTransformer.from_pretrained("pythia-14M")

logits, cache = model.run_with_cache("The sky is blue and the grass is green.")

print(logits)
print(logits.shape)

print(cache.keys())
for k, v in cache.items():
    print(k, v.shape)
