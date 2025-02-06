For that we use Molformer's tokenizer to seg SMILES, please download the weights and related files for Molformer and put them into this dir. Molformer can be downloaded from [Molformer Hugging Face](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/tree/main).

These files are required:

1. config.json
2. configuration_molformer.py
3. convert_molformer_original_checkpoint_to_pytorch.py
4. modeling molformer.py
5. pytorch_model.bin
6. special_tokens_map.json
7. tokenization_molformer.py
8. tokenization_molformer_fast.py
9. tokenizer.json
10. tokenizer_config.json
11. vocab.json