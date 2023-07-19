----
QualAbstracts
----
Summarizing qualitative social science and humanities research

Models to consider:
- A small option: [Flan t5-small](https://huggingface.co/google/flan-t5-small). See [here](https://www.databricks.com/blog/2023/03/20/fine-tuning-large-language-models-hugging-face-and-deepspeed.html) for example fine-tuning this model (specifically its predecessor).
- A larger 3B parameter model from `stabilityai`: [stablelm](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b)
- Largest, designed to take in large input: [MPT-7B-8K](https://huggingface.co/mosaicml/mpt-7b-8k). NB: tried to do a simple example generation with the `stablelm` 7B model in Colab and it didn't have enough GPU memory.

Hungging Face summarization example: [](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)
