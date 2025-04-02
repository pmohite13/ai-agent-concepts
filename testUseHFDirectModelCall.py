from transformers import AutoTokenizer, AutoModel

inputText = "Hello World"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer(inputText, return_tensors="pt")

outputs = model(**inputs) #Best for model fine-tuning, research, or developing custom pipelines.

print(outputs.last_hidden_state.shape)