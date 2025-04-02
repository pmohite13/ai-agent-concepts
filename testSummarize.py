from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


def summarize(text):
    inputs = tokenizer.encode("summarize: focus on key impacts and industries: " +
                              text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=40, min_length=10,
                             length_penalty=3.5, num_beams=5, early_stopping=True)

    summary = tokenizer.decode(outputs, skip_special_tokens=True)

    unique_sentences = list(dict.fromkeys(summary.split(". ")))
    return ". ".join(unique_sentences)


if __name__ == '__main__':
    sample_text = (
        "Artificial Intelligence is the rapidly growing field that involves the creation of"
        "intelligent machines capable of performing tasks that typically requires human intelligence"
        "It is used in various industries including healthcare, finance and transporation"
        "to improve efficiency and solve complex problems"
    )
    
    print(summarize(sample_text))
