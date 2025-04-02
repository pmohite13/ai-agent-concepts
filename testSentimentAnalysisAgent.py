from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.to(device)


def analyze_sentiment(sentences):
    results = []
    for sentence in sentences:
        inputs = tokenizer(sentences, return_tensors="pt",
                           max_length=128, truncation=True).to(device)

        outputs = model(** inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        sentiment = "Positive" if torch.argmax(
            probabilities) == 1 else "Negative"

        results.append({"Sentence: ": sentence, "Sentiment": sentiment})

    return results


class SentimentAnalysisAgent:
    def __init__(self, data_source, interval=10):
        self.data_source = data_source
        self.interval = interval

    def run(self):
        print("Starting sentiment analysis agent:")
        while True:
            sentences = self.data_source()
            if not sentences:
                print(f"[{datetime.now()}] No new data to analyse, waiting...")
            else:
                results = analyze_sentiment(sentences)
                for result in results:
                    print(
                        f"Sentence: {result['sentence']} \nSentiment: {result['sentiment']}")
            time.sleep(self.interval)


def fetch_sentences():
    import random
    sample_sentences = ["The movie was fantastic and highly engaging",
                        "I did not enjoy the food at all",
                        "This product exceeds my expectations!",
                        "The customer service was terrible and unhelpful",
                        "What a wonderful experience at the resort!"]
    return random.sample(sample_sentences, random.randint(0, len(sample_sentences)))


if __name__ == "__main__":
    agent = SentimentAnalysisAgent(fetch_sentences, interval=10)
    agent.run()
