from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id  = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_id )
model = AutoModelForCausalLM.from_pretrained(model_id)

hf_pipeline = pipeline("text-generation", model=model,
                       tokenizer=tokenizer, device=0, max_length=500, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

llm = HuggingFacePipeline(pipeline=hf_pipeline)


prompt = PromptTemplate(input_variables="topic",
                        template="Provide a brief overview of {topic}. Give some example of what it is useful for. Explain where it is used.")


pipeline = prompt | llm

topic = "Large Language Models"
result = pipeline.invoke({"topic": topic})

print(f"Result for topifc '{topic}': \n {result}")
