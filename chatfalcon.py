from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os 
import torch

print(torch.cuda.is_available())

model_id = "tiiuae/falcon-7b"
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load Model 
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./workspace/', 
    torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", offload_folder="offload")
# Set PT model to inference mode
model.eval()
# Build HF Transformers pipeline 
pipeline = transformers.pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
# Test out the pipeline
pipeline('who is kim kardashian?')
template = """ Document: {input}

Your task is to extract mail's body from document. 

"""
# Setup prompt template
template = PromptTemplate(input_variables=['input'], template=template) 
# Pass hugging face pipeline to langchain class
llm = HuggingFacePipeline(pipeline=pipeline) 
# Build stacked LLM chain i.e. prompt-formatting + LLM
chain = LLMChain(llm=llm, prompt=template)

# Test LLMChain 

question = """
From:

Dear Enes, Zachary (Zack) M,
Your ticket "38338 AmortCF Clarification" has been Closed.
Regards,
Empyrean Solutions Support Team.

"""
answer = chain.run(question)
print("answer__________")
print(answer)
