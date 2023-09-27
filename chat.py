import time
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import PromptTemplate, LLMChain
import torch



model_id = "tiiuae/falcon-7b"
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./workspace/', 
    torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", offload_folder="offload")

# from transformers import AutoConfig,AutoModel
# config = AutoConfig.from_pretrained('tiiuae/falcon-7b')
# tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b')
# model =  AutoModel.from_config(config)


pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipeline)


template = """ Document: {document}

Your task is to extract mail's body from document. 

"""

prompt = PromptTemplate(template=template, input_variables=["document"])

local_llm = HuggingFacePipeline(pipeline=pipeline)

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )


question = """
From:

Dear Enes, Zachary (Zack) M,
Your ticket "38338 AmortCF Clarification" has been Closed.
Regards,
Empyrean Solutions Support Team.

"""
answer = llm_chain.run(question)
print("answer__________")
print(answer)

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_minutes = elapsed_time / 60

print(f"Time taken: {str(elapsed_time_minutes)} minutes")