# import requests
# from requests.auth import HTTPBasicAuth
# from langchain.document_loaders import ConfluenceLoader

# email = 'mophyhuang@gmail.com'
# api_token = 'ATATT3xFfGF0Eo4-PbBJLN-KnudQbsgOYiNsGz7-JVck1rWg-t46KOkdsYDBVvypOEPwPKJggFUmgzEi3j5yYEBu15kslnt3ErIKDdyYaH7Hhi_m6uBrGRU0QHIk2o28Ua4Ulct5USQp7KSvhKTF88hk7Z0f9MqtyMI8GAxCdU4ydKK71ux4a7M=C0AFB4E8'
# space_key = 'KB'
# base_url = 'https://mophyhuang.atlassian.net/wiki/rest/api/space/'

# response = requests.get(base_url + space_key, auth=HTTPBasicAuth(email, api_token))

# if response.status_code == 200:
#     print("Access verified. Space details:")
#     print(response.json())
# else:
#     print(f"Failed to access space. Status code: {response.status_code}")
#     print(response.json())

# from atlassian import Confluence
# import requests

# # Confluence credentials and instance details
# # base_url = 'https://mophyhuang.atlassian.net/wiki'
# base_url = 'https://mophyhuang.atlassian.net/wiki'
# space_key = 'KB'
# api_token = 'ATATT3xFfGF0Eo4-PbBJLN-KnudQbsgOYiNsGz7-JVck1rWg-t46KOkdsYDBVvypOEPwPKJggFUmgzEi3j5yYEBu15kslnt3ErIKDdyYaH7Hhi_m6uBrGRU0QHIk2o28Ua4Ulct5USQp7KSvhKTF88hk7Z0f9MqtyMI8GAxCdU4ydKK71ux4a7M=C0AFB4E8'
# email = 'mophyhuang@gmail.com'

# # Initialize the Confluence client
# # confluence = Confluence(
# #     url=base_url,
# #     username=email,
# #     password=api_token    
# # )
# loader = ConfluenceLoader(
#             url=base_url,
#             username=email,
#             api_key=api_token,
#             space_key = "KB"

#         )
# try:
#     # Attempt to retrieve all pages from the specified space
#     # pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=10)


#     # Check if pages were successfully retrieved
#     # print(type(pages))
#     # if pages:
#     #     print("Access verified. Number of pages retrieved from space:", len(pages))
#     #     for page in pages:
#     #         print(f"Page ID: {page['id']}, Title: {page['title']}")
#     # else:
#     #     print("No pages found or access might be restricted.")


#     docs = loader.load()      # space_key=space_key,
#             # include_attachments=True,
            
#     print(type(docs))
#     print(docs)

# except requests.exceptions.HTTPError as http_err:
#     print(f"HTTP error occurred: {http_err}")
# except Exception as err:
#     print(f"An error occurred: {err}")


from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.llms import BaseLLM
from langchain.chains import RetrievalQA
# from langchain.retrievers import SomeRetriever  # Replace with your specific retriever
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(name)
model = GPT2LMHeadModel.from_pretrained(name, pad_token_id=tokenizer.eos_token_id)
print(type(model))

from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3"
)  # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3 `

print(llm.invoke("Tell me a joke"))

# class GPT2Wrapper(BaseLLM):
#     def __init__(self, model_name: str = 'gpt2'):
#         super(GPT2Wrapper, self).__init__()
#         self.model = GPT2LMHeadModel.from_pretrained(model_name)
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         self.model.eval()

#     def _llm_type(self) -> str:
#         return "custom_gpt2"

#     def _generate(self, prompt: str, max_tokens: int = 512) -> str:
#         inputs = self.tokenizer(prompt, return_tensors='pt')
#         with torch.no_grad():
#             outputs = self.model.generate(**inputs, max_length=max_tokens)
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response

#     def __call__(self, prompt: str) -> str:
#         return self._generate(prompt)

# # Initialize the GPT-2 wrapper
# llm = GPT2Wrapper()

# # Initialize your retriever (replace SomeRetriever with your actual retriever class)
# # retriever = SomeRetriever()

# # Chain type kwargs (if any)
# chain_type_kwargs = {
#     # Add any specific kwargs needed for your chain type
# }

# # Create the RetrievalQA instance
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs=chain_type_kwargs
# )
