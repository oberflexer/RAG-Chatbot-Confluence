import sys

from dotenv import load_dotenv

import os

from langchain_community.llms.huggingface_hub import HuggingFaceHub

load_dotenv()

from openai import api_version, azure_endpoint

import load_db
import collections
# from langchain.llms import OpenAI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain_openai import AzureChatOpenAI, OpenAI

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from getpass import getpass
import torch




class HelpDesk():
    """Create the necessary objects to create a QARetrieval chain"""
    def __init__(self):

        self.template = self.get_template()
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()

        """initialises db"""
        self.db = load_db.DataLoader(embeddings=self.embeddings).get_db(embeddings=self.embeddings)

        """Retriever for retrieving relevant information to the users questions"""
        self.retriever = self.db.as_retriever()
        self.retrieval_qa_chain = self.get_retrieval_qa()


    def get_template(self):
        template = """
        Given this text extracts:
        -----
        {context}
        -----
        Please answer with to the following question:
        Question: {question}
        Helpful Answer:
        """
        return template

    def get_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        return prompt

    # def get_embeddings(self) -> OpenAIEmbeddings:
    def get_embeddings(self) -> HuggingFaceInferenceAPIEmbeddings:
        # embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"), model_name="sentence-transformers/all-MiniLM-l6-v2")
        return embeddings

    def get_llm(self):

        #getting the LLM Data from .env
        endpoint = os.getenv("OPENAI_ENDPOINT")
        api_key = os.getenv("OPENAI_APIKEY")
        api_version = os.getenv("API_VERSION")
        model_name = os.getenv("MODEL")
        provider = os.getenv("LLM_PROVIDER")

        if provider == "azure":
            llm = AzureChatOpenAI(
                openai_api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
                model_name=model_name
            )
        elif provider == "openai":
            llm = OpenAI(
                openai_api_key=api_key,
                model=model_name
            )
        elif provider == "huggingface":
            llm = HuggingFaceHub(
                model_name=model_name,
                api_key=api_key
            )
        else:
            raise ValueError("Unsupported LLM provider. Please check your environment configuration. Allowed: azure, openai or huggingface!")

        return llm

    def get_retrieval_qa(self):
        chain_type_kwargs = {"prompt": self.prompt}
        print(f"LLM instance: {self.llm}, Type: {type(self.llm)}")  # Debugging statement
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        return qa

    def retrieval_qa_inference(self, question, verbose=True):
        query = {"query": question}
        answer = self.retrieval_qa_chain(query)
        sources = self.list_top_k_sources(answer, k=2)

        if verbose:
            print(sources)

        return answer["result"], sources

    def list_top_k_sources(self, answer, k=2):
        sources = [
            f'[{res.metadata["title"]}]({res.metadata["source"]})'
            for res in answer["source_documents"]
        ]

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

        if len(distinct_sources) == 1:
            return f"Here is the source that could be useful to you :  \n- {distinct_sources_str}"

        elif len(distinct_sources) > 1:
            return f"Here are {len(distinct_sources)} sources that could be useful to you :  \n- {distinct_sources_str}"

        else:
            return "Sorry, I couldn't find any resources to answer your question."
