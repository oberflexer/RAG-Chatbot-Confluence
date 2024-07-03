import sys
import load_db
import collections
# from langchain.llms import OpenAI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from getpass import getpass
import torch
#HUGGINGFACEHUB_API_TOKEN = getpass()  # handle sensitive information as it hides the input when typing
HUGGINGFACEHUB_API_TOKEN = "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"


class HelpDesk():
    """Create the necessary objects to create a QARetrieval chain"""
    def __init__(self, new_db=True):

        self.new_db = new_db
        self.template = self.get_template()
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()

        if self.new_db:
            self.db = load_db.DataLoader().set_db(self.embeddings)
        else:
            self.db = load_db.DataLoader().get_db(self.embeddings)

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
                api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2")
        return embeddings

    def get_llm(self):
        # llm = OpenAI()
        # llm = Ollama(model="llama3")
        # Initialize the GPT-2 wrapper
        llm = Ollama(model="llama3")
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
            return f"Voici la source qui pourrait t'être utile :  \n- {distinct_sources_str}"

        elif len(distinct_sources) > 1:
            return f"Voici {len(distinct_sources)} sources qui pourraient t'être utiles :  \n- {distinct_sources_str}"

        else:
            return "Désolé je n'ai trouvé aucune ressource pour répondre à ta question"
