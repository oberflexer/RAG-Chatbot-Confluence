import sys, os
import logging
import shutil

# Using os.path for cross-platform compatibility
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
from config import (CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY,
                    CONFLUENCE_USERNAME, CONFLUENCE_API_KEY, PERSIST_DIRECTORY)

from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma

class DataLoader():
    """Create, load, save the DB using the Confluence Loader"""

    def __init__(
            self,
            embeddings,
            confluence_url=CONFLUENCE_SPACE_NAME,
            username=CONFLUENCE_USERNAME,
            api_key=CONFLUENCE_API_KEY,
            space_key=CONFLUENCE_SPACE_KEY,
            persist_directory=PERSIST_DIRECTORY,
            new_db=True
    ):
        self.confluence_url = confluence_url
        self.username = username
        self.api_key = api_key
        self.space_key = space_key
        self.persist_directory = persist_directory
        self.embeddings = embeddings

        # Initialize Chroma DB instance based on `new_db` flag
        if new_db:
            self.db = self.set_db(self.embeddings)
        else:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )


    def set_db(self, embeddings):
        """Create, save, and load db"""
        try:
            shutil.rmtree(self.persist_directory)
        except Exception as e:
            logging.warning("%s", e)

        # Load docs
        docs = self.load_from_confluence_loader()

        # Split Docs
        splitted_docs = self.split_docs(docs)

        # Save to DB
        db = self.save_to_db(splitted_docs, embeddings)

        return db


    def load_from_confluence_loader(self):
        """Load HTML files from Confluence"""
        loader = ConfluenceLoader(
            url=self.confluence_url,
            username=self.username,
            api_key=self.api_key,
            space_key=self.space_key
        )
        docs = loader.load()
        print(f"Number of documents loaded from Confluence: {len(docs)}")
        return docs



    def split_docs(self, docs):
        """Split documents with metadata preservation for enhanced granularity"""
        # Define headers for markdown splitting
        headers_to_split_on = [
            ("#", "Title 1"),
            ("##", "Subtitle 1"),
            ("###", "Subtitle 2"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Split based on markdown and add original metadata
        md_docs = []
        for doc in docs:
            md_doc = markdown_splitter.split_text(doc.page_content)
            for i in range(len(md_doc)):
                md_doc[i].metadata = md_doc[i].metadata | doc.metadata
            md_docs.extend(md_doc)

        # Recursive text splitter for further splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
        )

        # Split each markdown-based chunk into smaller text chunks if needed
        splitted_docs = splitter.split_documents(md_docs)
        print(f"Number of splitted documents loaded from Confluence: {len(splitted_docs)}")
        return splitted_docs


    def save_to_db(self, splitted_docs, embeddings):
        """Save chunks to Chroma DB"""
        from langchain_community.vectorstores import Chroma
        # Annahme: self.embeddings ist eine Funktion oder Methode zur Erstellung von Embeddings

        db = Chroma.from_documents(splitted_docs, embeddings, persist_directory=self.persist_directory)
        db.persist()
        return db

    def get_db(self, embeddings):
        """Create, save, and load db"""
        db = self.load_from_db(embeddings)
        return db

    def load_from_db(self, embeddings):
        """Loader chunks to Chroma DB"""
        from langchain.vectorstores import Chroma
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        return db


if __name__ == "__main__":
    # Example usage, assuming `embeddings` is defined elsewhere in your code
    embeddings = ...  # Provide your embeddings instance
    loader = DataLoader(embeddings, new_db=True)  # Set new_db=False if you want to load an existing DB