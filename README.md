**Help desk** allows you to create a Question Answering bot with a streamlit UI using your company Confluence data.

<p align="center">
  <img src="./docs/chatbot_with_confluence.jpg" alt="Chatbot with Confluence"/>
</p>

# How to use

- python 3.10.x needed
- Create a virtual environnement:
    - `python -m venv .venv` || `python3 -m venv .venv`
    -  Linux: `source .venv/bin/activate` || Windows: `.venv\Scripts\Activate`
    -  Linux: `pip freeze | xargs pip uninstall -y` || Windows: `pip freeze | ForEach-Object {pip uninstall -y $_}`
    -  `pip install -r requirements.txt`

- Copy the env.template and fill your environment variables
     - `cp .env.template .env`

- Check the `config.py` and `env.template` file.
- To collect data from Confluence you will have to:
  - Create your own Conluence space with page informations
  - Create and feed your API key [here]('https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/')
  - Insert in the  `env` file:
    -  the confluence_api_key
    -  the space_key: `https://yoursite.atlassian.com/wiki/spaces/<space_key>/pages/`
    -  the space_name: `<space_name>/spaces/<space_key>/pages/`
    -  the email adress you used for your Confluence space
    -  the open_apikey
    -  optionally the model of your llm_provider
    -  the openai_endpoint
    -  the huggingfacehub_api_token
    -  the api_version
    -  the llm_provider

- Finally run the app
  - `cd src`
  - `streamlit run GUI_with_Streamlit.py`

- To use and deep dive with the notebook
```
ipython kernel install --name RAG --user  # Add the notebook kernel
jupyter lab
```

## How it works ?


    .
    ├── data/
        ├── evaluation_dataset.tsv  # Questions and answers useful for evaluation

    ├── docs/                       # Documentation files
    ├── src/                        # The main directory for computer demo
        ├── __init__.py
        ├── load_db.py              # Load data from confluence and creates smart chunks
        ├── help_desk.py            # Instantiates the LLMs, retriever and chain
        ├── main.py                 # Run the Chatbot for a simple question
        ├── streamlit.py            # Run the Chatbot in streamlit where you can ask your own questions
        ├── evaluate.py             # Evaluate the RAG model based on questions-answers samples

    ├── notebooks/                  # Interactive code, useful for try and learn
    ├── config.py
    ├── .env.template               # Environment variables to feed
    ├── .gitignore
    ├── LICENSE                     # MIT License
    ├── README.md                   # Where to start
    └── requirements.txt            # The dependencies


The process is the following:
- Loading data from Confluence
  - You can keep the Markdown style using the `keep_markdown_format` option added in our [MR]('https://github.com/langchain-ai/langchain/pull/8246')
  - See the `help_desk.ipynb` for a more deep dive analysis
  - Otherwise you cannot split text in a smart manner using the [MarkdownHeaderTextSplitter]('https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/markdown_header_metadata')
- Load data
- Markdown and RecursiveCharacterTextSplitter
- LLM used: Llam3 local
- The QARetrievalChain
- Streamlit as a data interface