# Imports
# Env var
import os
import sys
from dotenv import load_dotenv, find_dotenv

# Env variables
sys.path.append('../..')
_ = load_dotenv(find_dotenv())

OPEN_AI_API_KEY = os.environ['OPENAI_API_KEY']

CONFLUENCE_SPACE_NAME = os.environ['CONFLUENCE_SPACE_NAME']  # Change to your space name
CONFLUENCE_API_KEY = os.environ['CONFLUENCE_API_KEY']
# https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/
# my confluence private api_key: ATATT3xFfGF0Eo4-PbBJLN-KnudQbsgOYiNsGz7-JVck1rWg-t46KOkdsYDBVvypOEPwPKJggFUmgzEi3j5yYEBu15kslnt3ErIKDdyYaH7Hhi_m6uBrGRU0QHIk2o28Ua4Ulct5USQp7KSvhKTF88hk7Z0f9MqtyMI8GAxCdU4ydKK71ux4a7M=C0AFB4E8
CONFLUENCE_SPACE_KEY = os.environ['CONFLUENCE_SPACE_KEY']
# Hint: space_key and page_id can both be found in the URL of a page in Confluence
# https://yoursite.atlassian.com/wiki/spaces/<space_key>/pages/<page_id>
#eg. https://mophyhuang.atlassian.net/wiki/spaces/KB/pages/196718/Template+-+How-to+guide
CONFLUENCE_USERNAME = os.environ['CONFLUENCE_USERNAME']
PATH_NAME_SPLITTER = './splitted_docs.jsonl'
PERSIST_DIRECTORY = '../db/chroma/'
EVALUATION_DATASET = '../data/evaluation_dataset.tsv'
