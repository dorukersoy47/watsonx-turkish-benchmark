''' 
    STEP 1: pip install unitxt ibm_watsonx_ai pandas tqdm

'''

import os
from dotenv import load_dotenv

load_dotenv()

# Access environment variables from .env file
PROJECT_ID = os.getenv('PROJECT_ID')
PROJECT_URL = os.getenv('PROJECT_URL')
API_KEY = os.getenv('API_KEY')


