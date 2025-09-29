from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="DSS Backend API")

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY")
DEFAULT_FORMAT = os.getenv("DEFAULT_FORMAT", "json")
DEFAULT_OFFSET = int(os.getenv("DEFAULT_OFFSET", 0))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", 50))

# Catalog ID (constant)
CATALOG_ID = "1f13a8ca-dd4b-4bf6-9d14-f86cff8ae49a"

@app.get("/")
def root():
    return {"message": "DSS Backend is running!"}


@app.get("/catalog")
def get_catalog_data(
    format: str = Query(DEFAULT_FORMAT, description="Output format: json/xml/csv"),
    offset: int = Query(DEFAULT_OFFSET, description="Number of records to skip"),
    limit: int = Query(DEFAULT_LIMIT, description="Maximum number of records to return")
):
    """
    Fetch resource-level data from the catalog endpoint
    """
    url = f"https://ejalshakti.gov.in/api/catalog/{CATALOG_ID}"
    params = {
        "api-key": API_KEY,
        "format": format,
        # "offset": offset,
        # "limit": limit
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        if format.lower() == "json":
            data = response.json()
        else:
            # For xml/csv, return raw text
            data = response.text

        return {"status": "success", "data": data}

    except Exception as e:
        return {"status": "error", "message": str(e)}



# from fastapi import FastAPI, Query
# from fastapi.middleware.cors import CORSMiddleware
# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI(title="DSS Backend for JJM PWS Data")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# JJM_API_KEY = os.getenv("JJM_API_KEY")
# JJM_STATE_CODE = os.getenv("JJM_STATE_CODE", "28")
# DEFAULT_FORMAT = os.getenv("DEFAULT_FORMAT", "json")
# DEFAULT_OFFSET = int(os.getenv("DEFAULT_OFFSET", "0"))
# DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "50"))

# @app.get("/jjm")
# def get_jjm_data(
#     format: str = Query(DEFAULT_FORMAT, description="Output format: json/xml/csv"),
#     offset: int = Query(DEFAULT_OFFSET, description="Number of records to skip"),
#     limit: int = Query(DEFAULT_LIMIT, description="Maximum number of records to return")
# ):
#     """
#     Fetch House Connections by PWS scheme for habitations under JJM
#     Supports optional parameters: format, offset, limit
#     """
#     url = (
#         f"https://ejalshakti.gov.in/api/JJM/houseconnections?"
#         f"state={JJM_STATE_CODE}&apikey={JJM_API_KEY}&format={format}&offset={offset}&limit={limit}"
#     )

#     try:
#         response = requests.get(url)
#         response.raise_for_status()

#         if format.lower() == "json":
#             data = response.json()
#         else:
#             # For xml/csv, just return raw text
#             data = response.text

#         return {"status": "success", "data": data}

#     except Exception as e:
#         return {"status": "error", "message": str(e)}




# import requests
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import os
# from dotenv import load_dotenv

# # Load .env
# load_dotenv()

# JJM_API_KEY = os.getenv("JJM_API_KEY")

# app = FastAPI(title="DSS Backend for JJM PWS Data")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/jjm")
# def get_jjm_data():
#     url = 
#     headers = {
#         "Authorization": f"Bearer {JJM_API_KEY}"
#     }
#     try:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status()
#         data = response.json()
#         # Return first 50 entries for testing
#         return {"status": "success", "data": data.get("records", [])[:50]}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}



# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import requests
# import os
# from dotenv import load_dotenv

# # Load .env variables
# load_dotenv()

# app = FastAPI(title="DSS Backend for JJM PWS Data")

# # Access the environment variables
# JJM_API_KEY = os.getenv("JJM_API_KEY")
# JJM_STATE_CODE = os.getenv("JJM_STATE_CODE", "28")  # fallback if not set

# # CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



# # DSS_backend.py
# """
# Minimal FastAPI backend to fetch Jal Jeevan Mission (JJM) PWS House Connections data.
# This serves as a test to check the connection between your DSS frontend and the JJM API.
# """

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import requests
# import os
# import dotenv

# app = FastAPI(title="DSS Backend for JJM PWS Data")

# # Allow frontend to access this backend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # For testing only. Replace '*' with your frontend domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load API key from environment variable or fallback to placeholder
# JJM_API_KEY = os.getenv("JJM_API_KEY", "YOUR_API_KEY_HERE")
# JJM_STATE_CODE = "28"  # Example: Madhya Pradesh, change as needed

# @app.get("/jjm")
# def get_jjm_data():
#     """
#     Fetch House Connections by PWS scheme for habitations under JJM
#     and return the first 50 records for testing.
#     """
#     # url = f"https://ejalshakti.gov.in/api/JJM/houseconnections?state={JJM_STATE_CODE}&apikey={JJM_API_KEY}"
#     url = f"https://ejalshakti.gov.in/api/JJM/houseconnections?apikey={JJM_API_KEY}"

#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()
#         # Return first 50 entries for testing
#         return {"status": "success", "data": data[:50]}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
