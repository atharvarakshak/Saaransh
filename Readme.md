streamlit run app/app.py

pip install 

## database

docker run -p 6333:6333 qdrant/qdrant
http://localhost:6333/dashboard

## Setup

1. Install dependencies (CPU-only):
   
   pip install -r requirements.txt

2. Configure Gemini API key:
   
   cp env.example .env
   # edit .env and set GOOGLE_API_KEY

3. Run the app:
   
   streamlit run app/app.py