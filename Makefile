install:
	pip install -r requirements.txt


run:
	streamlit run app/app.py


init-db:
	python3 app/collections.py


