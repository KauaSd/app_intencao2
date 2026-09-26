PY := .venv/Scripts/python.exe

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m spacy download pt_core_news_sm
	$(PY) -m spacy download en_core_web_sm

test:
	$(PY) -m pytest

serve:
	$(PY) app.py

demo:
	$(PY) -m intent.cli demo

rules-validate:
	$(PY) -m intent.cli rules validate src/intent/rules/pt.json
	$(PY) -m intent.cli rules validate src/intent/rules/en.json
