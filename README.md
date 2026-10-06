# MUJ-DS-23FE10CDS00443

## Student & Project Information

| Field | Details |
|---|---|
| **Name** | Sahil |
| **Registration Number** | 23FE10CDS00443 |
| **Branch** | B.Tech Computer Science Engineering (Data Science) |
| **Batch** | 2023-2027 (Batch F) |
| **Project Title** | Customer Feedback Analyzer (LLM-powered NLP) |
| **GitHub Username** | sahilsinghh1273 |
| **Training Program** | Machine Learning & Data Science Training, Manipal University Jaipur, 2023–2027 |

---

## 📌 Project Overview
The **Customer Feedback Analyzer** is an end-to-end NLP system powered by Large Language Models (xAI Grok via OpenAI-compatible endpoint). It automates the parsing, multi-class classification, urgency scoring, entity recognition, and response suggestion for incoming customer reviews and support tickets. The application includes both a robust Command Line Interface (CLI) and an interactive Streamlit analytics dashboard supporting both real-time API inference and an offline demo mode with zero secret dependencies.

---

## 📂 Repository Structure

| Directory / File | Description |
|---|---|
| `README.md` | Primary repository documentation with student and project details |
| `assignments/` | Training coursework and academic assignments |
| `notebooks/` | Jupyter notebooks for exploratory data analysis and prototyping |
| `code/` | Source code for applications; contains `nlp-feedback-analyzer` |
| `resources/` | Reference datasets, schemas, and external documentation |
| `presentations/` | Project demonstration slides and visual presentation material |
| `capstone/` | Final capstone documentation and collaborative team artifacts |

---

## 🚀 NLP Project Link & Quickstart

The full project code, test suite, and configuration reside in:
👉 **[`code/nlp-feedback-analyzer`](code/nlp-feedback-analyzer/README.md)**

### Quickstart Commands
```bash
# Navigate to the NLP project
cd code/nlp-feedback-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the test suite (100% offline)
pytest -q

# Launch the Streamlit dashboard in demo mode
streamlit run app.py

# Run CLI analysis
python main.py analyze "The app crashes whenever I tap the checkout button."
python main.py batch --input data/sample_feedback.csv
python main.py evaluate --input data/labeled_eval.csv
```
