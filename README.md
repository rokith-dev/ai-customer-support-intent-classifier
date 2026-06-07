# AI Customer Support Intent Classifier

This project scaffolds a customer support intent classification workflow with:
- data ingestion and preprocessing
- TF-IDF feature generation
- baseline model training and evaluation
- notebook-based exploration
- a Streamlit demo app

## Project Structure

- `data/raw/`: raw ticket data
- `data/processed/`: cleaned datasets
- `notebooks/`: exploration and experimentation notebooks
- `src/`: reusable Python modules
- `models/`: serialized model artifacts
- `app/`: Streamlit application
- `tests/`: unit tests

## Next Steps

1. Add the real ticket dataset to `data/raw/`.
2. Run preprocessing to generate `data/processed/cleaned_tickets.csv`.
3. Train the baseline model and save artifacts in `models/`.
