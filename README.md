### basic start but tried prod
# Housing Price Prediction

This project demonstrates an end-to-end **Machine Learning pipeline** for predicting median house values using the **Housing Dataset**.  
It covers data preprocessing, model training with multiple algorithms, and final model persistence and inference using **Joblib**.

---

## Project Structure (Once app.py runs)

```bash
├── housing.csv            # Dataset file
├── main.py                # Model training and evaluation script
├── app.py                 # Final pipeline, model persistence, and inference
├── model.pkl              # Saved Random Forest model (auto-generated)
├── pipeline.pkl           # Saved preprocessing pipeline (auto-generated)
├── input.csv              # Test input data (auto-generated)
├── output.csv             # Prediction output file (auto-generated)
├── requirements.txt       # Required dependencies
└── README.md
```

---

## 1. Pipeline Creation and Model Testing (`main.py`)

In **main.py**, the pipeline and models were developed and evaluated step by step:

1. **Data Loading:**  
   Loaded `housing.csv` containing features like median income, total rooms, population, and `ocean_proximity`.

2. **Stratified Sampling:**  
   Implemented stratified sampling based on income categories to ensure balanced train-test splits using `StratifiedShuffleSplit`.

3. **Pipeline Creation:**  
   - **Numerical pipeline:** Imputation with median strategy + StandardScaler.  
   - **Categorical pipeline:** OneHotEncoder for handling text categories.  
   - Combined using `ColumnTransformer`.

4. **Model Evaluation:**  
   Tested and compared:
   - `LinearRegression`
   - `DecisionTreeRegressor`
   - `RandomForestRegressor`  
   Models were evaluated using **cross-validation RMSE**.

5. **Model Selection:**  
   Based on the lowest RMSE, **Random Forest Regressor** was chosen for the final model.

---

## 2. Model Persistence and Inference (`app.py`)

In **app.py**, the project moves from experimentation to deployment readiness:

1. **Model and Pipeline Saving:**  
   - The pipeline and the final trained `RandomForestRegressor` model are serialized using **Joblib**.  
   - Files saved as `pipeline.pkl` and `model.pkl`.

2. **Automated Logic:**  
   - If `model.pkl` doesn’t exist → The script trains a new model and saves both model and pipeline.  
   - If `model.pkl` exists → The script loads them and performs inference.

3. **Inference Flow:**  
   - Reads `input.csv` (created during stratified sampling).  
   - Transforms input data using the saved pipeline.  
   - Predicts house values using the saved model.  
   - Saves results in `output.csv`.

---

## 3. How to Run the Project

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Required libraries listed in `requirements.txt`

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Step 2: Train and Evaluate Models

Run main.py to:

- Build preprocessing pipelines.

- Train multiple models (Linear, Decision Tree, Random Forest).

- Compare RMSE results.

```bash
python main.py
```
Step 3: Build Final Model & Perform Inference

Run app.py to:

- Train and persist the final model if it doesn’t exist.

- Or load existing model to perform inference on new data.
```bash
python app.py
```

Output:

- Creates model.pkl and pipeline.pkl (if not already present).
- Generates input.csv and output.csv containing predictions.
