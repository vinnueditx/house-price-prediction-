
# House Price Prediction

An end-to-end Machine Learning pipeline built with Scikit-Learn to predict residential property prices using Random Forest Regression. Features automated preprocessing, median/mode imputation, standard scaling, and one-hot encoding without data leakage, exporting a clean serialized model pipeline for production-ready inference.

---

## Features
- **Data Preprocessing**: Handles missing values with median (numeric) and most-frequent (categorical) imputation.
- **Feature Engineering**: Automated standard scaling and one-hot encoding via `ColumnTransformer`.
- **Zero Data Leakage**: Full Scikit-Learn `Pipeline` implementation.
- **Evaluation Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score.
- **Model Serialization**: Saves full end-to-end pipeline to `.pkl` for direct inference.

---

## Project Structure

```text
house-price-prediction/
│
├── data/
│   └── house_data.csv          # Dataset file
├── models/
│   └── house_price_model.pkl   # Serialized model pipeline
├── train.py                    # Training & evaluation script
├── predict.py                  # Inference script



---
