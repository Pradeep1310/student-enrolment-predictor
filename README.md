# Student Enrolment Predictor

A machine learning project that uses Linear Regression to predict future student enrolments based on marketing spend, number of agents, and course fees.

## What it does
- Builds a full ML pipeline: data to features to model to predictions
- Trains a Linear Regression model on 36 months of historical data
- Evaluates model performance with R2 score and Mean Absolute Error
- Forecasts enrolments for the next 3 months
- Produces visualisations: actual vs predicted, and trend forecast chart

## Tools used
- **Scikit-learn** - Linear Regression, train/test split, evaluation metrics
- **Pandas** - data handling
- **NumPy** - numerical operations
- **Matplotlib** - visualisation

## How to run
```bash
pip install scikit-learn pandas numpy matplotlib
python predictor.py
```

## Output
- Model performance report in terminal
- 3-month enrolment forecast
- `enrolment_prediction_chart.png` - visualisation
- `enrolment_predictions.csv` - full historical predictions

## Author
Pradeep Pandey | BITS Pilani Digital B.S. AI & Data Science
