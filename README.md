# Employee-Salary-Prediction-Using-MLR
Employee salary prediction using multiple linear regression with data preprocessin and user input prediction.

# Employee Salary Prediction Using Multiple Linear Regression

## 📌 Project Overview

This project predicts employee salaries using Multiple Linear Regression, a supervised Machine Learning algorithm.

The project includes data cleaning, missing value handling, categorical data encoding, model training, performance evaluation, data visualization, and interactive salary prediction based on user input.

## 🎯 Project Objectives

- Clean and preprocess employee salary data.
- Handle missing values in the dataset.
- Convert categorical features into numerical values using Label Encoding.
- Train a Multiple Linear Regression model.
- Evaluate the model using regression metrics.
- Visualize relationships between employee attributes and salary.
- Predict an estimated salary using user-provided details.

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 📂 Project Files

- `Mult_Linear_Regression.py` — Main Python script containing data preprocessing, model training, evaluation, visualizations, and user-input prediction.
- `employee_salary_real_world_dataset.csv` — Dataset used for salary prediction.
- `README.md` — Project documentation.

##  Machine Learning Workflow

1. Load the employee salary dataset.
2. Handle missing values in Degree, City, Skill Rating, and Projects Completed.
3. Encode categorical features using Label Encoder.
4. Select eight input features for salary prediction.
5. Split the dataset into training and testing sets using an 80:20 ratio.
6. Train the Multiple Linear Regression model.
7. Generate predictions on the test dataset.
8. Evaluate model performance.
9. Create visualizations to explore the data and prediction results.
10. Collect user input and predict an estimated salary.

##  Model Evaluation

The model is evaluated using the following regression metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted salaries.
- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error in the target's units.
- **R² Score:** Measures how well the model explains variation in salary.

The script also compares training and testing R² scores to help assess model performance.

## 📈 Data Visualizations

The project includes six visualizations:

1. Age vs Salary — Scatter Plot
2. Experience vs Salary — Bar Chart
3. Actual vs Predicted Salary — Scatter Plot
4. Residual Plot — Prediction Error Analysis
5. Degree Distribution — Pie Chart
6. City Distribution — Histogram

## User Input-Based Salary Prediction

The program asks the user to enter:

- Age
- Degree
- Passing Year
- Years of Experience
- Company Size
- City
- Skill Rating
- Number of Completed Projects

The trained model uses these inputs to generate an estimated salary prediction.

##  How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Ankit101AIML/Employee-Salary-Prediction-Using-MLR.git
```

### 2. Open the project directory

```bash
cd Employee-Salary-Prediction-Using-MLR
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 4. Run the Python script

```bash
python Mult_Linear_Regression.py
```

**Important:** Update the dataset path in the Python script to match the location of the CSV file on your computer. The script currently uses a local Windows file path.

## 💡 Key Learnings

- Data cleaning and missing value handling
- Categorical feature encoding
- Supervised Machine Learning
- Multiple Linear Regression
- Train-test splitting
- Regression model evaluation
- Data visualization
- Interactive prediction using user input

## 👨‍💻 Author

**Ankit101AIML**

GitHub: https://github.com/Ankit101AIML
