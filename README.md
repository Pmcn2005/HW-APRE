# Machine Learning Coursework (APRE) 🧠📊

This repository contains a collection of Machine Learning assignments, data analysis projects, and technical reports developed for the Machine Learning (*Aprendizagem*) course. 

Additionally, it includes a separate Networking module implementing a Named Data Networking (NDN) protocol architecture in C.

## 📁 Repository Structure

The repository is organized into distinct homework assignments (HW) and data folders:

* **`HW01/` to `HW04/`**: Contain the core Machine Learning assignments. Each folder typically includes:
  * `*_notebook.ipynb`: The Jupyter Notebook containing the Python code, Exploratory Data Analysis (EDA), model training, and evaluation.
  * `*.pdf`: The detailed technical report discussing the methodology, results, and conclusions for each assignment.

* **`data/`**: The datasets used across the different homework assignments:
  * `Breast_cancer_dataset.csv` - Used for medical diagnostic classification.
  * `diabetes.csv` - Used for predictive modeling of diabetes progression/onset.
  * `hungarian_heart_diseases.csv` - Used for heart disease classification.
  * `rent.csv` - Used for regression tasks predicting real estate rental prices.

* **`Proj_EE.zip`**: A distinct, academic Networking project developed in C. It implements a prototype of the Named Data Networking (NDN) architecture, including Pending Interest Tables (PIT), cache management, and UDP socket communication.

## 🚀 Machine Learning Highlights

Throughout the `HW01` - `HW04` notebooks, the following data science and machine learning pipelines were developed:

1. **Exploratory Data Analysis (EDA)**: Understanding data distributions, identifying outliers, and analyzing feature correlations.
2. **Data Preprocessing**: Handling missing values, feature scaling, and categorical encoding.
3. **Model Training & Selection**: Implementing various classification and regression algorithms (e.g., Logistic Regression, Decision Trees, Random Forests, etc.).
4. **Model Evaluation & Tuning**: Assessing model performance using appropriate metrics (Accuracy, F1-Score, RMSE) and optimizing models via hyperparameter tuning.

## 🛠️ Technologies & Tools

**Machine Learning & Data Science:**
* Python 3
* Jupyter Notebook
* Pandas & NumPy (Data manipulation)
* Scikit-Learn (Machine Learning models and evaluation)
* Matplotlib & Seaborn (Data visualization)

**Networking Project (`Proj_EE`):**
* C Programming
* POSIX Sockets (UDP communication)
* Makefile

## ⚙️ How to Run the Notebooks

1. Clone this repository:
```bash
> git clone [https://github.com/YourUsername/hw-apre.git](https://github.com/YourUsername/hw-apre.git)
```
2. Navigate to the project directory and ensure you have Python installed along with the required libraries (Pandas, Scikit-Learn, Jupyter, etc.).

3. Start the Jupyter Notebook server:
```bash
> jupyter notebook
```

4. Open any `_notebook.ipynb` file from the `HW` folders and run the cells. *(Note: Make sure the relative paths to the `data/` folder are correct within the notebooks).*
