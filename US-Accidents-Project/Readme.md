![Python version](https://img.shields.io/badge/Python%20version-3.10.10-light)
![Type of ML](https://img.shields.io/badge/Class-Multi--Class--Classification-orange)
![Type of ML](https://img.shields.io/badge/Type%20of%20ML-Logistic%20Regression-red)
![License](https://img.shields.io/badge/License-Public-green)
![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)
![gradio-ui](https://img.shields.io/badge/UI-Gradio--UI-brightgreen)
![Open Source Love svg1](https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F-Open%20Source-pink)

# US Accidents (2016-2023) Project 

## Motivation and Brief of the Project
This project includes performing EDA, Data Cleaning and training Logistic Regression Models on the March 2023 Update of US-Accidents Dataset, which consists of more than *7.7 million accident instances*

The project can be divided into following major tasks from 2 Kaggle Notebooks
- [Data Processing & EDA](https://www.kaggle.com/code/yuvrajdhepe/project-4-us-accidents-data-eda)
    - a) Data Cleaning 
    - b) EDA 
    - c) Data Cleaning & Transformation
        
- [Model Training](https://www.kaggle.com/code/yuvrajdhepe/project-4-model-training-on-us-accidents-dataset)
    - a) Data Splitting
    - b) Logistic Regressor Model Training
    
## Star Analysis
- Situation: To gain experience in handling, cleaning and working on huge dataset 
- Task: Performing EDA and cleaning of Data to train models on the dataset to find the best predictive model
- Action: Develop idea on the task lines, by exploring data, transforming data to a suitable format for model training
- Result: Gain Hands-on experience in Machine Learning processing huge datasets and building efficient models on the same

## Insights from the EDA n Model Training
- Here I am simply mapping few insights from the EDA & Model Training, detailed work can be found inside the following jupyter notebooks.
    - [US Accidents Dataset EDA]("./images/project-4-us-accidents-data-processing.ipynb")
    - [ML Model Training on Dataset]("./images/project-4-us-accidents-model-runs.ipynb")
    
### Essential Plots from EDA & Key Findings From the Plots
- California is the state with highest number of accidents, followed by Florida & Texas
![Major Accident States]("./images/Top_15_States_Accidents.png")

- As of 2023, Miami records highest number of accidents followed by Houston, Los Angeles & Charlotte
![Major Accident Cities]("./images/Top_15_Cities_Accidents.png")

- Many accidents take place place on Friday, Thursday & Wednesday
![Weekday_Accident_Distribution]("./images/Weekday_Accident_Distribution.png")

- Around 2.7 million accidents happen in Fair Weather
![Weather_Accident_Distribution]("./images/Weather_Accident_Distribution.png")

- More than 6 million accidents are of Severity Level 2
![Severity Distribution in Accidents]("./images/Num_Accidents_Per_Severity_Level.png")

- Number of Accidents have increased over the years, though severity level have decreased
![Annual Accidents Distribution per Severity Level]("./images/Accidents_Organized_by_Severity_Level_per_Year.png")

- Major words used to Describe the Severity of Accidents
![Frequent_Words_PerSeverityLevel]("./images/Frequent_Words_PerSeverityLevel.png")

### Insights From Model Training & Testing
- Top 2 Models (With Hyper Parameter Tuning)

| Model                     | Avg f1 score|
|---------------------------|-------------|
| Random Forest  Classifier | 88.29%      |
| Gradient Boost Clasifier  | 87.27%      |