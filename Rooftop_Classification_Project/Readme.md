![Python version](https://img.shields.io/badge/Python%20version-3.10.10-light)
![Type of ML](https://img.shields.io/badge/Class-Multi--Class--Classification-orange)
![Type of ML](https://img.shields.io/badge/Type%20of%20ML-Logistic%20Regression-red)
![License](https://img.shields.io/badge/License-Public-green)
![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)
![gradio-ui](https://img.shields.io/badge/UI-Gradio--UI-brightgreen)
![Open Source Love svg1](https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F-Open%20Source-pink)

# Rooftop Classification Computer Vision Project

## Project Organization
```
├── Notes Of Project.md                                        : Notes Taken During EDA
├── Readme.md                                                  : Report
├── Dataset/                                                   : Contains Images used for Model Training
├── images/                                                    : Contains Model Comparision Metric Viz's
└── Rooftop_Classification.ipynb                               : EDA & Data Processing NB
```

## Motivation and Brief of the Project
This project is a comprehensive work of Rooftop Classification, with limited number of images. This task made use of *only 25 train and 5 test rooftop images with their binary pixeled label variants*. For coming up with a robust ML solution, I tried 4 different image segmentation approaches by making use of 2 models which were **trained from scratch (CNN, UNET)** and 2 models that used **Transfer learning (Vgg16+Unet, MobilenetV2+Unet)**

The project can be divided into following major tasks from 2 Kaggle Notebooks
- [Data Processing & EDA](https://www.kaggle.com/code/yuvrajdhepe/project-4-us-accidents-data-eda)
    - a) Data Cleaning 
    - b) EDA 
    - c) Data Cleaning & Transformation
    - d) Balancing the Severity Class Distribution, for unbiased Model training
        
- [Model Training](https://www.kaggle.com/code/yuvrajdhepe/project-4-model-training-on-us-accidents-dataset)
    - a) Data Splitting
    - b) Logistic Regressor Model Training
    
## Star Analysis
- Situation: To gain experience in handling, cleaning and working on huge dataset 
- Task: Performing EDA and cleaning of Data to train models on the dataset to find the best predictive model
- Action: Develop idea on the task lines, by exploring data, transforming data to a suitable level for model training
- Result: Gain Hands-on experience in Machine Learning processing huge datasets and building efficient models on the same

## Insights from the EDA n Model Training
- Here I am simply mapping few insights from the EDA & Model Training, detailed work can be found inside the following jupyter notebooks.
    - [US Accidents Dataset EDA](./project-4-us-accidents-data-processing.ipynb)
        - This notebook comprises of EDA and Saving of **Processed Data consisting of around 2L Instances, with  balanced Severity Class Distribution**
    
    - [ML Model Training on Dataset](./project-4-us-accidents-model-runs.ipynb)
        - Notebook which comprises of Model Training Code on Processed Data
        - For comparision and faster training of models, I used a subset of processed data. **Another reason to use subset of Data is few models take too much time *(Kaggle Compute Exausts in this time halting the whole process)* to find best params, so having a subset of data is a quick way to find best models n best params for them.**
        - After finding best params n models, the best models are **trained on whole of processed train data (~2L instances)**, and *validated via the test data (~50K instances)*
    
### Essential Plots from EDA & Key Findings From the Plots
- California is the state with highest number of accidents, followed by Florida & Texas
![Major Accident States](./images/Top_15_States_Accidents.png)

- As of 2023, Miami records highest number of accidents followed by Houston, Los Angeles & Charlotte
![Major Accident Cities](./images/Top_15_Cities_Accidents.png)

- Many accidents take place place on Friday, Thursday & Wednesday
![Weekday_Accident_Distribution](./images/Weekday_Accident_Distribution.png)

- Around 2.7 million accidents happen in Fair Weather
![Weather_Accident_Distribution](./images/Weather_Accident_Distribution.png)

- More than 6 million accidents are of Severity Level 2
![Severity Distribution in Accidents](./images/Num_Accidents_Per_Severity_Level.png)

- Number of Accidents have increased over the years, though severity level have decreased
![Annual Accidents Distribution per Severity Level](./images/Accidents_Organized_by_Severity_Level_per_Year.png)

- Major words used to Describe the Severity of Accidents
![Frequent_Words_PerSeverityLevel](./images/Frequent_Words_PerSeverityLevel.png)

### Insights From Model Training & Testing
- Top 2 Models (With Hyper Parameter Tuning)

| Model                     | Avg f1 score|
|---------------------------|-------------|
| Random Forest  Classifier | 81.80%      |
| XGBoost Clasifier         | 81.16%      |

- **The final model used for this project is thus Random Forest Classifier, to make predictions on real-data, apart from having a good f1 score this model also has a better ROC Curve and PR Curve in comparision with XGBoost Classifier**

- ROC Curve for Test Data via Best Models Trained On All of Processed Train Data
![ROC_Best_Models](./images/Best_Model_Comparision_ROC_Curve_plot.png)

- PR Curve for Test Data via Best Models Trained On All of Processed Train Data
![PR_Best_Models](./images/Best_Model_Comparision_PR_Curve_plot.png)

- Conf Matrix for Test Data via Best Models Trained On All of Processed Train Data
![Conf_Mat_Best_Models](./images/Best_Models_Conf_Matrix_Plot.png)

**Metrics Used: F1**

**Why choose F1 as metrics?**
F1 is a commonly used metric for Logistic Regression evaluation purposes, and complements ROC n PR Curve visualizations by providing a concise and interpretable metric that captures the balance between precision and recall for multi-class classification tasks.

**Following Plots were used to Compare Models n Find Best One**
- ROC Curves for All Models  trained on Sample Subset of Processed Data
![ROC_ALL_Models](./images/All_Models_ROC_Curve_plot.png)

- PR Curves for All Models trained on Sample Subset of Processed Data
![PR_ALL_Models](./images/All_Models_PR_Curve_plot.png)

- Accuracy Metric Comparision for All Models trained on Sample Subset of Processed Data
![Accuracy_ALL_Models](./images/All_Models_Accuracy_Score_Plot_on_Val_Set.png)

- F1-Score Metric Comparision for All Models trained on Sample Subset of Processed Data
![F1_ALL_Models](./images/All_Models_F1_Score_Plot_on_Val_Set.png)

- Conf Matrices for All Models
![Conf_Mat_ALL_Models](./images/All_Models_Conf_Matrix_Plot.png)

## Conclusions
- Random Forest classifier performs the best on **US-Accidents Severity Level Predictions on basis of ROC, PR Curve, Accuracy and F1-Scores**
- Overall every model performs well on Severity Levels 1,3,4 on Validation sets, however the accuracy and recall decreases on Severity Level 2
    - The reasons for this can mainly be while balancing the data the sample that we chose for 2 Severity is not a good representative 
    - This is where one can retrain the models by choosing various random samples, or try including more data of Level 2 Severity and then retrain the models

## What can be improved
- Every model can be trained using cuml GPU implementations on whole dataset, and then do more dense comparisions between models to select the best one

- Trying different subsets of Severity level 2, for balancing the Severity class distribution so our models learn a robust representation for Severity level 2. 
    - This can be achieved by training model on these subsets in parallel and comparing the scores on a fixed validation or test set

- Trying different features from subset which seem similar for ex. in the subset of (zipcode, city, county,  state) I chose to go with city, but one can check model performance with County for ex.
- Trying combination of features which are very less correlated to the Target variable, but might give better correlation when grouped

- One can build a Web-App with simple UI for prediction of Severity of Accidents given feature value ip's
