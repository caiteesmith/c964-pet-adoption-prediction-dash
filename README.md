# Pet Shelter Adoption Prediction Dashboard
**Author:** Caitee Smith  
**School:** Western Governors University  
**Program:** Bachelor of Science in Computer Science

## Project Overview
This project is a machine learning dashboard that predicts how quickly a pet will be adopted. Shelters can use it to plan space, staff, and marketing so more pets find homes faster.

**Live App:** [View on Hugging Face](https://huggingface.co/spaces/caiteesmith/pet-adoption-prediction-dash)

![Dashboard](https://imagefa.st/images/2025/09/26/huggingface_dash.png)

## Features
* Upload a shelter CSV and instantly get adoption-probability predictions
* Three interactive charts: adoption fee distribution, age distribution, and photos vs. fee
* Downloadable CSV of predictions for offline use
* Clean, mobile-friendly interface that runs in any browser

## Tech Stack
* **Python 3.13.7+**: Main programming language  
* **Pandas / NumPy**: Data cleaning and feature engineering  
* **scikit-learn**: Machine learning model (Random Forest) and evaluation metrics  
* **Streamlit**: Interactive web dashboard framework  
* **Altair**: Data visualizations and charts  
* **Google Colab**: Environment for training and experimentation  
* **Hugging Face Spaces**: Free hosting and HTTPS deployment
* **Kaggle**: Source for the [Petfinder.my Adoption Prediction dataset](https://www.kaggle.com/competitions/petfinder-adoption-prediction)

## Model Performance
* Accuracy: ~75%  
* Precision (fast adoption): ~66%  
* Recall (fast adoption): ~61%  
* ROC AUC: ~0.79  

## Google Colab Notebook
### Notebook (Data Cleaning & Modeling)
![Google Colab Notebook](https://imagefa.st/images/2025/09/26/Screenshot-2025-09-26-at-1.55.49PM.png)

## Quick Start

### Run the Hosted App
Open the live, hosted app on HuggingFace:
[Pet Shelter Adoption Prediction Dashboard](https://huggingface.co/spaces/caiteesmith/pet-adoption-prediction-dash)
