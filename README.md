# ANN-Classification-Churn
1. Introduction

The objective of this project was to design, train, and deploy an Artificial Neural Network (ANN) capable of predicting customer churn in a banking environment. The model identifies customers who are likely to discontinue their services based on behavioral and demographic indicators. By applying deep learning techniques to structured tabular data, the project aims to help organizations reduce churn and improve customer retention strategies.

2. Dataset and Preprocessing

The dataset used for this study, Churn_Modelling.csv, contains 10,000 records representing bank customers. It includes attributes such as credit score, age, tenure, account balance, number of products, credit-card ownership, active-member status, and estimated salary. The target variable, Exited, indicates whether the customer has left the bank (1) or remained (0).

Non-informative identifiers such as RowNumber, CustomerId, and Surname were removed to avoid bias. The remaining features were a combination of numerical and categorical variables. The categorical attributes “Gender” and “Geography” were encoded numerically using Label Encoding and One-Hot Encoding, respectively. The encoded geography produced three binary columns representing France, Germany, and Spain.

To ensure consistent feature scaling and improved model convergence, numerical columns were standardized using the StandardScaler. After preprocessing, the data consisted of 11 predictor features and one binary target variable. The dataset was split into training and test subsets in an 80:20 ratio using the train_test_split method.

All transformation objects such as label encoders and scaler instances were serialized with Pickle for later reuse during deployment.

3. Model Architecture and Training

The Artificial Neural Network was implemented using TensorFlow’s Keras Sequential API. The architecture consisted of three layers: an input layer with 64 neurons activated by ReLU, a hidden layer with 32 neurons also using ReLU, and an output layer with a single neuron activated by the sigmoid function. This setup allows the network to output a probability score between 0 and 1, representing the likelihood of churn.

The network was compiled with the Adam optimizer using a learning rate of 0.01 and the binary cross-entropy loss function. Accuracy was used as the primary performance metric. The model contained a total of 2,945 trainable parameters and was trained until convergence, achieving approximately 86% accuracy on the test set.

This configuration provided a strong balance between performance and computational efficiency, enabling quick inference without overfitting on the relatively small dataset.

4. Deployment and Application Interface

To make the model accessible to non-technical users, it was deployed using Streamlit as a lightweight web interface. The application allows users to input key customer attributes such as geography, gender, credit score, balance, tenure, and activity level.

Upon submission, the inputs are preprocessed using the same encoders and scaler objects that were fitted during training. The processed data is then passed to the saved ANN model (model.h5) for prediction. The model returns a churn probability score, and based on a threshold of 0.5, the application outputs an interpretable message such as “The customer is likely to churn” or “The customer is not likely to churn.”

This design ensures consistent preprocessing, model reuse, and real-time inference, providing a practical demonstration of model deployment.

5. Results and Discussion

The ANN achieved around 86% accuracy on unseen test data, showing reliable separation between churned and retained customers. The binary cross-entropy loss remained low throughout training, confirming stable learning and effective optimization.

Feature contributions observed during experimentation indicated that credit score, balance, age, and activity level had strong correlations with churn probability. The model generalized well without significant overfitting, making it suitable for production-level scoring or further integration into business intelligence dashboards.

6. Technical Highlights

The project demonstrates a complete machine-learning pipeline, from raw data preparation to deployed application. The workflow includes categorical encoding, scaling, ANN model construction, serialization, and deployment. It integrates reproducibility by saving encoders and model artifacts and leverages a Streamlit front end for ease of use.

The model’s modular architecture allows it to be easily migrated to cloud platforms such as AWS SageMaker or Azure ML, or extended with API-based services using Flask or FastAPI.

7. Conclusion and Future Work

This project successfully showcases an end-to-end deep learning solution for predicting customer churn using structured data. The system combines high predictive performance with real-time deployment capabilities, making it practical for business decision-making.

Future improvements could include performing hyperparameter tuning, incorporating SHAP or LIME for model explainability, and integrating customer feedback loops to continuously retrain the model on new data. These enhancements would make the framework even more adaptable and insightful for enterprise-level churn analysis.
