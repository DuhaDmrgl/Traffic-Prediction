# Traffic Prediction using Multi-Layer Perceptron (MLP)
This project aims to predict traffic volume at different junctions using machine learning techniques, specifically employing a Multi-Layer Perceptron (MLP) model. Traffic prediction plays a crucial role in transportation planning, traffic management, and infrastructure development.

## Dataset
The dataset used for this project is available on Kaggle: [Traffic Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset/data).

This dataset contains traffic data collected from sensors at various junctions. Each data point includes information such as date-time, junction ID, and the number of vehicles.

## Dependencies
- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow (Keras)


## Usage
1. **Data Preparation**: Load the dataset, preprocess it, and extract relevant features such as date-time components.
2. **Exploratory Data Analysis (EDA)**: Visualize traffic trends over time, analyze correlations between features, and identify outliers.
3. **Model Training**: Define and train an MLP model using TensorFlow/Keras, considering appropriate architecture and hyperparameters.
4. **Model Evaluation**: Evaluate the trained model's performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
5. **Deployment**: Deploy the trained model for making real-time traffic predictions or integrate it into existing traffic management systems.

## Files
- `traffic_prediction.py`: Jupyter Notebook containing the code for data preprocessing, EDA, model training, and evaluation.
- `traffic.csv`: Dataset containing traffic data.

## Results
The trained MLP model achieved the following performance metrics on the test set:
- Mean Absolute Error (MAE): 0.124
- Root Mean Squared Error (RMSE): 0.162
