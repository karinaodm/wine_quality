# Wine quality

Predict wine quality based on physicochemical data

## Getting Started

Exploratory data analysis in src/EDA.ipynb

To train models run
src/KNN.ipynb
src/RandomForest.ipynb
src/MLP.ipynb

Models comparison in src/Comparison.ipynb

To create docker container 
cd app
docker build -t myapi .
docker run -d --name myapicontainer -p 80:80 myapi

Example of post request in src/request.py

### Prerequisites

joblib==0.13.0
scikit-learn==0.21.2
fastapi==0.45.0


