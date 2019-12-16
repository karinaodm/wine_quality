import requests

url = 'http://localhost/predict'
# Scaled feature vector
data = {'fixed_acidity': 0.25,
        'volatile_acidity': 0.1,
        'citric_acid': 0.1,
        'residual_sugar': 0.1,
        'chlorides': 0.1,
        'free_sulfur_dioxide': 0.1,
        'total_sulfur_dioxide': 0.12,
        'density': 0.1,
        'pH': 0.1,
        'sulphates': 0.1,
        'alcohol': 0.1}

resp = requests.post(url, json=data)
print("Wine quality:")
print(resp.json())