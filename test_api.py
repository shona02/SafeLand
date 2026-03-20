import sys
sys.path.insert(0, '.')
from backend.app import app

client = app.test_client()
r = client.post('/predict', json={'latitude': 10.1, 'longitude': 76.35})
print(f"Status: {r.status_code}")
try:
    print(r.json)
except Exception as e:
    print("Not JSON:")
    print(r.data.decode('utf-8'))
