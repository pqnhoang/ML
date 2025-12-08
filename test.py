import requests

url = "https://ml-production-4b33.up.railway.app/predict"

payload = {
  "customer_region": "Quang Nam",
  "source_id": "2.0",
  "campaign_id": "14.0",
  "salesperson_id": "16",
  "team_id": "6",
  "stage_id": "3",
  "stage_sequence": "3",
  "expected_revenue": 800.0,
  "probability": 0.5,
  "lead_age_days": 7,
  "priority": 3,
  "create_month": 11,
  "create_dow": 6
}


r = requests.post(url, json=payload)
print(r.json())
