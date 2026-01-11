## Run
From `infra/` folder:

```bash
docker compose up
```

- MLflow UI: http://localhost:4200
- Airflow UI: http://localhost:8088 (admin / admin)
- Router: http://localhost:8080

## First run (important)
1) Run the DAG `drift_retrain_ab_pipeline`.
   - It will train a model with PyCaret and register it into MLflow Model Registry as `IrisClassifier`.
   - Production & Staging stages will appear after successful run.

2) Then call:
```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 123, "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}}'
```
