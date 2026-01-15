# ML-pipeline: от мониторинга data drift до A/B-теста в production

Учебный проект, демонстрирующий полный production-процесс ML-модели:

- мониторинг data drift  
- автоматическое переобучение (PyCaret AutoML)  
- регистрация и версионирование моделей в MLflow  
- A/B-тестирование через Flask-роутер  
- сбор метрик, статистическая проверка и автоматический promotion в Production  

Проект реализован на учебном датасете **Iris** и сфокусирован на **инфраструктуре и MLOps-процессах**, а не на подготовке данных.

## Архитектура проекта

```
.
├── dags/                 # Airflow DAG (drift, retraining, A/B evaluation)
├── src/
│   ├── pipeline.py       # Drift detection + PyCaret AutoML + MLflow
│   └── app.py            # Flask A/B router
├── infra/
│   ├── dockerfiles/
│   │   ├── airflow.dockerfile
│   │   └── flask.dockerfile
│   └── docker-compose.yml
└── README.md
```

## Запуск проекта

Из директории `infra/`:

```bash
docker compose up -d
```

Доступные сервисы:

- **MLflow UI** — http://localhost:4200  
- **Airflow UI** — http://localhost:8088 (admin / admin)  
- **Flask A/B router** — http://localhost:8080  

## Первый запуск (важно)

1. Запуск DAG **`drift_retrain_ab_pipeline`** в Airflow.

- Проверится data drift (PSI + KS-test)
- При превышении порога drift -> запустится PyCaret AutoML
- Лучшая модель зарегистрируется в **MLflow Model Registry** как `IrisClassifier`
- Будут созданы алиасы:
  - `staging` -> новая модель
  - `prod` -> bootstrap (чтобы A-трафик сразу работал)
- **Promotion в `prod` выполняется только после успешного A/B-теста**

2. Первый запрос в API:

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 123,
    "features": {
      "sepal length (cm)": 5.1,
      "sepal width (cm)": 3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)": 0.2
    }
  }'
```

## A/B-роутер (Flask)

- **A-трафик** → `models:/IrisClassifier@prod`  
- **B-трафик** → `models:/IrisClassifier@staging`  

Распределение детерминированное по `user_id`.

## Динамическое управление долей трафика

```bash
curl -X POST http://localhost:8080/config \
  -H 'Content-Type: application/json' \
  -d '{"ab_split_b": 0.5}'
```

Проверка текущей конфигурации:

```bash
curl http://localhost:8080/config
```

## Логирование (volume)

### Flask
- Логи запросов:  
  `ab_router_logs:/usr/src/app/logs/requests.csv`
- Конфигурация роутера:  
  `ab_router_logs:/usr/src/app/logs/router_config.json`

### Airflow
- Drift-отчёты:  
  `airflow_logs:/opt/airflow/logs/drift_reports/`
- A/B отчёты:  
  `airflow_logs:/opt/airflow/logs/ab_reports/ab_eval_latest.json`

## A/B-оценка и promotion

Задача `ab_evaluate_and_promote`:
- считает Accuracy / Precision / Recall / F1
- выполняет chi-square test (p < 0.05)
- при успехе переводит `staging` -> `prod`

## Размеченные запросы (label)

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 10,
    "label": 0,
    "features": {
      "sepal length (cm)": 5.1,
      "sepal width (cm)": 3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)": 0.2
    }
  }'
```
