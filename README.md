# 🏠 House Price Prediction API — MLOps Project

[![CI/CD](https://github.com/Lokesh-DataScience/house-price-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Lokesh-DataScience/house-price-mlops/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-EB2027.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model Registry](https://img.shields.io/badge/Model%20Registry-MLflow-purple.svg)]()
[![API](https://img.shields.io/badge/API-REST-red.svg)]()

> A **production-ready Machine Learning API** that predicts California house prices using multiple ML algorithms, tracked & versioned with **MLflow**, served via **FastAPI**, and fully containerized with **Docker**.

This project demonstrates **end-to-end MLOps fundamentals**: moving a machine learning model from a Jupyter notebook into a deployable, scalable, production-grade REST API with **experiment tracking, model registry, and zero-downtime hot-reloading**.

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🏗️ Architecture](#️-architecture)
- [✨ Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [⚡ Quick Start](#-quick-start)
- [🔬 MLflow Workflow](#-mlflow-workflow)
- [🐳 Docker Deployment](#-docker-deployment)
- [📡 API Endpoints](#-api-endpoints)
- [🧪 Example Requests](#-example-requests)
- [✅ Testing](#-testing)
- [🤖 CI/CD Pipeline](#-cicd-pipeline)
- [🚀 Cloud Deployment](#-cloud-deployment)
- [📈 What I Learned](#-what-i-learned)
- [🛣️ Roadmap](#️-roadmap)
- [📝 License](#-license)
- [👤 Author](#-author)

---

## 🎯 Project Overview

This project predicts **median house prices in California** based on 8 input features such as median income, location, and house age. It is designed to showcase **MLOps best practices**:

- ✅ Multi-model training & comparison (Linear, Ridge, Random Forest, Gradient Boosting, XGBoost)
- ✅ **Experiment tracking with MLflow**
- ✅ **Model Registry with Staging → Production lifecycle**
- ✅ **Zero-downtime model hot-reloading**
- ✅ RESTful API design
- ✅ Input validation & error handling
- ✅ Structured logging
- ✅ Containerization with Docker & Docker Compose
- ✅ Automated API documentation (Swagger / OpenAPI)
- ✅ Unit testing
- ✅ GitHub Actions CI/CD pipeline
- ✅ Environment reproducibility

**Dataset:** [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) (built into scikit-learn)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING WORKFLOW                            │
│                                                                     │
│  ┌──────────┐     ┌──────────┐     ┌──────────────────┐            │
│  │  Train   │────►│  MLflow  │────►│  Model Registry  │            │
│  │ Multiple │     │ Tracking │     │ (Staging → Prod) │            │
│  │  Models  │     │  Server  │     │                  │            │
│  └──────────┘     └──────────┘     └────────┬─────────┘            │
│                                              │                      │
└──────────────────────────────────────────────┼──────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE WORKFLOW                            │
│                                                                     │
│  ┌────────┐     ┌─────────┐     ┌─────────────┐     ┌───────────┐  │
│  │ Client │────►│ FastAPI │────►│Preprocessor │────►│ MLflow    │  │
│  │ (curl) │◄────│/predict │◄────│  (numpy)    │◄────│ Prod Model│  │
│  └────────┘     └─────────┘     └─────────────┘     └───────────┘  │
│                      │                                              │
│                      ▼                                              │
│              ┌──────────────┐                                       │
│              │   Logging    │                                       │
│              │ + Validation │                                       │
│              └──────────────┘                                       │
└─────────────────────────────────────────────────────────────────────┘

    All components run inside Docker containers 🐳
    Orchestrated with Docker Compose 🎼
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **FastAPI Backend** | High-performance async REST API |
| 🔬 **MLflow Tracking** | Auto-logs params, metrics, artifacts of every run |
| 🗂️ **Model Registry** | Versioned models with Staging/Production stages |
| 🔄 **Hot Reloading** | Swap production model with zero downtime |
| 🧠 **7+ ML Models** | Linear, Ridge, RandomForest, GradientBoosting, XGBoost |
| 🏆 **Auto-Promotion** | Script auto-selects best model by R² score |
| 🛡️ **Input Validation** | Pydantic enforces type & range checks |
| 📚 **Auto Docs** | Interactive Swagger UI at `/docs` |
| 🐳 **Dockerized** | One-command deployment with Docker Compose |
| 📊 **Health Checks** | `/health` endpoint for uptime monitoring |
| 📝 **Structured Logging** | Trace every prediction request |
| 🔢 **Model Versioning** | Built-in version tracking in API responses |
| ✅ **Unit Tested** | pytest test suite included |
| 🤖 **CI/CD Pipeline** | GitHub Actions auto-tests & deploys |
| 🌐 **Production-Ready** | Uvicorn ASGI server with healthchecks |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10 |
| **ML Frameworks** | scikit-learn 1.4.0, XGBoost 2.0 |
| **Experiment Tracking** | **MLflow 2.11** |
| **Model Registry** | **MLflow Model Registry** |
| **API Framework** | FastAPI 0.110 |
| **Server** | Uvicorn (ASGI) |
| **Validation** | Pydantic v2 |
| **Containerization** | Docker + Docker Compose |
| **Model Serialization** | joblib + MLflow Pyfunc |
| **Testing** | pytest |
| **CI/CD** | GitHub Actions |
| **Version Control** | Git + GitHub |

---

## 📁 Project Structure

```
house-price-mlops/
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app (loads from MLflow Registry)
│   ├── preprocess.py              # Feature preprocessing logic
│   ├── schema.py                  # Pydantic request/response models
│   └── logger.py                  # Logging configuration
│
├── train/
│   ├── train_model.py             # Baseline training script
│   ├── train_with_mlflow.py       # 🔬 Single-model MLflow training
│   ├── compare_models.py          # 🏆 Multi-model comparison + tracking
│   └── promote_best_model.py      # 🚀 Auto-promote best model to Production
│
├── test/
│   └── test_api.py                # API unit tests
│
├── mlruns/                        # 🔬 MLflow artifact storage (gitignored)
├── mlflow.db                      # 🗄️ MLflow SQLite backend (gitignored)
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # 🤖 GitHub Actions pipeline
│
├── Dockerfile                     # API container build
├── docker-compose.yml             # 🎼 Orchestrates API + MLflow server
├── .dockerignore
├── .gitattributes
├── .gitignore
├── pyproject.toml                 # Python dependencies (uv)
├── requirements.txt               # Pinned dependencies
├── README.md                      # You're reading it!
└── LICENSE
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (recommended)
- Git

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Lokesh-DataScience/house-price-mlops.git
cd house-price-mlops
```

### 2️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Start the MLflow Tracking Server

In a **separate terminal**:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

🌐 Open the MLflow UI at: **http://localhost:5000**

### 4️⃣ Train & Compare Multiple Models

```bash
python train/compare_models.py
```

This trains **7 different models** and logs everything to MLflow. Check the UI to compare them side-by-side! 📊

### 5️⃣ Promote the Best Model to Production

```bash
python train/promote_best_model.py
```

This automatically finds the model with the highest R² score and promotes it to the **Production** stage in the Model Registry.

### 6️⃣ Run the API Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will automatically load the **Production** model from the MLflow Registry.

Open your browser at:
- 🌐 **API Root:** http://localhost:8000
- 📚 **Swagger Docs:** http://localhost:8000/docs
- 📖 **ReDoc:** http://localhost:8000/redoc
- 🔬 **MLflow UI:** http://localhost:5000

---

## 🔬 MLflow Workflow

This project implements the full **MLflow lifecycle**: from experimentation → registry → production.

### 🎯 The Complete MLOps Loop

```
1️⃣ Train multiple models       →    python train/compare_models.py
2️⃣ Compare in MLflow UI         →    http://localhost:5000
3️⃣ Promote best to Production   →    python train/promote_best_model.py
4️⃣ Hot-reload API               →    curl -X POST http://localhost:8000/reload-model
5️⃣ New model live in production ✅   (zero downtime, full traceability)
```

### 🏆 Model Comparison Results (Example)

| Rank | Model | R² Score | RMSE | MAE |
|------|-------|---------|------|-----|
| 🥇 1 | XGBoost | 0.8369 | 0.4622 | 0.3019 |
| 🥈 2 | RandomForest (n=300) | 0.8074 | 0.5012 | 0.3245 |
| 🥉 3 | RandomForest (n=100) | 0.7891 | 0.5240 | 0.3458 |
| 4 | Gradient Boosting | 0.7780 | 0.5391 | 0.3502 |
| 5 | Ridge (α=1.0) | 0.5758 | 0.7456 | 0.5332 |
| 6 | Linear Regression | 0.5758 | 0.7456 | 0.5332 |

### 🗂️ Model Registry Lifecycle

```
   None  ──►  Staging  ──►  Production  ──►  Archived
   (dev)       (test)         (live)         (history)
```

Every model registered gets a version number (v1, v2, v3...) with full lineage:
- 🔗 Linked to its training run
- 📋 Parameters & metrics preserved
- 📦 Artifacts (model + signature) stored
- 🏷️ Tags & descriptions

### 🔄 Zero-Downtime Model Updates

When you train a better model and promote it, the API can hot-reload it **without restarting**:

```bash
# 1. Train + promote new model
python train/compare_models.py
python train/promote_best_model.py

# 2. Hot-reload API (instant!)
curl -X POST http://localhost:8000/reload-model

# Response: {"status": "✅ reloaded", "model_version": "9", "model_stage": "Production"}
```

---

## 🐳 Docker Deployment

### Option A: Full Stack with Docker Compose (Recommended)

This runs **both MLflow + API** together with one command:

```bash
docker-compose up --build
```

Access:
- 🌐 **API:** http://localhost:8000
- 🔬 **MLflow:** http://localhost:5000

Stop everything:
```bash
docker-compose down
```

### Option B: Build & Run API Only

```bash
# Build the image
docker build -t house-price-api:v2 .

# Run (assumes MLflow server is accessible)
docker run -d -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_NAME=HousePricePredictor \
  -e MODEL_STAGE=Production \
  --name house-api house-price-api:v2
```

### Verify

```bash
curl http://localhost:8000/health
```

### Useful Docker Commands

```bash
docker logs house-api              # View logs
docker stop house-api              # Stop container
docker rm house-api                # Remove container
docker rmi house-price-api:v2      # Remove image
docker-compose logs -f             # Follow compose logs
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`              | Root — API & model info |
| `GET`  | `/health`        | Healthcheck |
| `POST` | `/predict`       | Predict house price |
| `POST` | `/reload-model`  | 🔄 Hot-reload Production model |
| `GET`  | `/model-info`    | 📊 Get current model metadata (metrics, params, run ID) |
| `GET`  | `/docs`          | Interactive Swagger UI |
| `GET`  | `/redoc`         | Alternative ReDoc UI |

### `POST /predict` — Request Schema

| Field | Type | Constraint | Description |
|-------|------|-----------|-------------|
| `MedInc`     | float | > 0           | Median income in block (10k USD) |
| `HouseAge`   | float | 0–100         | Median house age in years |
| `AveRooms`   | float | > 0           | Average rooms per household |
| `AveBedrms`  | float | > 0           | Average bedrooms per household |
| `Population` | float | > 0           | Block population |
| `AveOccup`   | float | > 0           | Average household occupancy |
| `Latitude`   | float | -90 to 90     | Block latitude |
| `Longitude`  | float | -180 to 180   | Block longitude |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MODEL_NAME`          | `HousePricePredictor`   | Registered model name |
| `MODEL_STAGE`         | `Production`            | Stage to load (Staging/Production) |

---

## 🧪 Example Requests

### ✅ Valid Prediction Request

```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "MedInc": 8.3,
    "HouseAge": 41,
    "AveRooms": 6.9,
    "AveBedrms": 1.0,
    "Population": 322,
    "AveOccup": 2.5,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

**Response (200 OK):**
```json
{
  "predicted_price": 4.1832,
  "model_version": "v8",
  "currency": "USD (100k)"
}
```

> 💡 `4.1832` means **$418,320** (price is in units of $100,000).
> 💡 `v8` indicates this prediction was made by version 8 of the registered model.

### 📊 Inspect the Current Model

```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_name": "HousePricePredictor",
  "version": "8",
  "stage": "Production",
  "run_id": "h4i7j368k0l1m2n3",
  "metrics": {
    "r2": 0.8369,
    "rmse": 0.4622,
    "mae": 0.3019
  },
  "params": {
    "model": "XGBoost",
    "n_estimators": "300",
    "max_depth": "8",
    "lr": "0.1"
  }
}
```

### 🔄 Hot-Reload After Promoting a New Model

```bash
curl -X POST http://localhost:8000/reload-model
```

**Response:**
```json
{
  "status": "✅ reloaded",
  "model_version": "9",
  "model_stage": "Production"
}
```

### ❌ Invalid Request (Validation Error)

```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "MedInc": -5,
    "HouseAge": 200,
    "Latitude": 999
  }'
```

**Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "MedInc"],
      "msg": "Input should be greater than 0"
    }
  ]
}
```

---

## ✅ Testing

Run the test suite with:

```bash
pytest test/ -v
```

**Expected output:**
```
test/test_api.py::test_health PASSED
test/test_api.py::test_predict_valid PASSED
test/test_api.py::test_predict_invalid PASSED
test/test_api.py::test_model_info PASSED
test/test_api.py::test_reload_model PASSED
```

---

## 🤖 CI/CD Pipeline

This project uses **GitHub Actions** for continuous integration & deployment:

| Stage | What It Does |
|-------|--------------|
| 🧹 **Lint** | Runs `flake8` checks |
| 🧪 **Test** | Runs `pytest` test suite |
| 🧠 **Train** | Validates the training pipeline |
| 🐳 **Build** | Builds Docker image |
| 📦 **Push** | Pushes image to Docker Hub |
| 🚀 **Deploy** | Triggers Render deployment |

Workflow file: [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml)

---

## 🚀 Cloud Deployment

This project is ready to deploy to any platform that supports Docker:

| Platform | Cost | Difficulty |
|----------|------|------------|
| [Render](https://render.com)        | Free tier  | ⭐ Easy |
| [Railway](https://railway.app)      | Free tier  | ⭐ Easy |
| [Fly.io](https://fly.io)            | Free tier  | ⭐⭐ Medium |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Free | ⭐ Easy |
| [DagsHub](https://dagshub.com) (for hosted MLflow) | Free | ⭐ Easy |
| AWS ECS / GCP Cloud Run / Azure ACI | Pay-as-go  | ⭐⭐⭐ Advanced |

### 🌐 Live Demo

Try the deployed API: **https://house-price-mlops-m5v1.onrender.com**

### Deploy to Render (Example)

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo → choose **Docker** environment
4. Set environment variables:
   - `MLFLOW_TRACKING_URI` → your remote MLflow URL (e.g., DagsHub)
   - `MODEL_NAME` → `HousePricePredictor`
   - `MODEL_STAGE` → `Production`
5. Done ✅ You'll get a public URL like `https://house-price-mlops-m5v1.onrender.com`

### 🔬 Free Remote MLflow Hosting

To avoid running MLflow locally, use **DagsHub** for free remote tracking:

```python
mlflow.set_tracking_uri("https://dagshub.com/<your-username>/<repo>.mlflow")
```

---

## 📈 What I Learned

Building this project taught me the core MLOps principles:

- 🧠 **Model Serving** — Exposing ML models via REST APIs
- 🔬 **Experiment Tracking** — Logging every training run with MLflow
- 🏆 **Multi-Model Comparison** — Training 7+ algorithms and selecting the winner
- 🗂️ **Model Registry** — Versioning models with Staging/Production lifecycle
- 🔄 **Zero-Downtime Deployment** — Hot-reloading models without restarting the API
- 🐳 **Containerization** — Ensuring environment reproducibility with Docker
- 🎼 **Orchestration** — Running multi-service stacks with Docker Compose
- 🛡️ **Input Validation** — Building robust APIs with Pydantic
- 📝 **Production Logging** — Tracing requests for debugging
- 🔢 **Model Versioning** — Full lineage from training run → production
- 📚 **API Documentation** — Auto-generated Swagger / OpenAPI specs
- ✅ **Testing** — Writing unit tests for API endpoints
- 🤖 **CI/CD** — Automating tests, builds & deployments with GitHub Actions
- 🚀 **Deployment** — Packaging ML models for the cloud

---

## 🛣️ Roadmap

Future enhancements I plan to add:

- [x] 🔬 Integrate **MLflow** for experiment tracking & model registry
- [x] 🤖 Set up **GitHub Actions** CI/CD pipeline
- [ ] ☁️ Move MLflow tracking to **DagsHub** (remote hosting)
- [ ] 📦 Implement **DVC** for data versioning
- [ ] 📊 Add **Prometheus + Grafana** monitoring
- [ ] 🎨 Build a **Streamlit** frontend UI
- [ ] ☁️ Deploy to **AWS ECS** with auto-scaling
- [ ] 🔄 Add **automated retraining** pipeline with Airflow
- [ ] 📈 Implement **model drift detection** with Evidently AI
- [ ] 🔐 Add **API authentication** (API keys / JWT)
- [ ] ⚡ Add **Redis caching** for repeated predictions
- [ ] ☸️ Add **Kubernetes** deployment manifests

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Lokesh Kumar**

- 🌐 Portfolio: [lokeshkumar.com](https://lokesh-datascience.github.io/portfolio/)
- 💼 LinkedIn: [linkedin.com/in/lokesh-kumar-info/](https://www.linkedin.com/in/lokesh-kumar-info/)
- 🐙 GitHub: [@Lokesh-DataScience](https://github.com/Lokesh-DataScience)
- 📧 Email: hellokumarlokesh@gmail.com

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a **⭐ star** on GitHub!

It motivates me to build more open-source MLOps content. 🚀

---

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) — for the California Housing dataset
- [FastAPI](https://fastapi.tiangolo.com/) — for the amazing API framework
- [MLflow](https://mlflow.org/) — for experiment tracking & model registry
- [XGBoost](https://xgboost.ai/) — for gradient boosting
- [Docker](https://www.docker.com/) — for containerization
- The MLOps community 💙

---

<div align="center">

**Built with ❤️ to learn MLOps from scratch**

[⬆ Back to Top](#-house-price-prediction-api--mlops-project)

</div>