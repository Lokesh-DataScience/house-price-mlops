# 🏠 House Price Prediction API — MLOps Project

[![CI/CD](https://github.com/Lokesh-DataScience/house-price-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Lokesh-DataScience/house-price-mlops/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-RandomForest-orange.svg)]()
[![API](https://img.shields.io/badge/API-REST-red.svg)]()

> A **production-ready Machine Learning API** that predicts California house prices using a Random Forest model, served via FastAPI, and fully containerized with Docker.

This project demonstrates **end-to-end MLOps fundamentals**: moving a machine learning model from a Jupyter notebook into a deployable, scalable, production-grade REST API.

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🏗️ Architecture](#️-architecture)
- [✨ Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [⚡ Quick Start](#-quick-start)
- [🐳 Docker Deployment](#-docker-deployment)
- [📡 API Endpoints](#-api-endpoints)
- [🧪 Example Requests](#-example-requests)
- [✅ Testing](#-testing)
- [🚀 Cloud Deployment](#-cloud-deployment)
- [📈 What I Learned](#-what-i-learned)
- [🛣️ Roadmap](#️-roadmap)
- [📝 License](#-license)
- [👤 Author](#-author)

---

## 🎯 Project Overview

This project predicts **median house prices in California** based on 8 input features such as median income, location, and house age. It is designed to showcase **MLOps best practices**:

- ✅ Model training & serialization
- ✅ RESTful API design
- ✅ Input validation & error handling
- ✅ Structured logging
- ✅ Model versioning
- ✅ Containerization with Docker
- ✅ Automated API documentation (Swagger / OpenAPI)
- ✅ Unit testing
- ✅ Environment reproducibility

**Dataset:** [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) (built into scikit-learn)

---

## 🏗️ Architecture

```
┌────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Client   │─────►│   FastAPI    │─────►│ Preprocessor │─────►│   ML Model   │
│  (curl /   │      │   Endpoint   │      │   (numpy)    │      │ (RandomForest)│
│   web app) │◄─────│  /predict    │◄─────│              │◄─────│              │
└────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Logging    │
                    │ + Validation │
                    └──────────────┘

       All components run inside a Docker container 🐳
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **FastAPI Backend** | High-performance async REST API |
| 🧠 **ML Model** | Random Forest Regressor (R² ≈ 0.80) |
| 🛡️ **Input Validation** | Pydantic enforces type & range checks |
| 📚 **Auto Docs** | Interactive Swagger UI at `/docs` |
| 🐳 **Dockerized** | One-command deployment anywhere |
| 📊 **Health Checks** | `/health` endpoint for uptime monitoring |
| 📝 **Structured Logging** | Trace every prediction request |
| 🔢 **Model Versioning** | Built-in version tracking in responses |
| ✅ **Unit Tested** | pytest test suite included |
| 🌐 **Production-Ready** | Uvicorn ASGI server with healthchecks |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10 |
| **ML Framework** | scikit-learn 1.4.0 |
| **API Framework** | FastAPI 0.110 |
| **Server** | Uvicorn (ASGI) |
| **Validation** | Pydantic v2 |
| **Containerization** | Docker |
| **Model Serialization** | joblib |
| **Testing** | pytest |
| **Version Control** | Git + GitHub |

---

## 📁 Project Structure

```
house-price-mlops/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── model.pkl            # Trained ML model (binary)
│   ├── preprocess.py        # Feature preprocessing logic
│   ├── schema.py            # Pydantic request/response models
│   └── logger.py            # Logging configuration
│
├── train/
│   └── train_model.py       # Model training script
│
├── test/
│   └── test_api.py          # API unit tests
│
├── Dockerfile               # Container build instructions
├── .dockerignore            # Files excluded from Docker context
├── .gitattributes
├── pyproject.toml           # Python dependecies (uv)
├── requirements.txt         # Python dependencies (pinned versions)
├── README.md                # You're reading it!
└── LICENSE
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
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

### 3️⃣ Train the Model

```bash
python train/train_model.py
```

You should see:
```
Model R² Score: 0.8042
✅ Model saved to app/model.pkl
```

### 4️⃣ Run the API Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at:
- 🌐 **API Root:** http://localhost:8000
- 📚 **Swagger Docs:** http://localhost:8000/docs
- 📖 **ReDoc:** http://localhost:8000/redoc

---

## 🐳 Docker Deployment

### Build the Image

```bash
docker build -t house-price-api:v1 .
```

### Run the Container

```bash
docker run -d -p 8000:8000 --name house-api house-price-api:v1
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
docker rmi house-price-api:v1      # Remove image
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`        | Root — API info |
| `GET`  | `/health`  | Healthcheck |
| `POST` | `/predict` | Predict house price |
| `GET`  | `/docs`    | Interactive Swagger UI |
| `GET`  | `/redoc`   | Alternative ReDoc UI |

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
  "predicted_price": 4.2156,
  "model_version": "v1.0.0",
  "currency": "USD (100k)"
}
```

> 💡 `4.2156` means **$421,560** (price is in units of $100,000).

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
```

---

## 🚀 Cloud Deployment

This project is ready to deploy to any platform that supports Docker:

| Platform | Cost | Difficulty |
|----------|------|------------|
| [Render](https://render.com)        | Free tier  | ⭐ Easy |
| [Railway](https://railway.app)      | Free tier  | ⭐ Easy |
| [Fly.io](https://fly.io)            | Free tier  | ⭐⭐ Medium |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Free | ⭐ Easy |
| AWS ECS / GCP Cloud Run / Azure ACI | Pay-as-go  | ⭐⭐⭐ Advanced |

### Deploy to Render (Example)

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo → choose **Docker** environment
4. Done ✅ You'll get a public URL like `https://house-price-mlops-m5v1.onrender.com`

---

## 📈 What I Learned

Building this project taught me the core MLOps principles:

- 🧠 **Model Serving** — Exposing ML models via REST APIs
- 🐳 **Containerization** — Ensuring environment reproducibility with Docker
- 🛡️ **Input Validation** — Building robust APIs with Pydantic
- 📝 **Production Logging** — Tracing requests for debugging
- 🔢 **Versioning** — Tracking model iterations in responses
- 📚 **API Documentation** — Auto-generated Swagger / OpenAPI specs
- ✅ **Testing** — Writing unit tests for API endpoints
- 🚀 **Deployment** — Packaging ML models for the cloud

---

## 🛣️ Roadmap

Future enhancements I plan to add:

- [ ] 🔬 Integrate **MLflow** for experiment tracking & model registry
- [✅] 🤖 Set up **GitHub Actions** CI/CD pipeline
- [ ] 📊 Add **Prometheus + Grafana** monitoring
- [ ] 📦 Implement **DVC** for data versioning
- [ ] 🎨 Build a **Streamlit** frontend UI
- [ ] ☁️ Deploy to **AWS ECS** with auto-scaling
- [ ] 🔄 Add **automated retraining** pipeline with Airflow
- [ ] 🔐 Add **API authentication** (API keys / JWT)
- [ ] ⚡ Add **Redis caching** for repeated predictions
- [ ] 📈 Implement **model drift detection** with Evidently AI

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
- [Docker](https://www.docker.com/) — for containerization
- The MLOps community 💙

---

<div align="center">

**Built with ❤️ to learn MLOps from scratch**

[⬆ Back to Top](#-house-price-prediction-api--mlops-project)

</div>