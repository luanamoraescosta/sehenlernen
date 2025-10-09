<<<<<<< HEAD
# sehenlernen
=======
# Sehen Lernen

This repository contains a refactored version of the Sehen Lernen application, split into two parts:

* **Backend**: A FastAPI service that handles all data processing, feature extraction, and model evaluation.
* **Frontend**: A Streamlit app that provides the user interface and communicates with the backend via HTTP.

---

## 📂 Directory Structure

```
SehenLernen Refactored/
├── Backend/                  # FastAPI backend service
│   ├── app/
│   │   ├── __init__.py       # Makes `app` a Python package
│   │   ├── main.py           # FastAPI app setup, CORS, router inclusion
│   │   ├── routers/          # HTTP endpoint definitions
│   │   │   ├── __init__.py
│   │   │   ├── data_input.py # Upload images & metadata, metadata configure
│   │   │   ├── sampling.py   # Filter and stratified sampling endpoints
│   │   │   ├── features.py   # Histogram, k-means, shape, texture endpoints
│   │   │   ├── stats.py      # Statistical analysis placeholder
│   │   │   └── visualization.py # Visualization placeholder
│   │   ├── services/         # Core processing logic (no HTTP)
│   │   │   ├── __init__.py
│   │   │   ├── data_service.py      # Image storage, metadata read/config
│   │   │   ├── sampling_service.py  # Sampling algorithms
│   │   │   ├── feature_service.py   # Image feature extraction and clustering
│   │   │   └── stats_service.py     # Statistics analysis stub
│   │   ├── models/           # Pydantic request/response schemas
│   │   │   ├── __init__.py
│   │   │   ├── requests.py
│   │   │   └── responses.py
│   │   └── utils/            # Helper modules
│   │       ├── __init__.py
│   │       ├── image_utils.py # Base64 ↔ PIL/fig conversions
│   │       └── csv_utils.py   # CSV/Excel ↔ pandas utilities
│   └── requirements.txt      # Backend dependencies
└── Fronted/                  # Streamlit frontend (note: named `Fronted` in this repo)
    ├── app.py                # Streamlit entrypoint; routes to pages
    ├── components/           # Per-page UI code
    │   ├── sidebar.py        # Navigation sidebar
    │   ├── home.py           # Home page
    │   ├── data_input.py     # File upload & metadata config UI
    │   ├── feature_selection.py # Feature selection UI (histogram, k-means, etc.)
    │   ├── stats_analysis.py # Statistics analysis UI placeholder
    │   └── visualization.py   # Visualization UI placeholder
    ├── utils/                # HTTP client wrapper
    │   └── api_client.py     # Functions to call backend endpoints
    └── requirements.txt      # Frontend dependencies
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Git

### 1. Clone the repository

```bash
git clone https://github.com/luanamoraescosta/sehenlernen.git
cd sehenlernen
```

### 2. Start the Backend

```bash
cd Backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000/`. You can explore API docs at `http://localhost:8000/docs`.

### 3. Start the Frontend

Open a new terminal:

```bash
cd Fronted
pip install -r requirements.txt
export SEHEN_LERNEN_API_URL=http://localhost:8000  # macOS/Linux
# set SEHEN_LERNEN_API_URL on Windows accordingly
streamlit run app.py
```

The Streamlit UI will launch (by default at `http://localhost:8501`).

---

## ⚙️ Configuration

* **`SEHEN_LERNEN_API_URL`**: URL of the backend service (default `http://localhost:8000`).
* **Storage Path**: Backend stores uploaded images under `storage/images/` (auto-created).

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add feature"`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under [MIT License](LICENSE).
>>>>>>> origin/master
