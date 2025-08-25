# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Relative imports within the `app` package
from .routers.data_input      import router as data_input_router
from .routers.sampling        import router as sampling_router
from .routers.features        import router as features_router
from .routers.stats           import router as stats_router
from .routers.visualization   import router as visualization_router
import matplotlib


matplotlib.use("Agg")  # Use non-GUI backend suitable for servers


# --- App Initialization ---
app = FastAPI(
    title="Sehen Lernen API",
    version="1.0.0",
    description="Backend API for Sehen Lernen image processing and analysis"
)

# --- CORS Middleware ---
# Only allow the Streamlit frontend (localhost:8501) to make requests with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
app.include_router(data_input_router,     prefix="/upload",        tags=["Upload"])
app.include_router(sampling_router,       prefix="/sampling",      tags=["Sampling"])
app.include_router(features_router,       prefix="/features",      tags=["Features"])
app.include_router(stats_router,          prefix="/stats",         tags=["Statistics"])
app.include_router(visualization_router,  prefix="/visualization",  tags=["Visualization"])

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Sehen Lernen API"}
