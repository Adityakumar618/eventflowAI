# EventFlow AI (GridGuard V12)
### Flipkart Gridlock Hackathon Submission

EventFlow AI is a 3-Phase Spatio-Temporal engine designed to predict, map, and prescribe operational solutions for unpredictable traffic gridlock in Bengaluru.

## Architecture Overview
1. **Phase 1 (Predictive ML):** Censored Regression Ensemble (LightGBM, CatBoost, XGBoost) to predict event clearance duration. Generates the Spatio-Temporal Impact Score (STIS) proxy.
2. **Phase 2 (Graph Topology):** NetworkX-based directed graph of Bengaluru corridors to compute Betweenness Centrality and cascade probabilities.
3. **Phase 3 (Prescriptive Operations):** PuLP (MILP) Integer Linear Programming engine that mathematically optimizes the dispatch of limited police officers across concurrent events, and simulates M/M/c delay reductions.

---

## 🚀 How to Run the Project Locally

To run the full-stack Command Center (FastAPI backend + React frontend), follow these steps:

### Prerequisites
* **Python 3.10+** (or newer)
* **Node.js 18+** (or newer)

### Step 1: Start the Backend (Intelligence Engine)
Open a terminal in the root directory of the project and run:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the FastAPI server
python -m uvicorn api:app --reload --port 8000
```
*The API will be running on `http://localhost:8000` with Swagger docs available at `/docs`.*

### Step 2: Start the Frontend (React Command Center)
Open a **second** terminal window and navigate to the `frontend` folder:

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install Node dependencies
npm install

# 3. Start the Vite development server
npm run dev
```
*The interactive UI will now be accessible at `http://localhost:5173`.*

---

## 🧠 Running the Machine Learning Pipeline (Kaggle)
The model training is heavily optimized for a GPU environment. The V12 engine runs over 200 Optuna trials across 5 gradient-boosted tree architectures.

To reproduce the model training yourself:
1. Upload the dataset to Kaggle.
2. Copy the `experiments/gridguard_v12_kaggle.py` script into a Kaggle Notebook cell.
3. Set the accelerator to **GPU P100** or **T4 x2**.
4. Run the script. (Expected runtime: ~60 minutes).

## Standalone Prescriptive Demo
If you want to test the **Phase 3 Prescriptive Engine (PuLP)** in isolation directly in your terminal, run the standalone script:
```bash
python scripts/demo_prescriptive.py
```
