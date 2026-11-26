# Cassava leaf disease classifier

This project classifies cassava leaf images into five disease categories. It
includes a FastAPI backend, a Streamlit dashboard, model training, and an
optional Docker setup.

## What you need

- Windows with PowerShell
- Python 3.12 for local use and TensorFlow training
- Docker Desktop, if you want to use Docker Compose
- Kaggle access and credentials, if you need to download the dataset

The original notebook, `cassava_leaf_disease_classification_improved (1).ipynb`,
is kept as a reference. The reusable code is in `src/`.

## Option 1: Run everything with Docker

This is the quickest way to start the backend and dashboard together.

1. Start Docker Desktop.
2. Open PowerShell in the project folder.
3. Start the services:

   ```powershell
   docker compose up --build -d
   ```

4. Check that they are running:

   ```powershell
   docker compose ps
   curl.exe http://localhost:8080/health
   curl.exe http://localhost:8501/_stcore/health
   ```

5. Open:
   - Dashboard: <http://localhost:8501>
   - API documentation: <http://localhost:8080/docs>

The dashboard talks to the API through Nginx. To stop the services, run:

```powershell
docker compose down
```

Docker Compose runs the project on your computer; it does not deploy it to a
cloud provider.

## Option 2: Run the backend and dashboard locally

### 1. Create a Python environment

Run these commands from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-ml.txt -r requirements-dashboard.txt
```

### 2. Start the backend

Open a PowerShell window and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

The local API address is <http://localhost:8000>. Its interactive
documentation is at <http://localhost:8000/docs>.

### 3. Start the dashboard

Open a second PowerShell window in the project folder and run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open <http://localhost:8501>. The dashboard uses `http://localhost:8000` as
the default API address. You can change the API URL in the dashboard sidebar.

## Train a model (optional)

The project includes a training script. It downloads the Kaggle dataset if
the files are not already under `data/cassava/`.

1. Accept the rules for the Cassava Leaf Disease Classification competition
   on Kaggle and configure Kaggle credentials on your computer. Do not commit
   your credentials to this repository.
2. Alternatively, place `train.csv` and the `train_images/` folder under
   `data/cassava/`.
3. From the project folder, run:

   ```powershell
   .\.venv\Scripts\python.exe -m src.train --epochs 10 --batch-size 32
   ```

The command prepares the image cache, trains the model, and evaluates it on a
held-out test split. It saves the model to `models/cassava_model.keras` and
metrics and plots under `results/`.

The dashboard displays the held-out evaluation metrics, confusion matrix, and
per-class recall from `results/metrics/metrics.json`. Keep this report in the
repository so it is available in both cloud services; locally generated plot
images are optional additions.

## Use retraining in the dashboard

Retraining uses the base dataset together with images uploaded and labelled
in the dashboard. Before triggering it:

1. Complete a training run so that
   `data/cassava_cache/224/manifest.csv` exists.
2. Start the backend and dashboard.
3. In the dashboard, upload images with their correct labels.
4. Select **Trigger retraining** and let the job finish.

There must be at least eight images for each of the five classes in the
combined training data. Retraining uses three epochs by default and can take
several minutes. Avoid triggering it more than once at a time.

For a local run, set the base manifest path in the same PowerShell window
before starting the backend:

```powershell
$env:BASE_MANIFEST_PATH = "data/cassava_cache/224/manifest.csv"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Docker Compose sets this path for you.

For Docker, follow its progress in another PowerShell window:

```powershell
docker compose logs -f api
```

The dashboard and Nginx wait up to 30 minutes for this request. If the base
manifest is missing, the API cannot run retraining.

## Run the tests

After installing the dependencies, run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Load-test the API (optional)

Keep Docker Compose running. Install the load-test dependencies and create an
output folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-load-test.txt
New-Item -ItemType Directory -Force results\locust
```

Run the same test for one, two, and three API replicas by changing
`$replicas` each time:

```powershell
$replicas = 1
docker compose up -d --scale api=$replicas
.\.venv\Scripts\locust.exe -f load_testing\locustfile.py --headless --host http://localhost:8080 -u 5 -r 1 -t 2m --csv "results\locust\replicas-$replicas-users-5" --html "results\locust\replicas-$replicas-users-5.html"
```

The default test image is
`data\cassava\train_images\1000015157.jpg`. After all three runs, create a
comparison report:

```powershell
.\.venv\Scripts\python.exe load_testing\compare_results.py --results-dir results\locust
```

The script creates
[comparison.csv](results/locust/comparison.csv) and
[comparison.png](results/locust/comparison.png). The following results were
recorded locally on November 26, 2026. Each run used five simulated users for
two minutes, sending image predictions through Nginx to the API:

| API replicas | Requests | Failures | Requests/second | Average (ms) | Median (ms) | p95 (ms) | p99 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 356 | 0 | 3.00 | 144.2 | 120 | 230 | 440 |
| 2 | 357 | 0 | 3.01 | 142.4 | 130 | 200 | 300 |
| 3 | 350 | 0 | 2.96 | 151.8 | 140 | 170 | 210 |

These are local, preliminary results, not cloud or production measurements.
All three runs had no failed requests and similar throughput at this small
load. The three-replica run had lower p95 and p99 latency, but the average
latency was slightly higher. This test does not demonstrate increased
throughput from adding replicas; more users and repeated runs would be needed
to draw stronger scaling conclusions.

The request totals above come from Locust's saved stats CSV files. The final
terminal summaries showed slightly different totals (357, 359, and 352
requests); use the saved CSV results as the source for the comparison report.

## Cloud deployment

The dashboard and API are deployed separately: the API runs on Render and
the dashboard runs on Streamlit Community Cloud. The prediction-only setup
does not persist uploaded training images or enable retraining.

Current deployment URLs:

- Dashboard: <https://cassava-disease-classification-mkw2rfnypwzbqwnvdvtox5.streamlit.app/>
- API: <https://cassava-disease-classification.onrender.com>
- API health: <https://cassava-disease-classification.onrender.com/health>
- Model status: <https://cassava-disease-classification.onrender.com/model/status>

### 1. Deploy the API to Render

1. Push the deployment files, `models/cassava_model.keras`, and
   `results/metrics/metrics.json` to GitHub. Make sure you are allowed to
   publish the trained model under the dataset and competition terms before
   adding it to a public repository.
2. In Render, create a **New Web Service** and connect this GitHub repository.
3. Choose **Docker** as the runtime and keep `Dockerfile` as the Dockerfile
   path.
4. Add the environment variable `ENABLE_TRAINING_WORKFLOW` with value `false`.
   The Docker image starts Uvicorn on Render's assigned port and checks
   `/health`.
5. Deploy. When it is live, open
   `https://cassava-disease-classification.onrender.com/health` and confirm it
   returns `{"status":"ok"}`. Then check
   `https://cassava-disease-classification.onrender.com/model/status` and
   confirm `"loaded": true`.

### 2. Deploy the dashboard to Streamlit Community Cloud

1. Create an app from this GitHub repository and select the `main` branch.
2. Set the app file path to `dashboard/app.py` and the Python version to 3.12.
   The app dependencies are listed in `dashboard/requirements.txt`.
3. In the app's **Settings → Secrets**, add the following TOML:

   ```toml
   API_URL = "https://cassava-disease-classification.onrender.com"
   ENABLE_TRAINING_WORKFLOW = false
   ```

4. Deploy or reboot the app at
   <https://cassava-disease-classification-mkw2rfnypwzbqwnvdvtox5.streamlit.app/>.
   Check that the API and model show as available, then try a prediction with a
   cassava leaf image.

Do not commit `.streamlit/secrets.toml`, Kaggle credentials, or the dataset.
In this prediction-only deployment, the dashboard hides training controls
and the API rejects upload/retraining requests. Render's local filesystem is
not configured for durable training-data storage.

## Common issues

- **Docker commands fail:** Start Docker Desktop and try again.
- **Predictions are unavailable:** Check that the model exists at
  `models/cassava_model.keras`. The API can start without a model, but
  predictions need one.
- **Retraining is unavailable:** Make sure the cached manifest exists at
  `data/cassava_cache/224/manifest.csv`.
- **The dashboard cannot reach the API:** Check that the API is running and
  that the API URL in the sidebar is correct.

## Main API endpoints

| Endpoint | Purpose |
| --- | --- |
| `/health` | Check whether the API is running |
| `/model/status` | Check whether a model is loaded |
| `/predict` | Predict the class of an uploaded image |
| `/upload-data` | Save labelled images for retraining |
| `/retrain` | Start a retraining run |
| `/metrics` | View available model and service metrics |
| `/docs` | Open the interactive API documentation |
