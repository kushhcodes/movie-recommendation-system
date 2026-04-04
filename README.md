# 🎬 Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn)
![Deployed on Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square&logo=render)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**A full-stack content-based movie recommendation engine built on 45,000+ films.**  
Combines TF-IDF vectorization with live TMDB metadata to deliver intelligent, poster-rich recommendations.

[🚀 Live Demo](https://movie-recommendation-system-3e9k.onrender.com) · [📂 GitHub](https://github.com/kushhcodes/movie-recommendation-system) · [📬 Contact](mailto:your@email.com)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [ML Methodology](#-ml-methodology)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [Running the App](#-running-the-app)
- [Deployment](#-deployment)
- [Known Limitations & Future Work](#-known-limitations--future-work)
- [Author](#-author)

---

## 🧠 Overview

This project is a **content-based movie recommendation system** that recommends similar movies based on a selected title. It uses natural language processing (TF-IDF) on movie metadata — overviews, genres, and taglines — to compute pairwise similarity across a corpus of 45,447 films.

The system is split into three distinct layers:

| Layer | Technology | Role |
|---|---|---|
| ML / Data | scikit-learn, pandas, NLTK | Preprocessing, TF-IDF vectorization, cosine similarity |
| Backend | FastAPI, httpx | REST API, TMDB integration, recommendation endpoints |
| Frontend | Streamlit | Interactive UI with poster grids, search, and details view |

The backend is deployed independently on Render and communicates with the Streamlit frontend over HTTP, making the architecture genuinely decoupled and production-realistic.

---

## ✨ Features

- 🔍 **Keyword search** with live TMDB autocomplete suggestions and dropdown
- 🎞️ **Poster-rich home feed** — trending, popular, top rated, now playing, upcoming
- 📄 **Movie detail page** — poster, backdrop, overview, release date, genres
- 🤖 **TF-IDF content recommendations** — top-N similar movies from local dataset
- 🎭 **Genre-based recommendations** — TMDB discover by primary genre
- ⚡ **Async FastAPI backend** — non-blocking TMDB API calls via httpx
- 🧩 **Pydantic-validated responses** — typed models for every endpoint
- 🌐 **CORS-enabled API** — consumable by any frontend
- 🗂️ **Single-page routing** in Streamlit using query params (`?view=details&id=...`)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User (Browser)                        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Streamlit Frontend (app.py)                 │
│  - Home feed / search / detail views                     │
│  - Poster grid rendering                                 │
│  - Session state & query param routing                   │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP (requests)
                           ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (main.py)                   │
│                                                          │
│  ┌─────────────────┐    ┌──────────────────────────┐    │
│  │  Local ML Layer │    │   TMDB API Integration   │    │
│  │                 │    │                          │    │
│  │  df.pkl         │    │  /search/movie           │    │
│  │  tfidf.pkl      │    │  /movie/{id}             │    │
│  │  tfidf_matrix   │    │  /discover/movie         │    │
│  │  indices.pkl    │    │  /trending/movie/day      │    │
│  │                 │    │                          │    │
│  │  TF-IDF cosine  │    │  Async via httpx         │    │
│  │  similarity     │    │                          │    │
│  └─────────────────┘    └──────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  TMDB API (external)   │
              │  https://api.tmdb.org  │
              └────────────────────────┘
```

**Data flow for a recommendation request:**

1. User searches for a movie → Streamlit calls `/tmdb/search` → TMDB returns results
2. User opens a movie → Streamlit calls `/movie/id/{tmdb_id}` → TMDB returns full details
3. Details page calls `/movie/search?query=<title>` (bundle endpoint)
4. Backend looks up the title in local `indices` map → retrieves TF-IDF row → computes cosine similarity across 45K films
5. Top-N similar titles are fetched from TMDB for posters → returned as `TFIDFRecItem` list
6. Genre recommendations are fetched from TMDB `/discover/movie` using the primary genre ID
7. Both recommendation grids are rendered in the Streamlit UI

---

## 🛠️ Tech Stack

| Category | Library / Tool | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | Core language |
| ML | scikit-learn | 1.8.0 | TF-IDF vectorizer, cosine similarity |
| ML | NLTK | latest | Stopword removal, lemmatization |
| Data | pandas | 2.3.3 | DataFrame manipulation |
| Data | numpy | 2.4.3 | Matrix operations |
| Data | scipy | 1.17.1 | Sparse matrix (CSR format) |
| Backend | FastAPI | 0.135.2 | Async REST API framework |
| Backend | uvicorn | 0.42.0 | ASGI server |
| Backend | httpx | 0.28.1 | Async HTTP client for TMDB |
| Backend | pydantic | 2.12.5 | Request/response validation |
| Backend | python-dotenv | 1.2.2 | Environment variable management |
| Frontend | Streamlit | 1.55.0 | Interactive web UI |
| Frontend | requests | 2.33.0 | HTTP calls from Streamlit to API |
| Deployment | Render | — | Cloud hosting (backend + frontend) |
| External API | TMDB | v3 | Movie metadata, posters, search |

---

## 📊 Dataset

**Source:** [TMDB 5000 Movie Dataset / Full MovieLens Metadata](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

| Property | Value |
|---|---|
| Total movies | 45,447 |
| Features used | `title`, `overview`, `genres`, `tagline`, `vote_average`, `popularity` |
| Engineered feature | `tags` = `overview + genres + tagline` (NLP-preprocessed) |
| TF-IDF vocabulary | 50,000 features |
| N-gram range | Unigrams + bigrams `(1, 2)` |
| TF-IDF matrix shape | 45,447 × 50,000 |
| Matrix format | scipy CSR sparse matrix |
| Sparsity | ~99.93% |

### Preprocessing pipeline (in `movie.ipynb`)

```
Raw CSV
  ↓ Drop duplicates
  ↓ Select: title, overview, genres, tagline, vote_average, popularity
  ↓ Drop rows with null title
  ↓ Fill null overview / tagline with ''
  ↓ Parse genres JSON → space-separated string (e.g. "Animation Comedy Family")
  ↓ Concatenate: tags = overview + genres + tagline
  ↓ Lowercase
  ↓ Remove punctuation (regex)
  ↓ Remove stopwords (NLTK English stopwords)
  ↓ Lemmatize words (WordNetLemmatizer)
  ↓ TF-IDF vectorization (max_features=50000, ngram_range=(1,2))
  ↓ Save: df.pkl, tfidf.pkl, tfidf_matrix.pkl, indices.pkl
```

---

## 🤖 ML Methodology

### Content-Based Filtering using TF-IDF + Cosine Similarity

The recommendation model is a **content-based filtering system**. It does not require user interaction data (no ratings, no watch history). Instead, it measures textual similarity between movies based on their metadata.

#### Step 1 — Feature Engineering

For each movie, a `tags` string is constructed by concatenating:
- **Overview** — plot description
- **Genres** — extracted and joined from JSON array
- **Tagline** — marketing tagline

This unified text field captures both narrative content and categorical genre signals.

#### Step 2 — Text Preprocessing

Each `tags` string is preprocessed using a standard NLP pipeline:

```python
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # remove punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    words = [lemmatizer.lemmatize(w) for w in words]   # lemmatize
    return ' '.join(words)
```

#### Step 3 — TF-IDF Vectorization

```python
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])
# Shape: (45447, 50000) — stored as scipy CSR sparse matrix
```

**Why TF-IDF?**  
TF-IDF (Term Frequency–Inverse Document Frequency) weights words that are frequent in a specific movie's description but rare across the corpus — making them strong discriminators. Common words like "film" or "story" are down-weighted; specific words like "wizard" or "heist" are up-weighted.

**Why bigrams?** `ngram_range=(1, 2)` captures two-word phrases like "science fiction", "serial killer", "romantic comedy" — preserving semantically important compound terms.

#### Step 4 — Cosine Similarity (at query time)

Rather than pre-computing an N×N similarity matrix (which would be ~16 GB for 45K movies), similarity is computed **on demand** per query:

```python
def tfidf_recommend_titles(query_title, top_n=10):
    idx = get_local_idx_by_title(query_title)
    query_vector = tfidf_matrix[idx]                          # sparse row
    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()  # dot product with all rows
    order = np.argsort(-scores)                                # sort descending
    # Return top-N (excluding self)
```

Cosine similarity between two TF-IDF vectors measures the angle between them in 50,000-dimensional space — a higher score means more similar metadata.

#### Step 5 — Genre Recommendations (Hybrid Layer)

In addition to TF-IDF, the system fetches genre-based recommendations from TMDB's `/discover/movie` endpoint using the primary genre ID of the selected movie. This provides a **popularity-ranked fallback** when the TF-IDF match quality is low.

### Why not collaborative filtering?

Collaborative filtering requires a user-item interaction matrix (ratings, views). This dataset does not include per-user ratings, making collaborative filtering infeasible without additional data collection. This is noted as a planned upgrade.

---

## 📡 API Reference

**Base URL:** `https://movie-rec-466x.onrender.com`

---

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

---

### `GET /home`

Returns a poster-ready list of movies for the home feed.

**Query Parameters:**

| Param | Type | Default | Options |
|---|---|---|---|
| `category` | string | `popular` | `trending`, `popular`, `top_rated`, `now_playing`, `upcoming` |
| `limit` | int | `24` | 1–50 |

**Response:** `List[TMDBMovieCard]`

```json
[
  {
    "tmdb_id": 550,
    "title": "Fight Club",
    "poster_url": "https://image.tmdb.org/t/p/w500/...",
    "release_date": "1999-10-15",
    "vote_average": 8.4
  }
]
```

---

### `GET /tmdb/search`

Search TMDB for movies by keyword. Returns raw TMDB response with `results` list.

**Query Parameters:**

| Param | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✅ | Search keyword |
| `page` | int | ❌ | Page number (1–10), default 1 |

**Response:** Raw TMDB search response shape `{ results: [...] }`

---

### `GET /movie/id/{tmdb_id}`

Fetch full details of a movie by TMDB ID.

**Path Parameters:** `tmdb_id` — integer TMDB movie ID

**Response:** `TMDBMovieDetails`

```json
{
  "tmdb_id": 550,
  "title": "Fight Club",
  "overview": "A ticking-time-bomb insomniac...",
  "release_date": "1999-10-15",
  "poster_url": "https://image.tmdb.org/t/p/w500/...",
  "backdrop_url": "https://image.tmdb.org/t/p/w500/...",
  "genres": [
    { "id": 18, "name": "Drama" }
  ]
}
```

---

### `GET /recommend/genre`

Fetch genre-based recommendations for a movie using TMDB discover.

**Query Parameters:**

| Param | Type | Required | Description |
|---|---|---|---|
| `tmdb_id` | int | ✅ | TMDB movie ID |
| `limit` | int | ❌ | Max results (default 18) |

**Response:** `List[TMDBMovieCard]`

---

### `GET /recommend/tfidf`

Get raw TF-IDF content recommendations by movie title (local dataset only, no posters).

**Query Parameters:**

| Param | Type | Required | Description |
|---|---|---|---|
| `title` | string | ✅ | Exact movie title in local dataset |
| `top_n` | int | ❌ | Number of recommendations (default 10) |

**Response:**
```json
[
  { "title": "The Dark Knight", "score": 0.342 },
  { "title": "Batman Begins", "score": 0.289 }
]
```

---

### `GET /movie/search` ⭐ Bundle Endpoint

The primary recommendation endpoint. Given a movie title query, returns:
- Full TMDB movie details
- Top-N TF-IDF similar movies (with TMDB posters)
- Genre-based recommendations from TMDB discover

**Query Parameters:**

| Param | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✅ | Movie title query |
| `tfidf_top_n` | int | ❌ | TF-IDF results count (default 12) |
| `genre_limit` | int | ❌ | Genre results count (default 12) |

**Response:** `SearchBundleResponse`

```json
{
  "query": "Toy Story",
  "movie_details": { ... },
  "tfidf_recommendations": [
    {
      "title": "A Bug's Life",
      "score": 0.412,
      "tmdb": {
        "tmdb_id": 9487,
        "title": "A Bug's Life",
        "poster_url": "https://image.tmdb.org/t/p/w500/..."
      }
    }
  ],
  "genre_recommendations": [ ... ]
}
```

---

### Response Models

```python
class TMDBMovieCard:
    tmdb_id: int
    title: str
    poster_url: Optional[str]
    release_date: Optional[str]
    vote_average: Optional[float]

class TMDBMovieDetails:
    tmdb_id: int
    title: str
    overview: Optional[str]
    release_date: Optional[str]
    poster_url: Optional[str]
    backdrop_url: Optional[str]
    genres: List[dict]

class TFIDFRecItem:
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard]

class SearchBundleResponse:
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]
```

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── 📓 movie.ipynb              # Data preprocessing + TF-IDF training notebook
│
├── 🐍 main.py                  # FastAPI backend — all routes and ML inference
├── 🎨 app.py                   # Streamlit frontend — UI, routing, API calls
│
├── 📦 df.pkl                   # Preprocessed DataFrame (45,447 × 7)
├── 📦 tfidf.pkl                # Fitted TfidfVectorizer object
├── 📦 tfidf_matrix.pkl         # Sparse TF-IDF matrix (45,447 × 50,000)
├── 📦 indices.pkl              # Pandas Series: title → DataFrame index
│
├── 📋 requirements.txt         # All Python dependencies with pinned versions
├── 🔒 .env                     # Local environment variables (not committed)
├── 🚫 .gitignore               # Git ignore rules
└── 📄 README.md                # This file
```

### Key files explained

**`movie.ipynb`** — The full data science pipeline. Reads `movies_metadata.csv`, cleans and preprocesses text using NLTK, trains the TF-IDF vectorizer on 45K movie tags, and serializes all artifacts as `.pkl` files.

**`main.py`** — The FastAPI application. Loads pickle files at startup (`@app.on_event("startup")`), exposes 7 REST endpoints, handles async TMDB API calls via `httpx`, and validates all I/O with Pydantic models.

**`app.py`** — The Streamlit UI. Implements single-page routing via `st.query_params`, renders poster grids, handles search autocomplete with TMDB, and displays movie details + two recommendation grids.

**Pickle files** — The serialized ML artifacts. `tfidf_matrix.pkl` is a scipy CSR sparse matrix that enables fast dot-product similarity computation without loading a dense array into memory.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or higher
- A free [TMDB API key](https://www.themoviedb.org/settings/api)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/kushhcodes/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (first time only)

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 5. Set up environment variables

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

### 6. Generate pickle files (if not present)

If the `.pkl` files are not present, run the Jupyter notebook:

```bash
jupyter notebook movie.ipynb
```

Run all cells in order. This will:
1. Load and preprocess `movies_metadata.csv`
2. Train the TF-IDF model
3. Export `df.pkl`, `tfidf.pkl`, `tfidf_matrix.pkl`, `indices.pkl`

> ⚠️ You need `movies_metadata.csv` from the [TMDB Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Download and place it in the root directory before running the notebook.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TMDB_API_KEY` | ✅ | Your TMDB API v3 key. Get it at [themoviedb.org](https://www.themoviedb.org/settings/api) |

The backend raises a `RuntimeError` at startup if this key is missing, giving a clear error message rather than failing silently at request time.

---

## 🚀 Running the App

### Start the FastAPI backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://127.0.0.1:8000`  
Interactive API docs (Swagger UI): `http://127.0.0.1:8000/docs`  
ReDoc: `http://127.0.0.1:8000/redoc`

### Start the Streamlit frontend

In a separate terminal:

```bash
streamlit run app.py
```

UI will be available at `http://localhost:8501`

> By default, `app.py` points to the deployed Render backend. To use your local backend, change `API_BASE` in `app.py`:
> ```python
> API_BASE = "http://127.0.0.1:8000"
> ```

---

## ☁️ Deployment

Both services are deployed on **Render** (free tier).

### Backend (FastAPI on Render)

1. Push your code to GitHub
2. Create a new **Web Service** on Render
3. Set the build command:
   ```bash
   pip install -r requirements.txt
   ```
4. Set the start command:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. Add environment variable: `TMDB_API_KEY = <your key>`
6. Note the deployed URL (e.g. `https://movie-rec-466x.onrender.com`)

### Frontend (Streamlit on Render or Streamlit Cloud)

**Option A — Render:**
- Create a new Web Service
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

**Option B — Streamlit Community Cloud (easier):**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Set `TMDB_API_KEY` in the Secrets manager
4. Deploy — Streamlit handles everything else

> ⚠️ **Render free tier cold starts:** The backend may take 30–60 seconds to respond after a period of inactivity. This is expected on the free tier.

---

## ⚠️ Known Limitations & Future Work

### Current Limitations

| Limitation | Impact | Root Cause |
|---|---|---|
| TF-IDF only, no collaborative filtering | Recommendations don't personalise to user taste | No user-item interaction data |
| No model evaluation metrics | Can't quantify recommendation quality | Not implemented yet |
| scikit-learn pickle version mismatch warning | Cosmetic warning on load; may break across major versions | Trained on sklearn 1.7, running on 1.8 |
| No test suite | Changes can silently break endpoints | Not implemented yet |
| Render free tier cold starts | 30–60s first-load delay | Free tier limitation |
| Title must exist in local dataset | TMDB-only movies get no TF-IDF recs | Dataset is from 2017 and below |

### Planned Improvements

- [ ] **Collaborative filtering** — implement SVD (via `surprise` library) and compare against TF-IDF using Precision@K and NDCG@K metrics
- [ ] **Hybrid re-ranking** — blend TF-IDF cosine score with TMDB `vote_average` and `popularity` as a weighted score: `final = 0.7 × tfidf + 0.2 × rating + 0.1 × popularity`
- [ ] **Evaluation notebook** — offline evaluation with train/test split on implicit feedback
- [ ] **API test suite** — `pytest` with `httpx.AsyncClient` for all endpoints
- [ ] **Dataset refresh** — extend to post-2017 films via TMDB API bulk export
- [ ] **User feedback loop** — thumbs up/down on recommendations to collect implicit signal
- [ ] **Retrain pickles** — fix sklearn version mismatch by retraining on current version

---

## 👤 Author

**Kush** — [@kushhcodes](https://github.com/kushhcodes)

Built as a portfolio project demonstrating end-to-end ML system design: from data preprocessing and NLP-based model training, through a production REST API, to a deployed interactive UI.

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License — use it, modify it, build on it.
```

---

<div align="center">

⭐ If you found this project useful, please consider giving it a star on GitHub!

Made with Python, FastAPI, and a lot of movies 🎬

</div>
