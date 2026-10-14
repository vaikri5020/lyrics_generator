# Lyric Predictor

Lyric Predictor is an end-to-end next-word prediction system trained on song lyrics. A Long Short-Term Memory (LSTM) network learns word-sequence patterns from a lyrics corpus, and a web interface lets users enter the opening of a lyric and receive a generated continuation.

The project covers the full pipeline: data preparation, model training, a REST API, a React frontend, and containerized deployment.

---



## Features

- Next-word prediction from a user-provided seed phrase
- Configurable output length from 5 to 100 words
- Prediction history stored in the browser
- Copy, download, and continue actions for generated text
- Interactive API documentation via Swagger UI
- Single-container deployment serving both the API and the web interface

## Architecture

```
Browser (React + Vite)
        │  POST /api/predict
        ▼
FastAPI server (app.py)
        │  NextWordPredictor (predict.py)
        ▼
LSTM model (model.py, weights in models/lstm.pt)
```

- **Data preparation** (`data/prepare.ipynb`): cleans the lyrics, tokenizes them with NLTK, builds the vocabulary, and produces training windows.
- **Training** (`train.py`): trains the LSTM with cross-entropy loss and saves the weights and vocabulary.
- **Inference** (`predict.py`): loads the saved model and generates words one at a time using top-k sampling.
- **Serving** (`app.py`): exposes the prediction endpoint and serves the compiled React application.

## Technology Stack

| Layer | Technologies |
|---|---|
| Model | PyTorch (LSTM, embedding, linear layer) |
| Data processing | pandas, NLTK |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React, Vite |
| Deployment | Docker |



## Prerequisites

- Python 3.10 or later
- Node.js 18 or later and npm
- Docker (optional, for containerized deployment)

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vaikri5020/lyrics_generator.git
cd lyrics_generator
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the data and train the model

Place the lyrics dataset at `data/songs.csv`, then run `data/prepare.ipynb` to generate `data/processed.pt`. Train the model:

```bash
python train.py
```

This writes the trained model to `models/lstm.pt`. A pre-trained `lstm.pt` can be used in place of retraining.

### 4. Install frontend dependencies

```bash
cd lyric-ui/lyric-ui
npm install
cd ../..
```

## Usage

### Development mode

Run the backend and the frontend in separate terminals.

**Backend** (from the project root):

```bash
uvicorn app:app --port 5000 --reload
```

**Frontend** (from `lyric-ui/lyric-ui`):

```bash
npm run dev
```

Open http://localhost:5173 in your browser. The Vite development server forwards `/api` requests to the backend on port 5000.

### Production build

Build the frontend, then serve it through FastAPI:

```bash
cd lyric-ui/lyric-ui
npm run build
cd ../..
mkdir static
cp -r lyric-ui/lyric-ui/dist/* static/
uvicorn app:app --port 7860
```

## API Reference

### `POST /api/predict`

Generates a continuation for a seed phrase.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `seed` | string | yes | Opening words of the lyric |
| `n_words` | integer | no | Number of words to generate, 1 to 100 (default 30) |

**Example request**

```json
{
  "seed": "i want to dance",
  "n_words": 30
}
```

**Responses**

- `200 OK`: `{ "text": "i want to dance all night ..." }`
- `400 Bad Request`: `{ "error": "Please enter a seed phrase." }` when the seed is empty
- `422 Unprocessable Entity`: validation error, for example when `n_words` is outside the allowed range

Interactive documentation is available at http://127.0.0.1:5000/docs while the server is running.

## Docker Deployment

The included Dockerfile builds the React frontend, then packages it with the Python backend into one image.

```bash
docker build -t lyric-predictor .
docker run -p 7860:7860 lyric-predictor
```

Open http://localhost:7860. The application listens on the port set by the `PORT` environment variable, which defaults to 7860.

## Limitations

- The model is trained on a single lyrics dataset, so its output reflects that dataset's style, vocabulary, and biases.
- Predictions use a fixed context window of 10 words and do not model long-range song structure.
- Words outside the training vocabulary are mapped to `<unk>` and are not predicted.
- Generated text is statistically plausible but not guaranteed to be meaningful or grammatical.
- Free hosting tiers may have limited memory, and PyTorch may exceed it.

## Future Work

- Add the artist and genre classifier to the web interface
- Experiment with GRU and Transformer architectures
- Add a validation split and report perplexity on held-out songs
- Support sampling controls such as temperature in the UI



*Built with PyTorch, FastAPI, and React.*
