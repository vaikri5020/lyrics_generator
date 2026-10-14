import { useState, useEffect, useRef } from 'react'
import './App.css'

const EXAMPLES = [
  'i want to dance all night',
  'when the lights go down',
  'baby you know that i',
  'we are the champions',
]
const MAX_CHARS = 300
const MAX_HISTORY = 10

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('lyricTheme') || 'light')
  const [seed, setSeed] = useState('')
  const [nWords, setNWords] = useState(30)
  const [output, setOutput] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState('')
  const [history, setHistory] = useState(
    () => JSON.parse(localStorage.getItem('lyricHistory') || '[]')
  )
  const toastTimer = useRef(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('lyricTheme', theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem('lyricHistory', JSON.stringify(history))
  }, [history])

  function showToast(message) {
    setToast(message)
    clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(''), 2000)
  }

  async function predict(text) {
    const clean = text.trim()
    if (!clean) {
      setError('Please enter a seed phrase.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed: clean, n_words: nWords }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Something went wrong.')

      setOutput(data.text)
      setHistory((h) =>
        [{ seed: clean, text: data.text, time: new Date().toLocaleString() }, ...h]
          .slice(0, MAX_HISTORY)
      )
    } catch (err) {
      setError(
        err instanceof TypeError
          ? 'Could not reach the server. Is app.py running?'
          : err.message
      )
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      predict(seed)
    }
  }

  function handleContinue() {
    setSeed(output)
    predict(output)
  }

  function handleCopy() {
    navigator.clipboard
      .writeText(output)
      .then(() => showToast('Copied to clipboard'))
      .catch(() => showToast('Copy failed. Select the text and copy it.'))
  }

  function handleDownload() {
    const blob = new Blob([output], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'prediction.txt'
    link.click()
    URL.revokeObjectURL(url)
  }

  function loadHistoryItem(item) {
    setSeed(item.seed)
    setOutput(item.text)
    setError('')
  }

  const wordCount = output ? output.split(/\s+/).filter(Boolean).length : 0

  return (
    <>
      <header className="header">
        <button
          className="theme-btn"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          {theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>
        <h1>Lyric Predictor</h1>
        <p>Type the start of a lyric and let the model finish it</p>
      </header>

      <main className="layout">
        <div>
          <section className="card">
            <h2>Your lyric</h2>
            <textarea
              value={seed}
              maxLength={MAX_CHARS}
              placeholder="e.g. i want to dance all night"
              onChange={(e) => setSeed(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <div className="meta">
              <span>Press Ctrl + Enter to predict</span>
              <span>{seed.length} / {MAX_CHARS}</span>
            </div>

            <div className="chips">
              {EXAMPLES.map((ex) => (
                <button key={ex} className="chip" onClick={() => setSeed(ex)}>
                  {ex}
                </button>
              ))}
            </div>

            <div className="slider-row">
              <label htmlFor="words">Words to generate</label>
              <input
                id="words"
                type="range"
                min="5"
                max="100"
                value={nWords}
                onChange={(e) => setNWords(Number(e.target.value))}
              />
              <span className="slider-value">{nWords}</span>
            </div>

            <div className="buttons">
              <button className="btn" disabled={loading} onClick={() => predict(seed)}>
                {loading ? 'Predicting...' : 'Predict'}
              </button>
              <button className="btn secondary" onClick={() => setSeed('')}>
                Clear
              </button>
            </div>
          </section>

          <section className="card output-card">
            <h2>Prediction</h2>

            <div className={`output ${!output && !error && !loading ? 'empty' : ''}`}>
              {loading && 'Generating...'}
              {!loading && error && <span className="error">{error}</span>}
              {!loading && !error && (output || 'Your prediction will appear here.')}
            </div>

            <div className="output-footer">
              <span className="word-count">{output ? `${wordCount} words` : ''}</span>
              <div className="actions">
                <button className="btn small secondary" onClick={handleCopy} disabled={!output}>
                  Copy
                </button>
                <button className="btn small secondary" onClick={handleDownload} disabled={!output}>
                  Download
                </button>
                <button className="btn small" onClick={handleContinue} disabled={!output || loading}>
                  Continue
                </button>
              </div>
            </div>
          </section>
        </div>

        <aside className="card">
          <div className="history-header">
            <h2>History</h2>
            <button
              className="btn small secondary"
              onClick={() => {
                setHistory([])
                showToast('History cleared')
              }}
            >
              Clear
            </button>
          </div>

          {history.length === 0 && (
            <p className="empty-history">No predictions yet.</p>
          )}

          <ul className="history-list">
            {history.map((item, i) => (
              <li key={i} className="history-item" onClick={() => loadHistoryItem(item)}>
                <div className="history-seed">{item.seed}</div>
                <div className="history-time">{item.time}</div>
              </li>
            ))}
          </ul>
        </aside>
      </main>

      <footer>Built with a PyTorch LSTM, Flask, and React</footer>
      <div className={`toast ${toast ? 'show' : ''}`}>{toast}</div>
    </>
  )
}