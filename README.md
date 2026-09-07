# A Deep Learning Enhanced Framework for Multi-Agent Financial Trading

**Team Finovators** · Athish Raj Mohan · Unnati Ulhas Nandrekar · University of Southern California

An extension of the [TradingAgents](https://arxiv.org/abs/2412.20138) multi-agent LLM trading framework that replaces subjective, general-purpose LLM sentiment with domain-specific deep learning signals and grounds agent debates in verifiable evidence.

---

## Motivation

TradingAgents simulates a trading firm using specialized LLM agents (analysts, researchers, traders, risk managers). It works, but three weaknesses limit it in high-stakes settings:

1. **Subjective sentiment.** General-purpose LLMs produce inconsistent sentiment scores that shift with prompt phrasing.
2. **The telephone effect.** Multi-round natural language debate degrades information across turns.
3. **No event awareness.** Technical indicators and diffuse news sentiment fail to surface high-impact corporate events such as earnings surprises in time to act on them.

This repo addresses all three.

---

## Contributions

| # | Enhancement | What it does |
|---|---|---|
| 1 | **FinBERT Sentiment Analyst** | Replaces LLM sentiment with a fine-tuned FinBERT model producing quantitative positive/neutral/negative scores |
| 2 | **Event Impact Analyst** | Detects financial events and assigns a structured impact score on a -5 to +5 scale, routed into analyst reports |
| 3 | **Evidence-anchored debate protocol** | Central evidence registry with unique `evidence-id` per data point; Bull/Bear agents may only cite tagged evidence, validated by a Judge module |

---

## Architecture

```
Data Sources                  Analyst Team              Researcher Team      Risk Mgmt         Decision
─────────────                 ────────────              ───────────────      ─────────         ────────
Yahoo Finance ─┐
Price Charts   ├─ Market ──▶  Market Analyst      ─┐
               │                                   │
Reddit         │              Event Impact         │
Bloomberg      ├─ News   ──▶  Analyst (LLM,        ├──▶  Bullish        ─┐
FinHub API     │              impact scoring)      │     Researcher      │
               │                                   │                     ├──▶ Trader ──▶ Risk Analyst ──▶ Fund
Twitter        ├─ Social ──▶  Social Media &       │     Bearish        ─┘              Compliance      Manager
Reddit         │              News Analyst         │     Researcher                     Analyst
               │              (FinBERT)            │
Insider Txns   │                                   │     Debate Facilitator
Financials     ├─ Fund.  ──▶  Fundamental Analyst ─┤
Company Profile│                                   │
               │              Sentiment Analyst   ─┘
ECTSum         │
Kaggle Market ─┘
```

**Agent roles**

- **Social Media & News Analyst** (FinBERT): extracts sentiment from unstructured news and social feeds.
- **Sentiment Analyst**: aggregates FinBERT outputs into structured sentiment reports.
- **Event Impact Analyst**: identifies and scores financial events, feeding structured signals downstream.
- **Bull / Bear Researchers**: debate using evidence-tagged inputs only.
- **Trader**: synthesizes validated claims, sentiment scores, and event impacts.
- **Risk Management Team**: monitors exposure across risk-seeking, neutral, and conservative stances.
- **Fund Manager**: final trading decision.

---

## Modules

### 1. FinBERT Sentiment Analysis

Fine-tuned on the Financial PhraseBank and Kaggle Financial News corpora.

| Component | Detail |
|---|---|
| Base model | FinBERT (BERT-base architecture) |
| Transformer layers | 12 encoder layers |
| Hidden size | 768 |
| Attention heads | 12 |
| Intermediate size | 3072 |
| Normalization | LayerNorm after each encoder block |
| Output layer | Fully connected + softmax over 3 classes |

`get_finbert_sentiment` wraps `score_finbert`, which loads the model and tokenizer, batches inputs (padding, truncation, `max_length=256`), applies softmax to logits, and maps the resulting probabilities to a label. Class ordering is resolved dynamically from the model ID, since checkpoint conventions differ.

### 2. Event Impact Analyst

An autonomous agent that queries an LLM to score structured financial event rows. Each event row is serialized to text, sent with a system prompt requesting strict JSON, parsed from the first `{` to the last `}`, and clamped to `[-5, 5]`. Parse failures default to `0.0`.

| Score | Interpretation |
|---|---|
| +5 | Extremely positive impact |
| +3 | Clearly positive |
| 0 | Neutral or negligible |
| -3 | Clearly negative |
| -5 | Extremely negative |

### 3. Evidence-Anchored Debate Protocol

A centralized evidence registry stores structured outputs from the Sentiment Analyst, Event Detection Agent, and Technical Indicator module. Every data point carries an `evidence-id`. Bull and Bear researchers are constrained to evidence-tagged reasoning, a Judge module validates claim relevance, and the Trader and Risk Manager consume only validated claims. Any data lacking an `evidence-id` is unusable in debate.

---

## Datasets

### FinBERT evaluation set

Kaggle **Stock Market Dataset for Financial Analysis**. Roughly 2,000 records across AAPL, AMZN, GOOG, MSFT, TSLA, one row per ticker-day, with Open/Close/High/Low, RSI, MACD, and Signal. About 100 rows had missing RSI and were dropped, leaving 1,900 clean records.

Ground-truth `Decision` labels come from a rule-based function:

- **Buy** if `RSI < 30` and `Close > Open`
- **Sell** if `RSI > 70` and `Close < Open`
- **Hold** otherwise

Resulting distribution: ~32% Buy, ~28% Sell, ~40% Hold.

### Event Impact Analyst evaluation set

An integrated dataset built from two sources:

- **Kaggle 9000+ Tickers Stock Market Dataset**: full OHLCV history, temporally rich but with no textual annotations and irregular per-ticker coverage.
- **ECTSum**: earnings call transcript summaries for 100+ tickers, textually rich but sparse in coverage.

Quarterly ECTSum summaries were merged per fiscal period per ticker, then aligned to the corresponding Kaggle OHLCV date. Only events with 3 to 5 consecutive surrounding trading days were retained, guaranteeing enough temporal context to assess short-term (1 to 3 day) market impact.

| Field | Type |
|---|---|
| Ticker | str |
| Date | datetime |
| Open, High, Low, Close | float |
| Volume | float |
| Dividend | float |
| Stock splits | float |
| ECTSum_Summary | str |

---

## Baselines

Because the 9000-ticker Kaggle dataset ships no BUY/SELL/HOLD ground truth, four heuristic label generators were implemented and the framework evaluated against all of them.

**Forward Return Threshold (supervised)**

```
r_t = (Close_{t+1} - Close_t) / Close_t

BUY   if r_t >  θ
SELL  if r_t < -θ
HOLD  if |r_t| ≤ θ          θ = 0.5%
```

**Momentum + Volume Spike (technical)**

```
Body_t  = |Close_t - Open_t|
Range_t = High_t - Low_t
ATR_t   = RollingMean_5(High - Low)
V̄_t     = RollingMean_10(Volume)

BUY  if Close_t > Open_t and Range_t > ATR_t and Volume_t > 1.2 · V̄_t
SELL if Close_t < Open_t and Range_t > ATR_t and Volume_t > 1.2 · V̄_t
HOLD otherwise
```

**Candlestick Body-Range Pattern (price action)**

```
BUY  if Body_t > 0.5 · Range_t and Close_t > Open_t
SELL if Body_t > 0.5 · Range_t and Close_t < Open_t
HOLD otherwise
```

**Unsupervised Clustering (data-driven)**

```
X = StandardScaler([Open, High, Low, Close, Volume])
KMeans(k = 3, random_state = 42)
Δ_t = Close_t - Open_t
```

Clusters are ranked by mean `Δ_t`: highest returns map to BUY, lowest to SELL, middle to HOLD.

A fifth reference point, the **API-based sentiment baseline**, uses precomputed sentiment from FinHub and comparable providers, matching the original TradingAgents implementation.

---

## Results

Evaluation metric is standard classification accuracy, reported alongside relative improvement over baseline.

### FinBERT integration

Balanced 100-record subset, stratified across five tickers.

| Ticker | Baseline (%) | FinBERT (%) | Relative Δ (%) |
|---|---|---|---|
| AAPL | 35.00 | 30.00 | -14.29 |
| AMZN | 15.00 | 40.00 | **+166.67** |
| GOOG | 10.53 | 26.32 | **+150.00** |
| MSFT | 40.00 | 35.00 | -12.50 |
| TSLA | 18.75 | 37.50 | **+100.00** |

**Aggregate:** baseline 28/100 (28.00%), FinBERT 37/100 (37.00%), **+32.14% relative improvement**.

Gains concentrate in volatile, news-sensitive tickers. The AAPL and MSFT declines are likely attributable to sentiment noise, overfitting, or lower signal clarity in their textual data.

### Event Impact Analyst

Balanced 100-record subset across KO, MMM, EME, CNP, GD.

| Baseline | With Event Detection (%) | Without (%) | Relative Δ (%) |
|---|---|---|---|
| Forward Return Threshold | 31.31 | 35.00 | -10.54 |
| Momentum + Volume Spike | 38.38 | 37.00 | +3.73 |
| **Candlestick Body-Range Pattern** | **40.40** | **32.00** | **+26.25** |
| Unsupervised Clustering | 31.31 | 34.00 | -7.91 |

Event detection helps most when the underlying baseline already encodes market psychology or volatility structure. Mechanical return thresholds and unsupervised cluster boundaries do not always align with event-driven insight, which explains the two negative deltas. In some of those cases the impact scoring plausibly overrode a baseline label that was itself misaligned with true market behavior.

---

## Sensitivity Analysis

| Parameter | Finding |
|---|---|
| RSI thresholds (30/70 → 25/75) | Stricter cutoffs shift mass toward HOLD and introduce class imbalance |
| Text source (news vs. social) | FinBERT stays stable across modalities; social inputs show higher score volatility from informal phrasing and sarcasm |
| Forward return θ | 0.3% floods HOLD; 0.7% biases SELL; **0.5% is the stable midpoint** |
| Rolling window (10d → 5d) | 10-day windows produce unstable ATR on shallow per-ticker history; **5 days** restores usable signal |
| KMeans k | k=3 gives clean regime separation (BUY ≈ 66.67, HOLD ≈ 28.33, SELL ≈ 5.00); k=5 fragments centroids and hurts interpretability |

---

## Repository Structure

```
.
├── tradingagents/
│   ├── agents/
│   │   ├── analysts/
│   │   │   ├── sentiment_analyst.py       # FinBERT-backed
│   │   │   ├── event_impact_analyst.py    # impact scoring agent
│   │   │   ├── news_analyst.py
│   │   │   ├── fundamentals_analyst.py
│   │   │   └── market_analyst.py
│   │   ├── researchers/                   # bull / bear / facilitator
│   │   ├── risk_mgmt/
│   │   └── trader/
│   ├── evidence/                          # registry, evidence-id issuance, Judge
│   ├── dataflows/                         # data source adapters
│   └── graph/                             # agent orchestration
├── finbert/
│   ├── score_finbert.py
│   └── get_finbert_sentiment.py
├── data/
│   ├── kaggle_stock_market/
│   ├── kaggle_9000_tickers/
│   └── ectsum/
├── baselines/
│   ├── forward_return.py
│   ├── momentum_volume.py
│   ├── candlestick.py
│   └── clustering.py
├── eval/
│   ├── run_finbert_eval.py
│   └── run_event_eval.py
└── results/
```

---

## Setup

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` in the repo root:

```bash
OPENAI_API_KEY=sk-...
ALPHAVANTAGE_API_KEY=...
FINNHUB_API_KEY=...
FINBERT_MODEL_ID=yiyanghkust/finbert-tone   # or your fine-tuned checkpoint path
```

Download the datasets into `data/` (see `data/README.md` for source links and expected filenames).

---

## Usage

Run the full framework on a single ticker and date:

```bash
python -m tradingagents.main --ticker AAPL --date 2024-06-03
```

Score sentiment standalone:

```python
from finbert.score_finbert import score_finbert

results = score_finbert(
    texts=["Q3 revenue beat consensus by 8%.", "Guidance cut for full year."],
    model_id="yiyanghkust/finbert-tone",
    device="cuda",
    batch_size=16,
)
# -> [{"pos": ..., "neu": ..., "neg": ..., "label": ...}, ...]
```

Reproduce the FinBERT evaluation:

```bash
python eval/run_finbert_eval.py --n 100 --tickers AAPL,AMZN,GOOG,MSFT,TSLA
```

Reproduce the Event Impact Analyst evaluation across all four baselines:

```bash
python eval/run_event_eval.py --n 100 --tickers KO,MMM,EME,CNP,GD --baselines all
```

---

## Known Limitations

- **API rate limits.** AlphaVantage caps at five requests per minute. Sleep intervals are inserted between calls, which slows the pipeline substantially. Evaluation was capped at 100 records per module despite a 2,000-entry dataset being prepared.
- **Free-tier LLM endpoints.** Latency and occasional downtime, compounded by the high call volume inherent to multi-agent communication.
- **No native ground truth.** All BUY/SELL/HOLD labels are heuristic. Results should be read as relative comparisons across baselines, not as absolute trading performance.
- **Ticker coverage mismatch.** The tickers used for FinBERT evaluation are absent from ECTSum, which forced a separate 9000-ticker Kaggle dataset and a separate ticker set for event evaluation.
- **Small evaluation window.** 100 records per module. Broader ticker and regime coverage is the obvious next step.

---

## Future Work

- Finalize and evaluate the evidence-anchored debate protocol, varying enforcement strictness (mandatory vs. optional `evidence-id` citation) and measuring unverifiable claim rate, reasoning drift, and decision traceability.
- Expand evaluation to a wider ticker universe and more market regimes as API budget permits.
- Add robustness metrics beyond accuracy: worst-group accuracy and accuracy gap across sentiment sources.
- Source-aware sentiment calibration to handle the higher volatility observed on social media inputs.
- Study interactions between event detection, sentiment modeling, and debate-based refinement rather than ablating each in isolation.

---

## Citation

```bibtex
@misc{finovators2026multiagent,
  title  = {A Deep Learning Enhanced Framework for Multi-Agent Financial Trading},
  author = {Mohan, Athish Raj and Nandrekar, Unnati Ulhas},
  year   = {2026},
  note   = {University of Southern California}
}
```

## Key References

1. Xiao, Y., Sun, E., Luo, D., Wang, W. (2024). *TradingAgents: Multi-Agents LLM Financial Trading Framework.* arXiv:2412.20138
2. Araci, D. (2019). *FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining.* arXiv:1908.10063
3. Huang, A. H., Wang, H., Yang, Y. (2022). *FinBERT: A Large Language Model for Extracting Information from Financial Text.* Contemporary Accounting Research. arXiv:2006.08097
4. Tian, F., Salim, F. D., Xue, H. (2025). *TradingGroup: A Multi-Agent Trading System with Self-Reflection and Data-Synthesis.* arXiv:2508.17565
5. Li, Y., Yu, Y., Li, H., Chen, Z. (2023). *TradingGPT: Multi-Agent System with Layered Memory and Distinct Characters.* arXiv:2309.03736
6. Ding, Q., Shi, H. (2024). *TradExpert: Revolutionizing Trading with Mixture of Expert LLMs.* arXiv:2411.00782
7. Vidler, A., Walsh, T. (2024). *TraderTalk: An LLM Behavioural ABM applied to Simulating Human Bilateral Trading Interactions.* arXiv:2410.21280
8. Lee, M., Lay-Ki, S. (2024). *Finance Wizard at the FinLLM Challenge Task: Financial Text Summarization.*

## Disclaimer

Research prototype. Not financial advice and not suitable for live trading. Accuracy figures are derived from heuristic labels on small stratified subsets and do not represent realized returns.
