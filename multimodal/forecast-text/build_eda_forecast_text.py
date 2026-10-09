import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "eda_forecast_text.ipynb"
HR  = "---"

def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }

def section(title: str, anchor: str) -> dict:
    return md(
        f"{HR}\n\n"
        f"# {title} <a name=\"{anchor}\"></a>\n\n"
        f"{HR}"
    )

cells = []

# ---------------------------------------------------------------------------
# Banner (3 cells) + TOC
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# Forecasting + Text EDA Template\n\n"
    "*Time-series EDA for one or many series, plus a text modality (news, reviews, or posts with dates): how much text arrives over time and at which lag it leads the target. Charts go to `eda-output/`, and the decisions, including the text lag, to `eda_decisions.json`.*"
))

cells.append(md(
    f"{HR}\n\n"
    "## Team Name\n\n"
    "- Name 1 (Role)\n"
    "- Name 2\n"
    "- Name 3"
))

cells.append(md(
    f"{HR}\n\n"
    "## Table of Contents\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Toolkit**](#3)\n"
    "4. [**Structure and Frequency**](#4)\n"
    "5. [**Completeness and Intermittency**](#5)\n"
    "6. [**Target Distribution and Transform**](#6)\n"
    "7. [**Series Overview**](#7)\n"
    "8. [**Decomposition and Strength**](#8)\n"
    "9. [**Seasonal Patterns**](#9)\n"
    "10. [**Autocorrelation and Lags**](#10)\n"
    "11. [**Stationarity**](#11)\n"
    "12. [**Anomalies and Level Shifts**](#12)\n"
    "13. [**Exogenous and Calendar Drivers**](#13)\n"
    "14. [**Cross-Series Structure**](#14)\n"
    "15. [**Baseline Forecastability**](#15)\n"
    "16. [**Text Signal**](#16)\n"
    "17. [**EDA Summary and Pipeline Decisions**](#17)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(md(
    "## Overview\n\n"
    "*Forecasting errors usually come from misunderstanding the data, not from the model: a wrong "
    "frequency, unhandled gaps, a missed seasonality, intermittent demand treated as smooth, or a "
    "validation split that leaks the future. This notebook answers those questions systematically "
    "before any model is trained.*"
))

cells.append(md(
    "## Aim\n\n"
    "*The notebook identifies the frequency, the panel structure, gaps and intermittency, the "
    "right variance-stabilising transform, trend and seasonal strength, dominant periods, "
    "autocorrelation lags, stationarity, anomalies and level shifts, exogenous and calendar "
    "effects, cross-series similarity, and how hard the series are to beat with naive forecasts. "
    "It ends with concrete pipeline decisions: horizon, seasonal periods, lag and rolling features, "
    "transform, backtesting scheme, metric, and model family.*"
))

cells.append(md(
    "## How to Point the Notebook at a Dataset\n\n"
    "*All changes happen in `Settings` (section 2). Data must be in long format: one row per series "
    "and timestamp.*\n\n"
    "| Setting | Purpose | Example |\n"
    "|---|---|---|\n"
    "| `KAGGLE_PATH` / `COLAB_PATH` / `LOCAL_PATH` | Training file per environment | `'/kaggle/input/comp'` (folder) or a file path |\n"
    "| `TEST_*_PATH` | Optional future frame; its dates define the forecast horizon | `'.../test.csv'` |\n"
    "| `DATE_COL`, `TARGET_COL` | Timestamp and value to forecast | `'date'`, `'sales'` |\n"
    "| `ID_COLS` | Columns identifying a series; `[]` for a single series | `['store', 'item']` |\n"
    "| `FREQ` | Pandas frequency, or `'auto'` to infer | `'D'`, `'W-MON'`, `'MS'`, `'h'` |\n"
    "| `EXOG_COLS` | Known covariates, or `'auto'` for all other numeric columns | `['price', 'promo']` |\n"
    "| `HORIZON` | Steps to forecast, or `'auto'` (from the test file, else a frequency default) | `28` |\n\n"
    "*When the configured file does not exist and `DEMO_IF_MISSING = True`, a realistic synthetic "
    "retail dataset is generated so the notebook can be explored end to end.*"
))

cells.append(md(
    "## Dataset\n"
    "\n"
    "*Any time series in long format works: one row per date (and per series), a date column, a numeric target, optional series identifiers (store, item, region), and optional exogenous columns (price, promotion, weather). Put the files in `data/` next to this notebook, or point `KAGGLE_PATH` / `COLAB_PATH` / `LOCAL_PATH` at the dataset folder.*\n"
    "\n"
    "```\n"
    "data/\n"
    "├── train.csv              (date, ids, target, exogenous)\n"
    "├── test.csv               (future dates to forecast, optional)\n"
    "└── sample_submission.csv  (id + target columns, optional)\n"
    "```\n"
    "\n"
    "*With `'auto'` the date column is recognised by name, the target comes from `sample_submission.csv` (or is the only train column missing from the test file), and the remaining text columns become series identifiers. When no data is found and `DEMO_IF_MISSING = True`, a synthetic retail panel is generated in `demo-data/` so the notebook still runs end to end.*"
))

cells.append(md(
    "## Approach: Diagnostic Ladder\n\n"
    "*Each step answers one modelling question, and its answer feeds the decisions file.*\n\n"
    "```\n"
    "Long table (id, date, target, exog)\n"
    "    |\n"
    "    +-- Structure      -> frequency, number of series, lifespans          -> FREQ, HORIZON\n"
    "    +-- Completeness   -> gaps, zeros, ADI / CV^2 intermittency           -> fill strategy, Croston/TSB\n"
    "    +-- Distribution   -> skew, variance-level slope                      -> log / none transform\n"
    "    +-- Decomposition  -> STL trend and seasonal strength                 -> seasonal models\n"
    "    +-- Seasonality    -> profiles, periodogram                           -> seasonal periods\n"
    "    +-- Autocorrelation-> ACF / PACF significant lags                     -> lag features\n"
    "    +-- Stationarity   -> ADF + KPSS, differencing order                  -> ARIMA d, detrending\n"
    "    +-- Anomalies      -> STL residual outliers, level shifts             -> cleaning, training window\n"
    "    +-- Drivers        -> lagged cross-correlation, calendar effects      -> exogenous features\n"
    "    +-- Cross-series   -> correlation, Pareto of volume                   -> global vs local models\n"
    "    +-- Baselines      -> naive / seasonal naive backtest (MASE, WAPE)    -> the score to beat\n"
    "         |\n"
    "    eda_summary.md + eda_decisions.json\n"
    "```"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))

cells.append(md(
    "## Environment Setup\n\n"
    "The notebook runs on CPU only. The `nvidia-smi` call is kept for consistency with the other "
    "templates and fails harmlessly on CPU machines."
))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(code(
    "%pip install -q numpy pandas matplotlib seaborn scipy scikit-learn statsmodels pyarrow openpyxl"
))

cells.append(md("## Import Libraries"))

cells.append(code(
    "import os\n"
    "import re\n"
    "import json\n"
    "import random\n"
    "import warnings\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib as mpl\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.dates as mdates\n"
    "from matplotlib.colors import LinearSegmentedColormap\n"
    "import seaborn as sns\n"
    "from sklearn.decomposition import TruncatedSVD\n"
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    "from scipy import signal, stats\n"
    "from statsmodels.tsa.seasonal import STL\n"
    "from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf\n"
    "from IPython.display import Markdown, display\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "pd.set_option('display.max_columns', 100)\n"
    "pd.set_option('display.width', 200)"
))

cells.append(md("## Seed Everything"))

cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "\n"
    "seed_everything(42)"
))

cells.append(md(
    "## Settings\n\n"
    "All paths, column roles, frequency, horizon, and analysis knobs are centralised here. "
    "Switching between Kaggle, Colab, and local execution only requires editing the path lines."
))

cells.append(code(
    "class Settings:\n"
    "    SEED       = 42\n"
    "    _ON_KAGGLE = Path('/kaggle/input').exists()\n"
    "    _ON_COLAB  = Path('/content').exists() and not _ON_KAGGLE\n"
    "\n"
    "    # 1) Data location: a file or a dataset folder (train / test / sample_submission are found by name)\n"
    "    KAGGLE_PATH      = '/kaggle/input/<dataset-slug>'\n"
    "    COLAB_PATH       = '/content/drive/MyDrive/<folder>'\n"
    "    LOCAL_PATH       = 'data'\n"
    "    TEST_KAGGLE_PATH = None\n"
    "    TEST_COLAB_PATH  = None\n"
    "    TEST_LOCAL_PATH  = None\n"
    "    READ_KWARGS      = {}\n"
    "    DEMO_IF_MISSING  = True\n"
    "\n"
    "    # 2) Columns\n"
    "    DATE_COL   = 'auto'\n"
    "    TARGET_COL = 'auto'\n"
    "    ID_COLS    = 'auto'\n"
    "    EXOG_COLS  = 'auto'\n"
    "    AGG        = 'sum'\n"
    "    # 2b) Text modality: a text column of the main table, or a separate dated text file (news, reviews, posts)\n"
    "    TEXT_KAGGLE_PATH = None\n"
    "    TEXT_COLAB_PATH  = None\n"
    "    TEXT_LOCAL_PATH  = None\n"
    "    TEXT_COL         = 'auto'\n"
    "    TEXT_DATE_COL    = 'auto'\n"
    "    TEXT_ID_COLS     = 'auto'\n"
    "    TEXT_DIM         = 8\n"
    "    TEXT_LAG         = 'auto'\n"
    "\n"
    "    # 3) Time settings\n"
    "    FREQ             = 'auto'\n"
    "    HORIZON          = 'auto'\n"
    "    SEASONAL_PERIODS = 'auto'\n"
    "\n"
    "    # 4) Analysis knobs\n"
    "    MAX_SERIES_PLOT = 12\n"
    "    MAX_SERIES_STAT = 500\n"
    "    MAX_LAGS        = 'auto'\n"
    "    OUTLIER_Z       = 4.0\n"
    "\n"
    "    # 5) Outputs\n"
    "    DATA_PATH  = KAGGLE_PATH if _ON_KAGGLE else COLAB_PATH if _ON_COLAB else LOCAL_PATH\n"
    "    TEST_PATH  = TEST_KAGGLE_PATH if _ON_KAGGLE else TEST_COLAB_PATH if _ON_COLAB else TEST_LOCAL_PATH\n"
    "    TEXT_PATH  = TEXT_KAGGLE_PATH if _ON_KAGGLE else TEXT_COLAB_PATH if _ON_COLAB else TEXT_LOCAL_PATH\n"
    "    OUTPUT_DIR = Path('/kaggle/working/eda-output') if _ON_KAGGLE else Path('eda-output')\n"
    "    FIG_DIR    = OUTPUT_DIR / 'figures'\n"
    "    TABLE_DIR  = OUTPUT_DIR / 'tables'\n"
    "\n"
    "CFG = Settings()\n"
    "for d in (CFG.OUTPUT_DIR, CFG.FIG_DIR, CFG.TABLE_DIR):\n"
    "    d.mkdir(parents=True, exist_ok=True)\n"
    "env = 'Kaggle' if CFG._ON_KAGGLE else 'Colab' if CFG._ON_COLAB else 'Local'\n"
    "print(f'Environment : {env}')\n"
    "print(f'Data path   : {CFG.DATA_PATH}')\n"
    "print(f'Test path   : {CFG.TEST_PATH}')\n"
    "print(f'Output dir  : {CFG.OUTPUT_DIR.resolve()}')"
))

cells.append(md(
    "## Demo Data Generator\n\n"
    "The generator builds a long-format retail panel that contains every pattern the notebook is "
    "designed to detect. It only runs when the configured training file is missing and "
    "`DEMO_IF_MISSING` is true, so it never overwrites real data."
))

cells.append(code(
    "def make_demo(train_path, test_path, seed=42):\n"
    "    rng = np.random.RandomState(seed)\n"
    "    dates = pd.date_range('2021-01-01', '2024-06-30', freq='D')\n"
    "    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=28, freq='D')\n"
    "    lebaran = pd.to_datetime(['2021-05-13', '2022-05-02', '2023-04-22', '2024-04-10'])\n"
    "    all_dates = dates.append(future)\n"
    "    t = np.arange(len(all_dates))\n"
    "    weekly = np.array([0.9, 0.85, 0.9, 0.95, 1.1, 1.35, 1.25])[all_dates.dayofweek]\n"
    "    yearly = 1 + 0.15 * np.sin(2 * np.pi * (all_dates.dayofyear - 80) / 365.25)\n"
    "    days_to_eid = np.min(np.abs((all_dates.values[:, None] - lebaran.values[None, :]).astype('timedelta64[D]').astype(int)), axis=1)\n"
    "    eid = 1 + 1.2 * np.exp(-days_to_eid / 4.0) * (days_to_eid <= 14)\n"
    "    rows, news = [], []\n"
    "    words = 'cuaca hari ini kota ramai jalan macet pasar harga stabil warga libur sekolah acara musik'.split()\n"
    "    for s in range(1, 7):\n"
    "        # store announcements lift sales 30 to 36 days later, longer than the 28-day horizon\n"
    "        ann = np.where((rng.rand(len(t)) < 0.03) & (t < len(dates)))[0]\n"
    "        boost = np.ones(len(t))\n"
    "        for i in ann:\n"
    "            boost[i + 30:i + 37] *= 1.7\n"
    "            news.append({'date': all_dates[i], 'store': f'S{s}', 'headline': f'toko S{s} umumkan diskon besar dan promo spesial bulan depan'})\n"
    "        for i in np.where(rng.rand(len(dates)) < 0.6)[0]:\n"
    "            news.append({'date': dates[i], 'store': f'S{s}', 'headline': ' '.join(rng.choice(words, 7))})\n"
    "        for it in range(1, 6):\n"
    "            base = rng.uniform(20, 120) if it <= 3 else rng.uniform(0.3, 1.5)\n"
    "            trend = 1 + rng.uniform(-0.1, 0.35) * t / len(t)\n"
    "            price0 = rng.uniform(10, 60)\n"
    "            price = price0 * (1 + 0.04 * np.sin(t / 90 + s)) * np.where(rng.rand(len(t)) < 0.03, 0.85, 1.0)\n"
    "            promo = (rng.rand(len(t)) < 0.06).astype(int)\n"
    "            lam = base * trend * weekly * yearly * eid * (1 + 0.5 * promo) * (price / price0) ** -1.5 * boost\n"
    "            if s == 3 and it == 1:\n"
    "                lam = lam * np.where(all_dates >= '2023-03-01', 1.6, 1.0)\n"
    "            y = rng.poisson(lam * rng.gamma(20, 1 / 20, len(t))).astype(float)\n"
    "            frame = pd.DataFrame({'date': all_dates, 'store': f'S{s}', 'item': f'I{it}', 'sales': y,\n"
    "                                  'price': price.round(2), 'promo': promo})\n"
    "            if s == 6:\n"
    "                frame = frame[frame['date'] >= '2022-07-01']\n"
    "            rows.append(frame)\n"
    "    full = pd.concat(rows, ignore_index=True)\n"
    "    train = full[full['date'] <= dates[-1]].copy()\n"
    "    spikes = train.sample(12, random_state=seed).index\n"
    "    train.loc[spikes, 'sales'] *= 8\n"
    "    train = train.drop(train[(train['date'].between('2022-02-10', '2022-02-16')) & (train['store'] == 'S2')].index)\n"
    "    train = train.drop(train.sample(frac=0.01, random_state=seed).index)\n"
    "    test = full[full['date'] > dates[-1]].drop(columns='sales')\n"
    "    Path(train_path).parent.mkdir(parents=True, exist_ok=True)\n"
    "    train.to_csv(train_path, index=False)\n"
    "    test.to_csv(test_path, index=False)\n"
    "    pd.DataFrame(news).to_csv(Path(train_path).parent / 'news.csv', index=False)\n"
    "    print(f'Demo data written: {train_path} ({len(train):,} rows), {test_path} ({len(test):,} rows)')"
))

cells.append(md(
    "## Load Dataset\n\n"
    "The file is resolved (with the Kaggle fallback to the largest tabular file), loaded, and the "
    "date column parsed. Duplicate rows for the same series and timestamp are aggregated with `AGG`, "
    "and every series receives a single string key."
))

cells.append(code(
    "TABULAR_EXT = ('.csv', '.tsv', '.txt', '.parquet', '.pq', '.feather', '.xlsx', '.xls', '.json', '.zip', '.gz')\n"
    "\n"
    "def load_table(path, **kwargs):\n"
    "    suffixes = [s.lower() for s in path.suffixes]\n"
    "    if '.parquet' in suffixes or '.pq' in suffixes:\n"
    "        return pd.read_parquet(path, **kwargs)\n"
    "    if '.feather' in suffixes:\n"
    "        return pd.read_feather(path, **kwargs)\n"
    "    if '.xlsx' in suffixes or '.xls' in suffixes:\n"
    "        return pd.read_excel(path, **kwargs)\n"
    "    if '.json' in suffixes:\n"
    "        return pd.read_json(path, **kwargs)\n"
    "    if '.tsv' in suffixes:\n"
    "        kwargs.setdefault('sep', '\\t')\n"
    "    return pd.read_csv(path, low_memory=False, **kwargs)\n"
    "\n"
    "def file_role(f):\n"
    "    \"\"\"Classify a data file by its name: 'train', 'test', 'sample' (submission template), or None.\"\"\"\n"
    "    stem = f.name.lower().split('.')[0]\n"
    "    if 'sample' in stem or 'submission' in stem:\n"
    "        return 'sample'\n"
    "    if stem.startswith('test'):\n"
    "        return 'test'\n"
    "    if stem.startswith('train'):\n"
    "        return 'train'\n"
    "    return None\n"
    "\n"
    "SKIP_DIRS = {'eda-output', 'pipeline-output', 'catboost_info', '.ipynb_checkpoints', '.git', '.venv', 'site-packages'}\n"
    "LOCAL_DATA_DIRS = ('data', 'input', 'dataset')\n"
    "\n"
    "def discover_files(data_path, test_path):\n"
    "    \"\"\"Resolve the train, test, and sample-submission files from a file, a dataset folder, or the input root.\"\"\"\n"
    "    p = Path(data_path) if data_path else None\n"
    "    explicit_test = Path(test_path) if test_path and Path(test_path).exists() else None\n"
    "    if test_path and explicit_test is None:\n"
    "        print(f'[warn] test path {test_path} not found; searching the dataset folder instead')\n"
    "    if p is not None and p.is_file():\n"
    "        train, files = p, [f for f in p.parent.iterdir() if f.is_file() and f.suffix.lower() in TABULAR_EXT]\n"
    "    else:\n"
    "        if p is not None and p.is_dir():\n"
    "            roots = [p]\n"
    "        elif CFG._ON_KAGGLE:\n"
    "            roots = [Path('/kaggle/input')]\n"
    "        elif CFG._ON_COLAB:\n"
    "            roots = [Path('/content')]\n"
    "        else:\n"
    "            # local: data/, input/, dataset/ next to the notebook first, then the notebook folder itself\n"
    "            roots = [Path(d) for d in LOCAL_DATA_DIRS] + [Path('.')]\n"
    "        if p is not None and not p.is_dir():\n"
    "            print(f'[warn] {p} not found; searching {[str(r) for r in roots if r.exists()]}')\n"
    "        train, files = None, []\n"
    "        for r in (r for r in roots if r.exists()):\n"
    "            files = [f for f in r.rglob('*') if f.is_file() and f.suffix.lower() in TABULAR_EXT\n"
    "                     and not SKIP_DIRS & set(f.parts)]\n"
    "            named = [f for f in files if file_role(f) == 'train']\n"
    "            others = [f for f in files if file_role(f) is None]\n"
    "            train = max(named or others, key=lambda f: f.stat().st_size, default=None)\n"
    "            if train is not None:\n"
    "                break\n"
    "    if train is None:\n"
    "        return None, explicit_test, None\n"
    "\n"
    "    def pick(role):\n"
    "        near = [f for f in files if file_role(f) == role and f.parent == train.parent]\n"
    "        cands = near or [f for f in files if file_role(f) == role]\n"
    "        exact = [f for f in cands if f.name.lower().split('.')[0] == role]\n"
    "        return min(exact or cands, key=lambda f: len(f.name), default=None)\n"
    "\n"
    "    test = explicit_test or pick('test')\n"
    "    sample = pick('sample')\n"
    "    for role, f in (('train', train), ('test', test), ('sample submission', sample)):\n"
    "        print(f'{role:<18}: {f if f else \"(none found)\"}')\n"
    "    return train, test, sample\n"
    "\n"
    "DATA_PATH, TEST_PATH, SAMPLE_PATH = discover_files(CFG.DATA_PATH, CFG.TEST_PATH)\n"
    "if DATA_PATH is None and CFG.DEMO_IF_MISSING:\n"
    "    make_demo('demo-data/train.csv', 'demo-data/test.csv', CFG.SEED)\n"
    "    DATA_PATH = Path('demo-data/train.csv')\n"
    "    TEST_PATH = Path('demo-data/test.csv')\n"
    "assert DATA_PATH is not None, f'{CFG.DATA_PATH} not found. Edit the path lines in Settings.'\n"
    "if TEST_PATH is None:\n"
    "    print('[warn] No test file configured or found in the dataset folder: the horizon falls back to the frequency default.')\n"
    "\n"
    "raw = load_table(DATA_PATH, **CFG.READ_KWARGS)\n"
    "raw_test = load_table(TEST_PATH, **CFG.READ_KWARGS) if TEST_PATH else None\n"
    "def resolve_columns():\n"
    "    \"\"\"Fill 'auto' columns: date by name, target from sample_submission or the train-only column, ids = text columns.\"\"\"\n"
    "    if CFG.DATE_COL == 'auto':\n"
    "        named = [c for c in raw.columns if 'date' in c.lower() or 'time' in c.lower() or c.lower() in ('ds', 'period', 'month', 'week', 'day', 'tanggal')]\n"
    "        CFG.DATE_COL = named[0] if named else raw.columns[0]\n"
    "    if CFG.TARGET_COL == 'auto':\n"
    "        sample = load_table(SAMPLE_PATH) if SAMPLE_PATH else None\n"
    "        only_train = [c for c in raw.columns if raw_test is not None and c not in raw_test.columns]\n"
    "        if sample is not None and sample.shape[1] >= 2 and sample.columns[1] in raw.columns:\n"
    "            CFG.TARGET_COL = sample.columns[1]\n"
    "        elif len(only_train) == 1:\n"
    "            CFG.TARGET_COL = only_train[0]\n"
    "        else:\n"
    "            CFG.TARGET_COL = [c for c in raw.columns if c != CFG.DATE_COL and pd.api.types.is_numeric_dtype(raw[c])][-1]\n"
    "    if CFG.ID_COLS == 'auto':\n"
    "        CFG.ID_COLS = [c for c in raw.columns if c not in (CFG.DATE_COL, CFG.TARGET_COL) and not pd.api.types.is_numeric_dtype(raw[c]) and raw[c].dropna().astype(str).head(2000).str.split().str.len().mean() < 3]\n"
    "    print(f'Columns: date {CFG.DATE_COL!r}, target {CFG.TARGET_COL!r}, series ids {CFG.ID_COLS}')\n"
    "\n"
    "resolve_columns()\n"
    "D, Y = CFG.DATE_COL, CFG.TARGET_COL\n"
    "IDS = [c for c in CFG.ID_COLS if c in raw.columns]\n"
    "\n"
    "def prepare(frame):\n"
    "    frame = frame.copy()\n"
    "    frame[D] = pd.to_datetime(frame[D], errors='coerce')\n"
    "    frame = frame.dropna(subset=[D])\n"
    "    frame['series'] = frame[IDS].astype(str).agg(' | '.join, axis=1) if IDS else 'total'\n"
    "    return frame\n"
    "\n"
    "raw = prepare(raw)\n"
    "raw_test = prepare(raw_test) if raw_test is not None else None\n"
    "EXOG = [c for c in raw.columns if c not in IDS + [D, Y, 'series'] and pd.api.types.is_numeric_dtype(raw[c])] \\\n"
    "    if CFG.EXOG_COLS == 'auto' else [c for c in CFG.EXOG_COLS if c in raw.columns]\n"
    "\n"
    "dups = int(raw.duplicated(['series', D]).sum())\n"
    "agg_map = {Y: CFG.AGG, **{c: 'mean' for c in EXOG}}\n"
    "df = raw.groupby(['series', D], as_index=False).agg(agg_map) if dups else raw[['series', D, Y] + EXOG].copy()\n"
    "df = df.sort_values(['series', D]).reset_index(drop=True)\n"
    "print(f'Loaded {len(raw):,} rows from {DATA_PATH}; {dups:,} duplicate (series, date) rows aggregated with {CFG.AGG}')\n"
    "print(f'Series: {df.series.nunique():,} | id columns: {IDS} | exogenous: {EXOG}')\n"
    "if raw_test is not None:\n"
    "    print(f'Test frame: {len(raw_test):,} rows, {raw_test[D].min().date()} to {raw_test[D].max().date()}')\n"
    "display(df.head())"
))

# ---------------------------------------------------------------------------
# Section 3: Toolkit
# ---------------------------------------------------------------------------
cells.append(section("Toolkit", "3"))

cells.append(md(
    "The plotting identity matches the other templates. Date axes use concise automatic "
    "formatting, and `save_fig` writes every figure into `eda-output/figures/` with a running number."
))

cells.append(code(
    "PRIMARY   = '#3D5A80'\n"
    "ACCENT    = '#EE6C4D'\n"
    "SOFT      = '#98C1D9'\n"
    "DARK      = '#1B263B'\n"
    "MUTED     = '#8D99AE'\n"
    "PALETTE   = ['#3D5A80', '#EE6C4D', '#2A9D8F', '#E9C46A', '#9B5DE5',\n"
    "             '#F15BB5', '#00BBF9', '#8AB17D', '#E76F51', '#264653']\n"
    "CMAP_SEQ  = LinearSegmentedColormap.from_list('seq', ['#F7F9FC', '#98C1D9', '#3D5A80', '#1B263B'])\n"
    "CMAP_DIV  = LinearSegmentedColormap.from_list('div', ['#EE6C4D', '#FBE3DC', '#F7F7F7', '#D6E4F0', '#3D5A80'])\n"
    "CMAP_HEAT = LinearSegmentedColormap.from_list('heat', ['#FFF8F0', '#F4A261', '#E76F51', '#9D0208'])\n"
    "\n"
    "sns.set_theme(style='whitegrid', palette=PALETTE)\n"
    "mpl.rcParams.update({\n"
    "    'figure.dpi': 100, 'savefig.dpi': 200, 'savefig.bbox': 'tight',\n"
    "    'figure.facecolor': 'white', 'axes.facecolor': 'white',\n"
    "    'axes.spines.top': False, 'axes.spines.right': False,\n"
    "    'axes.edgecolor': '#C9D1DB', 'axes.labelcolor': DARK, 'axes.titleweight': 'bold',\n"
    "    'axes.titlesize': 12, 'axes.labelsize': 10, 'axes.titlecolor': DARK,\n"
    "    'xtick.color': '#4A5568', 'ytick.color': '#4A5568',\n"
    "    'grid.color': '#E2E8F0', 'grid.linewidth': 0.8, 'legend.frameon': False, 'font.size': 10,\n"
    "})\n"
    "\n"
    "FIG_COUNTER = [0]\n"
    "SAVED_FIGS = []\n"
    "\n"
    "def slug(text):\n"
    "    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')[:60]\n"
    "\n"
    "def save_fig(fig, name):\n"
    "    FIG_COUNTER[0] += 1\n"
    "    path = CFG.FIG_DIR / f'{FIG_COUNTER[0]:02d}_{slug(name)}.png'\n"
    "    fig.savefig(path, facecolor='white')\n"
    "    SAVED_FIGS.append(path.name)\n"
    "    plt.show()\n"
    "    plt.close(fig)\n"
    "\n"
    "def save_table(table, name, index=True):\n"
    "    table.to_csv(CFG.TABLE_DIR / f'{slug(name)}.csv', index=index)\n"
    "\n"
    "def suptitle(fig, title, subtitle=None, layout=True):\n"
    "    h = fig.get_figheight()\n"
    "    if layout:\n"
    "        fig.tight_layout(rect=(0, 0, 1, 1 - (0.85 if subtitle else 0.55) / h))\n"
    "    fig.text(0.01, 1 - 0.1 / h, title, ha='left', va='top', fontsize=15, fontweight='bold', color=DARK)\n"
    "    if subtitle:\n"
    "        fig.text(0.01, 1 - 0.42 / h, subtitle, ha='left', va='top', fontsize=10, color=MUTED)\n"
    "\n"
    "def date_axis(ax):\n"
    "    loc = mdates.AutoDateLocator()\n"
    "    ax.xaxis.set_major_locator(loc)\n"
    "    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))\n"
    "\n"
    "def bar_labels(ax, fmt='{:,.0f}', size=8):\n"
    "    for container in ax.containers:\n"
    "        if hasattr(container, 'datavalues'):\n"
    "            ax.bar_label(container, labels=[fmt.format(v) for v in container.datavalues], padding=3, fontsize=size, color='#4A5568')\n"
    "\n"
    "def short(value, k=28):\n"
    "    value = str(value)\n"
    "    return value if len(value) <= k else value[:k - 3] + '...'\n"
    "\n"
    "def note(msg):\n"
    "    display(Markdown(f'> **Note:** {msg}'))"
))

# ---------------------------------------------------------------------------
# Section 4: Structure and Frequency
# ---------------------------------------------------------------------------
cells.append(section("Structure and Frequency", "4"))

cells.append(md(
    "The frequency is the most fundamental decision in forecasting. It is inferred from the most "
    "common gap between consecutive timestamps within each series, because a few missing dates make "
    "`pd.infer_freq` fail. The frequency then sets the default seasonal periods and horizon. The "
    "series lifespan chart shows when each series starts and ends, which reveals new and "
    "discontinued series (cold-start cases)."
))

cells.append(code(
    "OFFSET_DEFAULTS = {'min': ([60, 1440], 60), 'h': ([24, 168], 48), 'D': ([7, 365], 28), 'W': ([52], 13),\n"
    "                   'MS': ([12], 12), 'M': ([12], 12), 'QS': ([4], 8), 'Q': ([4], 8), 'YS': ([1], 3), 'Y': ([1], 3)}\n"
    "\n"
    "def infer_freq(frame):\n"
    "    diffs = frame.groupby('series')[D].diff().dropna()\n"
    "    step = diffs.mode().iloc[0]\n"
    "    days = step / pd.Timedelta(days=1)\n"
    "    if days < 1 / 24:\n"
    "        return f'{int(step / pd.Timedelta(minutes=1))}min', 'min'\n"
    "    if days < 1:\n"
    "        hours = int(round(days * 24))\n"
    "        return ('h' if hours == 1 else f'{hours}h'), 'h'\n"
    "    if days < 6:\n"
    "        return 'D', 'D'\n"
    "    if days < 25:\n"
    "        dow = frame[D].dt.dayofweek.mode().iloc[0]\n"
    "        return f'W-{[\"MON\", \"TUE\", \"WED\", \"THU\", \"FRI\", \"SAT\", \"SUN\"][dow]}', 'W'\n"
    "    if days < 80:\n"
    "        return ('MS' if (frame[D].dt.day == 1).mean() > 0.8 else 'ME'), 'MS'\n"
    "    if days < 300:\n"
    "        return 'QS', 'QS'\n"
    "    return 'YS', 'YS'\n"
    "\n"
    "if CFG.FREQ == 'auto':\n"
    "    FREQ, FREQ_KEY = infer_freq(df)\n"
    "else:\n"
    "    FREQ = CFG.FREQ\n"
    "    FREQ_KEY = next((k for k in OFFSET_DEFAULTS if FREQ.upper().startswith(k.upper())), 'D')\n"
    "default_periods, default_h = OFFSET_DEFAULTS.get(FREQ_KEY, ([7], 14))\n"
    "\n"
    "life = df.groupby('series').agg(start=(D, 'min'), end=(D, 'max'), n_obs=(D, 'size'), total=(Y, 'sum'))\n"
    "full_index = pd.date_range(df[D].min(), df[D].max(), freq=FREQ)\n"
    "life['expected'] = [len(pd.date_range(s, e, freq=FREQ)) for s, e in zip(life['start'], life['end'])]\n"
    "life = life.sort_values('total', ascending=False)\n"
    "\n"
    "if CFG.HORIZON != 'auto':\n"
    "    HORIZON, H_SOURCE = int(CFG.HORIZON), 'user'\n"
    "elif raw_test is not None:\n"
    "    HORIZON, H_SOURCE = int(raw_test.groupby('series')[D].nunique().median()), 'test file'\n"
    "else:\n"
    "    HORIZON, H_SOURCE = default_h, 'frequency default'\n"
    "\n"
    "structure = pd.Series({\n"
    "    'frequency': FREQ, 'start': df[D].min().date(), 'end': df[D].max().date(), 'timestamps': len(full_index),\n"
    "    'series': df['series'].nunique(), 'observations': len(df), 'median series length': int(life['n_obs'].median()),\n"
    "    'shortest series': int(life['n_obs'].min()), 'series starting late': int((life['start'] > df[D].min()).sum()),\n"
    "    'series ending early': int((life['end'] < df[D].max()).sum()), 'horizon': f'{HORIZON} ({H_SOURCE})',\n"
    "}, name='value').to_frame()\n"
    "save_table(structure, 'structure')\n"
    "save_table(life, 'series_lifespans')\n"
    "display(structure)"
))

cells.append(md(
    "## Series Lifespans\n\n"
    "Each horizontal bar spans one series from its first to its last observation, coloured by "
    "volume. Late starts are new series with little history, which need a global model or "
    "hierarchical pooling. Early ends are discontinued series that should usually be excluded from "
    "training targets. The dashed line marks the forecast origin."
))

cells.append(code(
    "show = life.head(40)\n"
    "fig, ax = plt.subplots(figsize=(14, max(3.5, 0.26 * len(show) + 1.8)))\n"
    "norm = plt.Normalize(np.log1p(show['total']).min(), np.log1p(show['total']).max())\n"
    "for i, (name, r) in enumerate(show.iloc[::-1].iterrows()):\n"
    "    ax.plot([r['start'], r['end']], [i, i], lw=6, solid_capstyle='butt', color=CMAP_SEQ(0.35 + 0.65 * norm(np.log1p(r['total']))))\n"
    "    if r['n_obs'] < r['expected']:\n"
    "        ax.text(r['end'], i, f'  {r[\"expected\"] - r[\"n_obs\"]} gaps', va='center', fontsize=7, color=ACCENT)\n"
    "ax.set_yticks(range(len(show)))\n"
    "ax.set_yticklabels([short(s, 24) for s in show.index[::-1]], fontsize=8)\n"
    "ax.axvline(df[D].max(), color=ACCENT, ls='--', lw=1.2)\n"
    "date_axis(ax)\n"
    "extra = f', top 40 of {len(life)} by volume' if len(life) > 40 else ''\n"
    "suptitle(fig, 'Series lifespans', f'frequency {FREQ}, darker = higher total volume{extra}')\n"
    "save_fig(fig, 'series_lifespans')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* What is the frequency and the horizon? How many series exist and "
    "how long are they? Are there new or discontinued series that need special handling?"
))

# ---------------------------------------------------------------------------
# Section 5: Completeness and Intermittency
# ---------------------------------------------------------------------------
cells.append(section("Completeness and Intermittency", "5"))

cells.append(md(
    "The long table is reshaped into a panel (timestamps x series) on the full regular grid. A "
    "missing timestamp *inside* a series' lifespan is a gap, while timestamps before its first or "
    "after its last observation are simply outside its life. Gaps must be filled before lags are "
    "computed, and whether they mean 'zero' (no sales were recorded) or 'unknown' (sensor outage) "
    "is a domain decision that the zero share helps make."
))

cells.append(code(
    "panel = df.pivot(index=D, columns='series', values=Y).reindex(full_index)\n"
    "panel = panel[life.index]\n"
    "inside = pd.DataFrame({s: (full_index >= life.loc[s, 'start']) & (full_index <= life.loc[s, 'end']) for s in panel.columns},\n"
    "                      index=full_index)\n"
    "missing = panel.isna() & inside\n"
    "STAT_SERIES = life.index[:CFG.MAX_SERIES_STAT].tolist()\n"
    "\n"
    "def longest_run(mask):\n"
    "    best = run = 0\n"
    "    for v in mask:\n"
    "        run = run + 1 if v else 0\n"
    "        best = max(best, run)\n"
    "    return best\n"
    "\n"
    "comp = pd.DataFrame({\n"
    "    'length': inside.sum(),\n"
    "    'gaps': missing.sum(),\n"
    "    'gap_%': missing.sum() / inside.sum() * 100,\n"
    "    'longest_gap': [longest_run(missing[s].values) for s in panel.columns],\n"
    "    'zero_%': (panel == 0).sum() / panel.notna().sum() * 100,\n"
    "    'negative': (panel < 0).sum(),\n"
    "})\n"
    "save_table(comp, 'completeness')\n"
    "display(comp.describe().T.round(2))\n"
    "\n"
    "show = panel.columns[:40]\n"
    "state = np.where(~inside[show], 0, np.where(missing[show], 1, np.where(panel[show] == 0, 2, 3))).T\n"
    "step = max(1, len(full_index) // 400)\n"
    "state = state[:, ::step]\n"
    "fig, ax = plt.subplots(figsize=(15, max(3, 0.22 * len(show) + 1.8)))\n"
    "cmap = mpl.colors.ListedColormap(['#F7F9FC', ACCENT, '#E9C46A', PRIMARY])\n"
    "ax.imshow(state, aspect='auto', cmap=cmap, vmin=0, vmax=3, interpolation='nearest',\n"
    "          extent=[mdates.date2num(full_index[0]), mdates.date2num(full_index[-1]), len(show), 0])\n"
    "ax.set_yticks(np.arange(len(show)) + 0.5)\n"
    "ax.set_yticklabels([short(s, 22) for s in show], fontsize=7)\n"
    "ax.xaxis_date()\n"
    "date_axis(ax)\n"
    "ax.grid(False)\n"
    "handles = [mpl.patches.Patch(color=c, label=l) for c, l in zip(['#F7F9FC', ACCENT, '#E9C46A', PRIMARY], ['outside lifespan', 'missing', 'zero', 'positive'])]\n"
    "ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)\n"
    "suptitle(fig, 'Data presence map', f'{int(missing.values.sum()):,} gaps inside lifespans ({missing.values.sum() / inside.values.sum():.2%}); top {len(show)} series by volume')\n"
    "save_fig(fig, 'presence_map')"
))

cells.append(md(
    "## Intermittency\n\n"
    "Demand classification by Syntetos and Boylan (2005) uses the average demand interval (ADI, "
    "mean number of periods between non-zero values) and the squared coefficient of variation of "
    "non-zero sizes (CV squared). With cut-offs of 1.32 and 0.49 it separates smooth series "
    "(standard models work), erratic (volatile sizes), intermittent (many zeros, stable sizes, "
    "suited to Croston or TSB), and lumpy (both, the hardest)."
))

cells.append(code(
    "def adi_cv2(x):\n"
    "    x = x[~np.isnan(x)]\n"
    "    nz = np.flatnonzero(x > 0)\n"
    "    if len(nz) < 2:\n"
    "        return np.nan, np.nan\n"
    "    sizes = x[nz]\n"
    "    return len(x) / len(nz), (sizes.std() / sizes.mean()) ** 2\n"
    "\n"
    "rows = []\n"
    "for s in STAT_SERIES:\n"
    "    a, c = adi_cv2(panel[s][inside[s]].values)\n"
    "    cls = np.nan if np.isnan(a) else ('smooth' if a < 1.32 and c < 0.49 else 'erratic' if a < 1.32 else 'intermittent' if c < 0.49 else 'lumpy')\n"
    "    rows.append({'series': s, 'adi': a, 'cv2': c, 'demand_class': cls})\n"
    "intermit = pd.DataFrame(rows).set_index('series')\n"
    "comp = comp.join(intermit)\n"
    "save_table(intermit, 'intermittency')\n"
    "class_counts = intermit['demand_class'].value_counts()\n"
    "print('Demand classes:', class_counts.to_dict())\n"
    "\n"
    "cls_color = {'smooth': PALETTE[2], 'erratic': PALETTE[3], 'intermittent': PRIMARY, 'lumpy': ACCENT}\n"
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={'width_ratios': [1.5, 1]})\n"
    "ax = axes[0]\n"
    "for cl, g in intermit.dropna().groupby('demand_class'):\n"
    "    ax.scatter(g['adi'], g['cv2'], s=40, color=cls_color[cl], alpha=0.8, label=f'{cl} ({len(g)})', edgecolor='white')\n"
    "ax.axvline(1.32, color=MUTED, ls='--', lw=1)\n"
    "ax.axhline(0.49, color=MUTED, ls='--', lw=1)\n"
    "ax.set_xscale('log')\n"
    "ax.set_xlabel('ADI (average demand interval, log)')\n"
    "ax.set_ylabel('CV squared of non-zero sizes')\n"
    "ax.legend()\n"
    "ax.set_title('Syntetos-Boylan demand classification', loc='left')\n"
    "cc = class_counts.reindex(['smooth', 'erratic', 'intermittent', 'lumpy']).fillna(0)\n"
    "axes[1].bar(cc.index, cc.values, color=[cls_color[c] for c in cc.index])\n"
    "bar_labels(axes[1])\n"
    "axes[1].set_title('Series per class', loc='left')\n"
    "suptitle(fig, 'Intermittency', 'thresholds ADI = 1.32, CV^2 = 0.49')\n"
    "save_fig(fig, 'intermittency')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* How many gaps exist and how long are they? Should they be filled with "
    "zero or interpolated? What share of series is intermittent or lumpy and therefore needs "
    "zero-aware models (Croston, TSB, or Tweedie / Poisson loss in gradient boosting)?"
))

# ---------------------------------------------------------------------------
# Section 6: Target Distribution and Transform
# ---------------------------------------------------------------------------
cells.append(section("Target Distribution and Transform", "6"))

cells.append(md(
    "Forecast models assume roughly constant variance. If the spread of a series grows with its "
    "level, a variance-stabilising transform is needed. The slope of log(std) against log(mean) "
    "across series measures this: near 1 means a log transform, near 0.5 a square root, and near "
    "0 no transform (the Box-Cox lambda is approximately 1 minus the slope)."
))

cells.append(code(
    "stats_s = pd.DataFrame({'mean': panel[STAT_SERIES].mean(), 'std': panel[STAT_SERIES].std(),\n"
    "                        'skew': panel[STAT_SERIES].skew(), 'min': panel[STAT_SERIES].min()})\n"
    "ok = (stats_s['mean'] > 0) & (stats_s['std'] > 0)\n"
    "slope = np.polyfit(np.log(stats_s.loc[ok, 'mean']), np.log(stats_s.loc[ok, 'std']), 1)[0] if ok.sum() >= 3 else np.nan\n"
    "vals = df[Y].dropna()\n"
    "if vals.min() < 0:\n"
    "    TRANSFORM = 'none'\n"
    "elif not np.isnan(slope) and slope > 0.75:\n"
    "    TRANSFORM = 'log1p'\n"
    "elif not np.isnan(slope) and slope > 0.3:\n"
    "    TRANSFORM = 'sqrt'\n"
    "else:\n"
    "    TRANSFORM = 'log1p' if vals.skew() > 2 and vals.min() >= 0 else 'none'\n"
    "if ok.sum() < 3:\n"
    "    t_series = df.groupby(D)[Y].sum()\n"
    "    roll = pd.DataFrame({'m': t_series.rolling(default_periods[0]).mean(), 's': t_series.rolling(default_periods[0]).std()}).dropna()\n"
    "    roll = roll[(roll['m'] > 0) & (roll['s'] > 0)]\n"
    "    if len(roll) > 10:\n"
    "        slope = np.polyfit(np.log(roll['m']), np.log(roll['s']), 1)[0]\n"
    "print(f'Variance-level slope: {slope:.2f} -> recommended transform: {TRANSFORM}')\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(19, 5))\n"
    "sns.histplot(vals, bins=60, color=PRIMARY, edgecolor='white', ax=axes[0])\n"
    "axes[0].set_title(f'{Y} (skew {vals.skew():.2f}, zeros {(vals == 0).mean():.1%})', loc='left')\n"
    "if vals.min() >= 0:\n"
    "    sns.histplot(np.log1p(vals), bins=60, color=PALETTE[2], edgecolor='white', ax=axes[1])\n"
    "    axes[1].set_title(f'log1p({Y}) (skew {np.log1p(vals).skew():.2f})', loc='left')\n"
    "else:\n"
    "    axes[1].set_visible(False)\n"
    "if ok.sum() >= 3:\n"
    "    axes[2].scatter(stats_s.loc[ok, 'mean'], stats_s.loc[ok, 'std'], s=35, color=ACCENT, alpha=0.75, edgecolor='white')\n"
    "    xs = np.linspace(np.log(stats_s.loc[ok, 'mean']).min(), np.log(stats_s.loc[ok, 'mean']).max(), 50)\n"
    "    b = np.polyfit(np.log(stats_s.loc[ok, 'mean']), np.log(stats_s.loc[ok, 'std']), 1)\n"
    "    axes[2].plot(np.exp(xs), np.exp(b[1] + b[0] * xs), color=DARK, ls='--')\n"
    "    axes[2].set_xscale('log')\n"
    "    axes[2].set_yscale('log')\n"
    "    axes[2].set_xlabel('series mean (log)')\n"
    "    axes[2].set_ylabel('series std (log)')\n"
    "    axes[2].set_title(f'Variance vs level across series (slope {slope:.2f})', loc='left')\n"
    "else:\n"
    "    axes[2].set_visible(False)\n"
    "suptitle(fig, 'Target distribution and variance stabilisation', f'recommended transform: {TRANSFORM}')\n"
    "save_fig(fig, 'target_distribution')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Is the target skewed, zero-heavy, or negative? Which transform is "
    "recommended and why? Remember to invert it on the forecasts and to evaluate on the original scale."
))

# ---------------------------------------------------------------------------
# Section 7: Series Overview
# ---------------------------------------------------------------------------
cells.append(section("Series Overview", "7"))

cells.append(md(
    "The aggregate series shows the overall level, trend, and calendar spikes, with a rolling mean "
    "over the main seasonal period. The number of active series is drawn on a second axis, because a "
    "rising total can simply reflect new series rather than real growth. Small multiples of the "
    "largest series then reveal how much individual series differ from the aggregate."
))

cells.append(code(
    "total = panel.sum(axis=1, min_count=1)\n"
    "active = inside.sum(axis=1)\n"
    "m0 = default_periods[0]\n"
    "fig, ax = plt.subplots(figsize=(16, 5))\n"
    "ax.plot(total.index, total.values, color=SOFT, lw=0.8, label=Y)\n"
    "ax.plot(total.index, total.rolling(m0, center=True).mean(), color=PRIMARY, lw=2, label=f'rolling mean ({m0})')\n"
    "if len(total) > 6 * m0 * 4:\n"
    "    ax.plot(total.index, total.rolling(m0 * 13 if FREQ_KEY == 'D' else m0 * 4, center=True).mean(), color=ACCENT, lw=2, label='long rolling mean')\n"
    "ax2 = ax.twinx()\n"
    "ax2.step(active.index, active.values, color=MUTED, lw=1.2, where='post', alpha=0.8)\n"
    "ax2.set_ylabel('active series', color=MUTED)\n"
    "ax2.grid(False)\n"
    "ax.legend(loc='upper left')\n"
    "date_axis(ax)\n"
    "suptitle(fig, f'Aggregate {Y} over time', f'sum over {panel.shape[1]} series; grey step = number of active series')\n"
    "save_fig(fig, 'aggregate_series')\n"
    "\n"
    "top = panel.columns[:CFG.MAX_SERIES_PLOT]\n"
    "ncols = 3\n"
    "nrows = int(np.ceil(len(top) / ncols))\n"
    "fig, axes = plt.subplots(nrows, ncols, figsize=(18, 2.8 * nrows), squeeze=False, sharex=True)\n"
    "for ax in axes.ravel()[len(top):]:\n"
    "    ax.set_visible(False)\n"
    "for i, (ax, s) in enumerate(zip(axes.ravel(), top)):\n"
    "    x = panel[s]\n"
    "    ax.plot(x.index, x.values, color=PALETTE[i % 10], lw=0.6, alpha=0.6)\n"
    "    ax.plot(x.index, x.rolling(m0 * 4, center=True, min_periods=1).mean(), color=DARK, lw=1.4)\n"
    "    ax.set_title(short(s, 30), loc='left', fontsize=10)\n"
    "    date_axis(ax)\n"
    "suptitle(fig, f'Top {len(top)} series by volume', f'dark line = rolling mean over {m0 * 4} periods')\n"
    "save_fig(fig, 'top_series')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Is there an overall trend, and is it driven by growth or by new "
    "series? Are there visible spikes, drops, or regime changes? How heterogeneous are the series?"
))

# ---------------------------------------------------------------------------
# Section 8: Decomposition and Strength
# ---------------------------------------------------------------------------
cells.append(section("Decomposition and Strength", "8"))

cells.append(md(
    "STL (Seasonal-Trend decomposition using Loess, Cleveland et al., 1990) splits a series into "
    "trend, seasonal, and remainder components and is robust to outliers. The strength of trend and "
    "seasonality (Wang, Smith and Hyndman, 2006) is then measured per series on a 0 to 1 scale. "
    "Strong seasonality favours seasonal models and seasonal lags, while strong trend calls for "
    "detrending, differencing, or trend-aware models."
))

cells.append(md(
    "$$F_T = \\max\\left(0, 1 - \\frac{\\text{Var}(R)}{\\text{Var}(T + R)}\\right), \\qquad "
    "F_S = \\max\\left(0, 1 - \\frac{\\text{Var}(R)}{\\text{Var}(S + R)}\\right)$$"
))

cells.append(code(
    "def fill_series(x):\n"
    "    x = x.copy()\n"
    "    return x.interpolate(limit_direction='both') if x.notna().sum() > 1 else x.fillna(0)\n"
    "\n"
    "def tf(x):\n"
    "    return np.log1p(np.clip(x, 0, None)) if TRANSFORM == 'log1p' else np.sqrt(np.clip(x, 0, None)) if TRANSFORM == 'sqrt' else x\n"
    "\n"
    "M = m0 if len(total) >= 3 * m0 else max(2, len(total) // 3)\n"
    "agg = fill_series(total)\n"
    "stl = STL(tf(agg), period=M, robust=True).fit()\n"
    "\n"
    "fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)\n"
    "for ax, comp_, name, col in zip(axes, [tf(agg), stl.trend, stl.seasonal, stl.resid],\n"
    "                                ['observed' + (f' ({TRANSFORM})' if TRANSFORM != 'none' else ''), 'trend', f'seasonal (period {M})', 'remainder'],\n"
    "                                [SOFT, PRIMARY, PALETTE[2], ACCENT]):\n"
    "    if name == 'remainder':\n"
    "        ax.scatter(comp_.index, comp_.values, s=3, color=col)\n"
    "        ax.axhline(0, color=DARK, lw=0.8)\n"
    "    else:\n"
    "        ax.plot(comp_.index, comp_.values, color=col, lw=1 if name.startswith('obs') else 1.6)\n"
    "    ax.set_title(name, loc='left', fontsize=10)\n"
    "date_axis(axes[-1])\n"
    "suptitle(fig, 'STL decomposition of the aggregate series', f'robust STL, period {M}')\n"
    "save_fig(fig, 'stl_aggregate')\n"
    "\n"
    "def strength(x, period):\n"
    "    x = fill_series(x.dropna())\n"
    "    if len(x) < 2 * period + 1 or x.std() == 0:\n"
    "        return np.nan, np.nan\n"
    "    r = STL(tf(x), period=period, robust=True).fit()\n"
    "    vr = np.var(r.resid)\n"
    "    return max(0, 1 - vr / np.var(r.trend + r.resid)), max(0, 1 - vr / np.var(r.seasonal + r.resid))\n"
    "\n"
    "st = pd.DataFrame([strength(panel[s][inside[s]], M) for s in STAT_SERIES[:300]], index=STAT_SERIES[:300], columns=['trend_strength', 'seasonal_strength'])\n"
    "comp = comp.join(st)\n"
    "save_table(st, 'strength')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(9, 7))\n"
    "for cl, g in comp.loc[st.index].groupby('demand_class'):\n"
    "    ax.scatter(g['trend_strength'], g['seasonal_strength'], s=45, color=cls_color.get(cl, MUTED), alpha=0.8, edgecolor='white', label=cl)\n"
    "ax.scatter([1 - np.var(stl.resid) / np.var(stl.trend + stl.resid)], [1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid)],\n"
    "           marker='*', s=300, color=DARK, label='aggregate')\n"
    "ax.set_xlim(-0.02, 1.02)\n"
    "ax.set_ylim(-0.02, 1.02)\n"
    "ax.set_xlabel('trend strength F_T')\n"
    "ax.set_ylabel(f'seasonal strength F_S (period {M})')\n"
    "ax.legend()\n"
    "suptitle(fig, 'Trend vs seasonal strength per series', 'aggregates are usually smoother and more seasonal than individual series')\n"
    "save_fig(fig, 'strength_map')\n"
    "display(st.describe().T.round(3))"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* How strong are trend and seasonality, at the aggregate and per "
    "series? A big gap between them suggests forecasting at a higher level and reconciling, or "
    "using a global model that borrows strength across series."
))

# ---------------------------------------------------------------------------
# Section 9: Seasonal Patterns
# ---------------------------------------------------------------------------
cells.append(section("Seasonal Patterns", "9"))

cells.append(md(
    "Seasonal profiles show the typical shape within each cycle. Values are normalised by each "
    "series' mean before pooling, so large series do not dominate. The periodogram then finds the "
    "dominant cycle lengths directly from the data, which confirms the default periods or reveals "
    "unexpected ones (for example a 14-day payroll cycle)."
))

cells.append(code(
    "norm_long = (panel[STAT_SERIES] / panel[STAT_SERIES].mean()).stack().rename('y').reset_index()\n"
    "norm_long.columns = ['date', 'series', 'y']\n"
    "cal = {}\n"
    "if FREQ_KEY in ('min', 'h'):\n"
    "    cal['hour of day'] = norm_long['date'].dt.hour\n"
    "if FREQ_KEY in ('min', 'h', 'D'):\n"
    "    cal['day of week'] = norm_long['date'].dt.dayofweek\n"
    "if FREQ_KEY == 'D':\n"
    "    cal['day of month'] = norm_long['date'].dt.day\n"
    "if (df[D].max() - df[D].min()).days > 400:\n"
    "    cal['month'] = norm_long['date'].dt.month\n"
    "    if FREQ_KEY in ('D', 'W'):\n"
    "        cal['week of year'] = norm_long['date'].dt.isocalendar().week.astype(int)\n"
    "\n"
    "fig, axes = plt.subplots(1, len(cal), figsize=(5.4 * len(cal), 4.4), squeeze=False)\n"
    "PROFILES = {}\n"
    "for i, (ax, (name, key)) in enumerate(zip(axes[0], cal.items())):\n"
    "    prof = norm_long.groupby(key)['y'].agg(['mean', 'sem'])\n"
    "    PROFILES[name] = prof['mean']\n"
    "    ax.bar(prof.index, prof['mean'], yerr=1.96 * prof['sem'], color=PALETTE[i % 10], alpha=0.85,\n"
    "           error_kw=dict(ecolor=DARK, lw=0.8, capsize=2))\n"
    "    ax.axhline(1, color=DARK, ls='--', lw=1)\n"
    "    lo = max(0, prof['mean'].min() - 0.15)\n"
    "    ax.set_ylim(lo, prof['mean'].max() + 0.1)\n"
    "    ax.set_title(f'{name} (range {prof[\"mean\"].max() / max(prof[\"mean\"].min(), 1e-9):.2f}x)', loc='left')\n"
    "    if name == 'day of week':\n"
    "        ax.set_xticks(range(7))\n"
    "        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])\n"
    "suptitle(fig, 'Seasonal profiles', 'mean of series-normalised target (1 = series average), whiskers = 95% CI')\n"
    "save_fig(fig, 'seasonal_profiles')"
))

cells.append(md(
    "## Year-over-Year View\n\n"
    "Overlaying each year on a common calendar axis shows whether the seasonal shape is stable and "
    "whether it moves. Moving holidays such as Eid al-Fitr shift by about 11 days per year, which a "
    "fixed yearly seasonality cannot capture and which therefore need explicit holiday features. "
    "The heatmap shows the aggregate by year and month."
))

cells.append(code(
    "if (df[D].max() - df[D].min()).days > 400:\n"
    "    weekly_total = total.resample('W').sum(min_count=1) if FREQ_KEY in ('min', 'h', 'D') else total\n"
    "    fig, axes = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={'width_ratios': [1.6, 1]})\n"
    "    for i, (yr, g) in enumerate(weekly_total.groupby(weekly_total.index.year)):\n"
    "        doy = g.index.dayofyear\n"
    "        axes[0].plot(doy, g.values, color=PALETTE[i % 10], lw=1.8, label=str(yr), alpha=0.9)\n"
    "    axes[0].set_xlabel('day of year')\n"
    "    axes[0].legend(ncol=2, fontsize=8)\n"
    "    axes[0].set_title('Weekly aggregate by year', loc='left')\n"
    "    ym = total.groupby([total.index.year, total.index.month]).mean().unstack()\n"
    "    sns.heatmap(ym, cmap=CMAP_HEAT, ax=axes[1], linewidths=0.4, linecolor='white', cbar_kws={'shrink': 0.7, 'label': f'mean {Y}'})\n"
    "    axes[1].set_xlabel('month')\n"
    "    axes[1].set_ylabel('year')\n"
    "    axes[1].set_title('Mean aggregate by year and month', loc='left')\n"
    "    suptitle(fig, 'Year-over-year seasonality', 'misaligned peaks across years point to moving holidays')\n"
    "    save_fig(fig, 'year_over_year')\n"
    "else:\n"
    "    note('Less than about 13 months of data, so the year-over-year view is skipped.')"
))

cells.append(md(
    "## Periodogram\n\n"
    "The periodogram of the detrended aggregate measures how much variance each cycle length "
    "explains. The highest peaks (excluding very long cycles that are really trend) are the "
    "candidate seasonal periods. Final periods combine the frequency defaults with the detected "
    "peaks, keeping only periods that fit at least twice in the data."
))

cells.append(code(
    "x = (stl.seasonal + stl.resid).values\n"
    "freqs, power = signal.periodogram(x - x.mean(), detrend='linear')\n"
    "mask = freqs > 0\n"
    "periods_all, power = 1 / freqs[mask], power[mask]\n"
    "keep = periods_all <= len(x) / 2\n"
    "periods_all, power = periods_all[keep], power[keep]\n"
    "peaks, _ = signal.find_peaks(power)\n"
    "top_peaks = peaks[np.argsort(-power[peaks])][:6]\n"
    "PEAKS = pd.DataFrame({'period': periods_all[top_peaks], 'power_share': power[top_peaks] / power.sum()}).sort_values('power_share', ascending=False)\n"
    "\n"
    "if CFG.SEASONAL_PERIODS == 'auto':\n"
    "    SEASONAL_PERIODS = [p for p in default_periods if len(total) >= 2 * p]\n"
    "    for p in PEAKS['period'].head(3):\n"
    "        rp = int(round(p))\n"
    "        harmonic = any(abs(q / p - round(q / p)) < 0.05 for q in SEASONAL_PERIODS)\n"
    "        if rp >= 2 and not harmonic and all(abs(rp - q) / q > 0.1 for q in SEASONAL_PERIODS) and PEAKS.set_index('period').loc[p, 'power_share'] > 0.05 and len(total) >= 2 * rp:\n"
    "            SEASONAL_PERIODS.append(rp)\n"
    "else:\n"
    "    SEASONAL_PERIODS = list(CFG.SEASONAL_PERIODS)\n"
    "SEASONAL_PERIODS = sorted(set(SEASONAL_PERIODS)) or [2]\n"
    "save_table(PEAKS, 'periodogram_peaks', index=False)\n"
    "display(PEAKS.round(4))\n"
    "print(f'Seasonal periods: {SEASONAL_PERIODS}')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 4.6))\n"
    "ax.semilogx(periods_all, power, color=PRIMARY, lw=1.2)\n"
    "for p in top_peaks:\n"
    "    ax.annotate(f'{periods_all[p]:.1f}', (periods_all[p], power[p]), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8, color=ACCENT)\n"
    "for p in SEASONAL_PERIODS:\n"
    "    ax.axvline(p, color=ACCENT, ls='--', lw=1, alpha=0.7)\n"
    "ax.set_xlabel('period (log scale, in time steps)')\n"
    "ax.set_ylabel('power')\n"
    "suptitle(fig, 'Periodogram of the detrended aggregate', f'dashed = selected seasonal periods {SEASONAL_PERIODS}')\n"
    "save_fig(fig, 'periodogram')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Which calendar effects are strong (weekday, month, hour)? Which "
    "seasonal periods were selected, and does the year-over-year view show moving holidays that "
    "need explicit event features?"
))

# ---------------------------------------------------------------------------
# Section 10: Autocorrelation and Lags
# ---------------------------------------------------------------------------
cells.append(section("Autocorrelation and Lags", "10"))

cells.append(md(
    "The autocorrelation function (ACF) shows how strongly a value relates to its own past at each "
    "lag, and the partial autocorrelation (PACF) removes the effect of intermediate lags. "
    "Significant lags become lag features for machine learning models, and the ACF averaged over "
    "series tells whether that pattern is shared. For direct multi-step forecasting, only lags of at "
    "least the horizon are usable without recursion, so both sets are reported."
))

cells.append(code(
    "MAX_LAGS = int(CFG.MAX_LAGS) if CFG.MAX_LAGS != 'auto' else int(min(len(total) // 3, max(3 * max(SEASONAL_PERIODS[0], 2), 2 * HORIZON, 30)))\n"
    "y_agg = tf(agg)\n"
    "acf_vals = acf(y_agg, nlags=MAX_LAGS, fft=True)\n"
    "pacf_vals = pacf(y_agg, nlags=min(MAX_LAGS, len(y_agg) // 2 - 1), method='ywm')\n"
    "conf = 1.96 / np.sqrt(len(y_agg))\n"
    "\n"
    "series_acf = []\n"
    "for s in STAT_SERIES[:100]:\n"
    "    xs = fill_series(panel[s][inside[s]])\n"
    "    if len(xs) > MAX_LAGS + 10 and xs.std() > 0:\n"
    "        series_acf.append(acf(tf(xs), nlags=MAX_LAGS, fft=True))\n"
    "mean_acf = np.nanmean(series_acf, axis=0) if series_acf else acf_vals\n"
    "\n"
    "lags = np.arange(len(acf_vals))\n"
    "sig = [int(l) for l in np.argsort(-np.abs(mean_acf[1:]))[:15] + 1 if abs(mean_acf[l]) > conf]\n"
    "LAGS_ALL = sorted(set(sig + [p for p in SEASONAL_PERIODS if p <= MAX_LAGS]))\n"
    "LAGS_DIRECT = sorted({l for l in LAGS_ALL if l >= HORIZON} | {int(np.ceil(HORIZON / p) * p) for p in SEASONAL_PERIODS if p <= 4 * HORIZON} | {p for p in SEASONAL_PERIODS if p >= HORIZON})\n"
    "base_w = sorted({w for w in [SEASONAL_PERIODS[0], 2 * SEASONAL_PERIODS[0], 4 * SEASONAL_PERIODS[0]] + ([SEASONAL_PERIODS[-1]] if SEASONAL_PERIODS[-1] > 4 * SEASONAL_PERIODS[0] else []) if w < len(total) // 2})\n"
    "ROLL_WINDOWS = base_w\n"
    "print(f'Significant lags (mean ACF): {sorted(sig)}')\n"
    "print(f'Lag features (recursive / one-step): {LAGS_ALL}')\n"
    "print(f'Lag features (direct, >= horizon {HORIZON}): {LAGS_DIRECT}')\n"
    "print(f'Rolling windows: {ROLL_WINDOWS}')\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(18, 4.8))\n"
    "axes[0].vlines(lags[1:], 0, acf_vals[1:], color=PRIMARY, lw=1.2, label='aggregate')\n"
    "axes[0].plot(lags[1:], mean_acf[1:], color=ACCENT, lw=1.8, label=f'mean over {len(series_acf)} series')\n"
    "axes[0].fill_between(lags, -conf, conf, color=MUTED, alpha=0.2)\n"
    "for p in SEASONAL_PERIODS:\n"
    "    if p <= MAX_LAGS:\n"
    "        axes[0].axvline(p, color=PALETTE[2], ls='--', lw=1)\n"
    "axes[0].axvline(HORIZON, color=DARK, ls=':', lw=1.2, label=f'horizon {HORIZON}')\n"
    "axes[0].legend(fontsize=8)\n"
    "axes[0].set_title('Autocorrelation (ACF)', loc='left')\n"
    "axes[0].set_xlabel('lag')\n"
    "pl = np.arange(len(pacf_vals))\n"
    "axes[1].vlines(pl[1:], 0, pacf_vals[1:], color=PALETTE[2], lw=1.4)\n"
    "axes[1].fill_between(pl, -conf, conf, color=MUTED, alpha=0.2)\n"
    "axes[1].set_title('Partial autocorrelation (PACF) of the aggregate', loc='left')\n"
    "axes[1].set_xlabel('lag')\n"
    "suptitle(fig, 'Autocorrelation structure', f'shaded = 95% band; green dashed = seasonal periods {SEASONAL_PERIODS}')\n"
    "save_fig(fig, 'acf_pacf')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* How persistent is the series (slow ACF decay means trend or "
    "non-stationarity)? Which lags are significant, and which are usable at the forecast horizon? "
    "A PACF that cuts off after p lags suggests an AR(p) component."
))

# ---------------------------------------------------------------------------
# Section 11: Stationarity
# ---------------------------------------------------------------------------
cells.append(section("Stationarity", "11"))

cells.append(md(
    "Two complementary tests are used because each has the opposite null hypothesis. The Augmented "
    "Dickey-Fuller (ADF) test assumes a unit root (non-stationary), and the KPSS test assumes "
    "stationarity. A series is called stationary when ADF rejects (p < 0.05) and KPSS does not "
    "(p >= 0.05). The tests are repeated after a log transform, a first difference, and a seasonal "
    "difference, which gives the differencing order an ARIMA model would need."
))

cells.append(code(
    "def stationarity(x):\n"
    "    x = pd.Series(x).dropna()\n"
    "    if len(x) < 20 or x.std() == 0:\n"
    "        return np.nan, np.nan\n"
    "    adf_p = adfuller(x, autolag='AIC')[1]\n"
    "    kpss_p = kpss(x, regression='c', nlags='auto')[1]\n"
    "    return adf_p, kpss_p\n"
    "\n"
    "m_s = SEASONAL_PERIODS[0]\n"
    "variants = {'level': agg, f'{TRANSFORM}': tf(agg), 'first difference': tf(agg).diff(),\n"
    "            f'seasonal difference ({m_s})': tf(agg).diff(m_s), f'seasonal + first difference': tf(agg).diff(m_s).diff()}\n"
    "rows = []\n"
    "for name, x in variants.items():\n"
    "    a, k = stationarity(x)\n"
    "    rows.append({'variant': name, 'adf_p': a, 'kpss_p': k, 'stationary': bool(a < 0.05 and k >= 0.05)})\n"
    "stat_table = pd.DataFrame(rows).set_index('variant')\n"
    "save_table(stat_table, 'stationarity_aggregate')\n"
    "display(stat_table.round(4))\n"
    "\n"
    "def n_diffs(x, max_d=2):\n"
    "    for d in range(max_d + 1):\n"
    "        a, k = stationarity(x)\n"
    "        if a < 0.05 and k >= 0.05:\n"
    "            return d\n"
    "        x = pd.Series(x).diff().dropna()\n"
    "    return max_d\n"
    "\n"
    "rows = []\n"
    "for s in STAT_SERIES[:60]:\n"
    "    xs = tf(fill_series(panel[s][inside[s]]))\n"
    "    a, k = stationarity(xs)\n"
    "    rows.append({'series': s, 'adf_p': a, 'kpss_p': k, 'd': n_diffs(xs)})\n"
    "stat_series = pd.DataFrame(rows).set_index('series')\n"
    "save_table(stat_series, 'stationarity_series')\n"
    "D_ORDER = int(stat_series['d'].median()) if len(stat_series) else n_diffs(tf(agg))\n"
    "print(f'Series stationary at level: {(stat_series[\"d\"] == 0).mean():.0%} | median differencing order d = {D_ORDER}')\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(18, 4.6), gridspec_kw={'width_ratios': [2, 1]})\n"
    "w = max(m_s * 4, 8)\n"
    "axes[0].plot(agg.index, tf(agg).rolling(w).mean(), color=PRIMARY, lw=2, label=f'rolling mean ({w})')\n"
    "ax2 = axes[0].twinx()\n"
    "ax2.plot(agg.index, tf(agg).rolling(w).std(), color=ACCENT, lw=1.6, label=f'rolling std ({w})')\n"
    "ax2.grid(False)\n"
    "ax2.set_ylabel('rolling std', color=ACCENT)\n"
    "axes[0].set_ylabel('rolling mean', color=PRIMARY)\n"
    "axes[0].set_title('Rolling mean and std of the (transformed) aggregate', loc='left')\n"
    "date_axis(axes[0])\n"
    "dc = stat_series['d'].value_counts().sort_index()\n"
    "axes[1].bar([f'd = {i}' for i in dc.index], dc.values, color=PALETTE[2])\n"
    "bar_labels(axes[1])\n"
    "axes[1].set_title('Differencing order needed per series', loc='left')\n"
    "suptitle(fig, 'Stationarity', 'drifting mean or changing spread = non-stationary')\n"
    "save_fig(fig, 'stationarity')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Is the aggregate stationary after the transform, a first difference, "
    "or a seasonal difference? For ARIMA this sets d and D. For tree models, a non-stationary level "
    "argues for predicting differences or ratios to a recent rolling mean rather than raw levels."
))

# ---------------------------------------------------------------------------
# Section 12: Anomalies and Level Shifts
# ---------------------------------------------------------------------------
cells.append(section("Anomalies and Level Shifts", "12"))

cells.append(md(
    "Point anomalies are detected on robust STL remainders: a value is flagged when its remainder "
    "is more than `OUTLIER_Z` robust standard deviations (1.4826 times the median absolute "
    "deviation, floored at the remainder standard deviation) from the median. Intermittent and "
    "lumpy series are skipped, because their zeros make any deviation look extreme. Level shifts are "
    "detected by comparing the median of a window "
    "before and after each point. A candidate must also persist over a window three times longer, "
    "which filters out temporary spikes such as holidays. A persistent jump means older history may be misleading, so "
    "training could start after the break or include a regime indicator."
))

cells.append(code(
    "rows, shifts = [], []\n"
    "w_shift = max(4 * SEASONAL_PERIODS[0], 28 if FREQ_KEY == 'D' else 8)\n"
    "smooth_series = [s for s in STAT_SERIES[:200] if comp.loc[s, 'demand_class'] in ('smooth', 'erratic')]\n"
    "for s in smooth_series:\n"
    "    raw_s = panel[s][inside[s]]\n"
    "    xs = fill_series(raw_s)\n"
    "    if len(xs) < 2 * M + 1 or xs.std() == 0:\n"
    "        continue\n"
    "    r = STL(tf(xs), period=M, robust=True).fit().resid\n"
    "    mad = max(1.4826 * np.median(np.abs(r - np.median(r))), r.std(), 1e-9)\n"
    "    z = (r - np.median(r)) / mad\n"
    "    for d_, zz in z[(np.abs(z) > CFG.OUTLIER_Z) & raw_s.notna()].items():\n"
    "        rows.append({'series': s, 'date': d_, 'value': raw_s[d_], 'robust_z': zz})\n"
    "    lv = tf(xs).rolling(SEASONAL_PERIODS[0], center=True, min_periods=1).mean()\n"
    "    before = lv.rolling(w_shift).median()\n"
    "    after = lv[::-1].rolling(w_shift).median()[::-1]\n"
    "    scale = max(1.4826 * (lv - lv.rolling(w_shift, center=True).median()).abs().median(), 0.25 * lv.std(), 1e-9)\n"
    "    jump = ((after.shift(-1) - before) / scale).dropna()\n"
    "    if len(jump):\n"
    "        t = jump.abs().idxmax()\n"
    "        b3 = lv[:t].tail(3 * w_shift).median()\n"
    "        a3 = lv[t:].iloc[1:3 * w_shift + 1].median()\n"
    "        shifts.append({'series': s, 'date': t, 'shift_in_mads': jump[t], 'persistent_shift': (a3 - b3) / scale})\n"
    "anoms = pd.DataFrame(rows, columns=['series', 'date', 'value', 'robust_z'])\n"
    "shift_df = pd.DataFrame(shifts, columns=['series', 'date', 'shift_in_mads', 'persistent_shift']).sort_values('shift_in_mads', key=np.abs, ascending=False)\n"
    "LEVEL_SHIFTS = shift_df[(shift_df['shift_in_mads'].abs() > 4) & (shift_df['persistent_shift'].abs() > 6)]\n"
    "save_table(anoms, 'anomalies', index=False)\n"
    "save_table(shift_df, 'level_shift_candidates', index=False)\n"
    "print(f'Point anomalies (|z| > {CFG.OUTLIER_Z}): {len(anoms):,} in {anoms[\"series\"].nunique()} series')\n"
    "print(f'Level shifts (> 4 MADs, persistent > 6): {len(LEVEL_SHIFTS)}')\n"
    "display(LEVEL_SHIFTS.head(10))\n"
    "\n"
    "pick = list(anoms['series'].value_counts().index[:2]) + list(LEVEL_SHIFTS['series'].head(2))\n"
    "pick = list(dict.fromkeys(pick))[:4]\n"
    "if pick:\n"
    "    fig, axes = plt.subplots(len(pick), 1, figsize=(16, 2.9 * len(pick)), squeeze=False)\n"
    "    for ax, s in zip(axes[:, 0], pick):\n"
    "        x = panel[s]\n"
    "        ax.plot(x.index, x.values, color=SOFT, lw=0.8)\n"
    "        ax.plot(x.index, x.rolling(w_shift, center=True, min_periods=1).median(), color=PRIMARY, lw=1.6)\n"
    "        a = anoms[anoms['series'] == s]\n"
    "        ax.scatter(a['date'], a['value'], color=ACCENT, s=30, zorder=3, label='anomaly')\n"
    "        sh = LEVEL_SHIFTS[LEVEL_SHIFTS['series'] == s]\n"
    "        for d_ in sh['date']:\n"
    "            ax.axvline(d_, color=DARK, ls='--', lw=1.4, label='level shift')\n"
    "        ax.set_title(short(s, 40), loc='left', fontsize=10)\n"
    "        ax.legend(loc='upper left', fontsize=8)\n"
    "        date_axis(ax)\n"
    "    suptitle(fig, 'Anomalies and level shifts', f'orange = STL remainder |z| > {CFG.OUTLIER_Z}, dashed = largest level shift')\n"
    "    save_fig(fig, 'anomalies_level_shifts')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Are anomalies data errors (to clip or remove) or real events (to keep "
    "and explain with features)? Do level shifts suggest restricting the training window or adding a "
    "regime flag?"
))

# ---------------------------------------------------------------------------
# Section 13: Exogenous and Calendar Drivers
# ---------------------------------------------------------------------------
cells.append(section("Exogenous and Calendar Drivers", "13"))

cells.append(md(
    "Exogenous variables are only useful for forecasting if their future values are known (planned "
    "prices and promotions) or if they lead the target. Each covariate is therefore checked for "
    "presence in the test frame, its within-series correlation with the target at several lags, "
    "and, for binary flags, the uplift ratio. The largest positive residuals of the aggregate STL "
    "decomposition are listed as candidate special days, which usually reveal holidays that need "
    "explicit event features."
))

cells.append(code(
    "EXOG_KNOWN = [c for c in EXOG if raw_test is not None and c in raw_test.columns]\n"
    "rows = []\n"
    "ccf_lags = list(range(0, min(3 * SEASONAL_PERIODS[0], 30) + 1))\n"
    "CCF = {}\n"
    "for c in EXOG:\n"
    "    ex = df.pivot(index=D, columns='series', values=c).reindex(full_index)[STAT_SERIES[:100]]\n"
    "    yy = tf(panel[STAT_SERIES[:100]])\n"
    "    exd, yd = ex - ex.mean(), yy - yy.mean()\n"
    "    ccf = [pd.concat([exd.shift(l).stack(), yd.stack()], axis=1).dropna().corr().iloc[0, 1] for l in ccf_lags]\n"
    "    CCF[c] = ccf\n"
    "    row = {'feature': c, 'known_in_future': c in EXOG_KNOWN, 'corr_lag0': ccf[0],\n"
    "           'best_lag': ccf_lags[int(np.nanargmax(np.abs(ccf)))], 'best_corr': ccf[int(np.nanargmax(np.abs(ccf)))]}\n"
    "    vals_c = df[c].dropna().unique()\n"
    "    if set(np.unique(vals_c)) <= {0, 1}:\n"
    "        g = df.groupby(['series', c])[Y].mean().unstack()\n"
    "        row['uplift_ratio'] = float((g[1] / g[0]).replace([np.inf, -np.inf], np.nan).median()) if 1 in g and 0 in g else np.nan\n"
    "    elif (df[c] > 0).all():\n"
    "        el = []\n"
    "        for s, g in df[df[Y] > 0].groupby('series'):\n"
    "            if g[c].std() > 0 and len(g) > 30:\n"
    "                el.append(np.polyfit(np.log(g[c]), np.log(g[Y]), 1)[0])\n"
    "        row['elasticity'] = float(np.median(el)) if el else np.nan\n"
    "    rows.append(row)\n"
    "exog_table = pd.DataFrame(rows).set_index('feature') if rows else pd.DataFrame()\n"
    "if not exog_table.empty:\n"
    "    save_table(exog_table, 'exogenous')\n"
    "    display(exog_table.round(3))\n"
    "\n"
    "resid_agg = pd.Series(stl.resid, index=agg.index)\n"
    "SPECIAL = resid_agg.nlargest(12)\n"
    "special_tbl = pd.DataFrame({'date': SPECIAL.index.date, 'weekday': SPECIAL.index.day_name(), 'residual': SPECIAL.values})\n"
    "save_table(special_tbl, 'special_day_candidates', index=False)\n"
    "\n"
    "ncols = 2 if EXOG else 1\n"
    "fig, axes = plt.subplots(1, ncols, figsize=(8.5 * ncols, 4.8), squeeze=False)\n"
    "if EXOG:\n"
    "    for i, (c, v) in enumerate(CCF.items()):\n"
    "        axes[0, 0].plot(ccf_lags, v, marker='o', ms=3, lw=1.8, color=PALETTE[i % 10],\n"
    "                        label=f'{c}' + (' (known future)' if c in EXOG_KNOWN else ' (past only)'))\n"
    "    axes[0, 0].axhline(0, color=DARK, lw=0.8)\n"
    "    axes[0, 0].set_xlabel('lag of covariate (periods)')\n"
    "    axes[0, 0].set_ylabel('correlation with target')\n"
    "    axes[0, 0].legend(fontsize=8)\n"
    "    axes[0, 0].set_title('Within-series cross-correlation', loc='left')\n"
    "ax = axes[0, -1]\n"
    "ax.plot(resid_agg.index, resid_agg.values, color=SOFT, lw=0.7)\n"
    "ax.scatter(SPECIAL.index, SPECIAL.values, color=ACCENT, s=30, zorder=3)\n"
    "for d_, v in SPECIAL.head(6).items():\n"
    "    ax.annotate(d_.strftime('%Y-%m-%d'), (d_, v), fontsize=7, xytext=(3, 3), textcoords='offset points', color=DARK)\n"
    "date_axis(ax)\n"
    "ax.set_title('Aggregate STL remainder: candidate special days', loc='left')\n"
    "suptitle(fig, 'Drivers of the target', 'use known-future covariates directly; past-only ones only as lags >= horizon')\n"
    "save_fig(fig, 'drivers')\n"
    "display(special_tbl)"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Which covariates matter, and are they known for the forecast period? "
    "What are the promotion uplift and price elasticity? Do the candidate special days match known "
    "holidays (Lebaran, Christmas, payday), and should they become event features?"
))

# ---------------------------------------------------------------------------
# Section 14: Cross-Series Structure
# ---------------------------------------------------------------------------
cells.append(section("Cross-Series Structure", "14"))

cells.append(md(
    "With many series, the key design question is local models (one per series) versus a single "
    "global model trained on all series. Highly correlated series and a heavy-tailed volume "
    "distribution both favour a global model, while the Pareto curve shows how much of the total "
    "the top series carry, which matters for volume-weighted metrics such as WAPE. When several ID "
    "columns exist, the volume by each level outlines the hierarchy."
))

cells.append(code(
    "share = life['total'] / life['total'].sum()\n"
    "cum = share.cumsum().values\n"
    "n80 = int(np.searchsorted(cum, 0.8) + 1)\n"
    "top = panel.columns[:min(20, panel.shape[1])]\n"
    "resampled = panel[top].resample('W').sum(min_count=1) if FREQ_KEY in ('min', 'h', 'D') else panel[top]\n"
    "corr = np.log1p(resampled.clip(lower=0)).diff().corr() if len(top) > 1 else pd.DataFrame()\n"
    "MEAN_CORR = float(corr.values[np.triu_indices_from(corr.values, 1)].mean()) if len(top) > 2 else np.nan\n"
    "\n"
    "n_extra = len(IDS) if len(IDS) > 1 else 0\n"
    "fig, axes = plt.subplots(1, 2 + n_extra, figsize=(7 * (2 + n_extra), 5.6), squeeze=False)\n"
    "ax = axes[0, 0]\n"
    "ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum * 100, color=PRIMARY, lw=2.4)\n"
    "ax.axhline(80, color=MUTED, ls='--', lw=1)\n"
    "ax.axvline(n80 / len(cum) * 100, color=ACCENT, ls='--', lw=1)\n"
    "ax.set_xlabel('% of series (largest first)')\n"
    "ax.set_ylabel('% of total volume')\n"
    "ax.set_title(f'Pareto: {n80} series ({n80 / len(cum):.0%}) carry 80% of volume', loc='left')\n"
    "if len(top) > 1:\n"
    "    sns.heatmap(corr, cmap=CMAP_DIV, vmin=-1, vmax=1, center=0, ax=axes[0, 1], square=True,\n"
    "                xticklabels=[short(c, 12) for c in corr.columns], yticklabels=[short(c, 12) for c in corr.index], cbar_kws={'shrink': 0.6})\n"
    "    axes[0, 1].tick_params(labelsize=7)\n"
    "    axes[0, 1].set_title(f'Correlation of weekly log-changes (mean {MEAN_CORR:.2f})', loc='left')\n"
    "else:\n"
    "    axes[0, 1].set_visible(False)\n"
    "for j, col in enumerate(IDS if n_extra else []):\n"
    "    lvl = raw.groupby(col)[Y].sum().sort_values()\n"
    "    axes[0, 2 + j].barh([short(i, 18) for i in lvl.index[-20:]], lvl.values[-20:], color=PALETTE[(j + 2) % 10])\n"
    "    axes[0, 2 + j].set_title(f'Volume by {col}', loc='left')\n"
    "suptitle(fig, 'Cross-series structure', f'{panel.shape[1]} series')\n"
    "save_fig(fig, 'cross_series')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Are series strongly co-moving (a shared driver that a global model "
    "can learn)? How concentrated is the volume? Should forecasting happen at the bottom level, at "
    "an aggregate level, or both with reconciliation?"
))

# ---------------------------------------------------------------------------
# Section 15: Baseline Forecastability
# ---------------------------------------------------------------------------
cells.append(section("Baseline Forecastability", "15"))

cells.append(md(
    "Before modelling, simple benchmarks are backtested on the last `HORIZON` periods of every "
    "series: the naive forecast (repeat the last value), the seasonal naive (repeat the last "
    "season), and the moving average of the last season. They are scored with MASE (error scaled by "
    "the in-sample seasonal naive error, below 1 means better than seasonal naive) and WAPE (total "
    "absolute error over total actuals, robust to zeros). Spectral entropy measures how much "
    "structure a series has: lower entropy means a more forecastable series (Goerg, 2013)."
))

cells.append(md(
    "$$\\text{MASE} = \\frac{\\frac{1}{h}\\sum_{t=1}^{h}|y_t - \\hat{y}_t|}{\\frac{1}{n-m}\\sum_{t=m+1}^{n}|y_t - y_{t-m}|} \\qquad "
    "\\text{WAPE} = \\frac{\\sum |y_t - \\hat{y}_t|}{\\sum |y_t|}$$"
))

cells.append(code(
    "def spectral_entropy(x):\n"
    "    x = np.asarray(x, dtype=float)\n"
    "    if len(x) < 16 or x.std() == 0:\n"
    "        return np.nan\n"
    "    _, p = signal.welch(x - x.mean(), nperseg=min(256, len(x)))\n"
    "    p = p / p.sum()\n"
    "    p = p[p > 0]\n"
    "    return float(-(p * np.log(p)).sum() / np.log(len(p)))\n"
    "\n"
    "H, m = HORIZON, SEASONAL_PERIODS[0]\n"
    "rows, err_tot = [], {'naive': 0.0, 'seasonal_naive': 0.0, 'moving_average': 0.0}\n"
    "act_tot = 0.0\n"
    "for s in STAT_SERIES[:300]:\n"
    "    xs = fill_series(panel[s][inside[s]]).values\n"
    "    if len(xs) < H + 2 * m + 5:\n"
    "        continue\n"
    "    train_, test_ = xs[:-H], xs[-H:]\n"
    "    scale = np.mean(np.abs(train_[m:] - train_[:-m])) + 1e-9\n"
    "    fc = {'naive': np.repeat(train_[-1], H),\n"
    "          'seasonal_naive': np.resize(train_[-m:], H),\n"
    "          'moving_average': np.repeat(train_[-m:].mean(), H)}\n"
    "    row = {'series': s, 'entropy': spectral_entropy(tf(train_))}\n"
    "    for k, f in fc.items():\n"
    "        e = np.abs(test_ - f)\n"
    "        row[f'mase_{k}'] = e.mean() / scale\n"
    "        err_tot[k] += e.sum()\n"
    "    act_tot += np.abs(test_).sum()\n"
    "    rows.append(row)\n"
    "bench = pd.DataFrame(rows).set_index('series')\n"
    "save_table(bench, 'baseline_backtest')\n"
    "summary_bench = pd.DataFrame({'median_MASE': [bench[f'mase_{k}'].median() for k in err_tot],\n"
    "                              'WAPE': [err_tot[k] / max(act_tot, 1e-9) for k in err_tot]}, index=list(err_tot))\n"
    "BEST_BENCH = summary_bench['WAPE'].idxmin()\n"
    "display(summary_bench.round(4))\n"
    "print(f'Best naive benchmark: {BEST_BENCH} (WAPE {summary_bench.loc[BEST_BENCH, \"WAPE\"]:.3f}) -> the score any model must beat')\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(17, 5))\n"
    "axes[0].bar(summary_bench.index, summary_bench['WAPE'], color=[ACCENT if k == BEST_BENCH else PRIMARY for k in summary_bench.index])\n"
    "bar_labels(axes[0], '{:.3f}')\n"
    "axes[0].set_title(f'WAPE of naive benchmarks on the last {H} periods', loc='left')\n"
    "cls_map = comp['demand_class'].reindex(bench.index)\n"
    "for cl in cls_map.dropna().unique():\n"
    "    g = bench[cls_map == cl]\n"
    "    axes[1].scatter(g['entropy'], g[f'mase_{BEST_BENCH}'], s=40, alpha=0.8, color=cls_color.get(cl, MUTED), edgecolor='white', label=cl)\n"
    "axes[1].axhline(1, color=MUTED, ls='--', lw=1)\n"
    "axes[1].set_xlabel('spectral entropy (lower = more forecastable)')\n"
    "axes[1].set_ylabel(f'MASE of {BEST_BENCH}')\n"
    "axes[1].legend(fontsize=8)\n"
    "axes[1].set_title('Forecastability vs benchmark error per series', loc='left')\n"
    "suptitle(fig, 'Baseline forecastability', f'{len(bench)} series backtested, horizon {H}, season {m}')\n"
    "save_fig(fig, 'baseline_forecastability')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Which naive benchmark is best, and what WAPE must a model beat? "
    "Which series are inherently hard (high entropy), and should they get simpler models or be "
    "aggregated?"
))

# ---------------------------------------------------------------------------
# Section 16: EDA Summary and Pipeline Decisions
# ---------------------------------------------------------------------------
cells.append(section("Text Signal", "16"))

cells.append(md(
    "Text can only help a forecast through what was already published when the forecast is made. "
    "This section loads the documents (a text column of the main table, or a separate dated text "
    "file found in the dataset folder), embeds them with TF-IDF and SVD, aggregates them per series "
    "and period, and correlates each text feature at time t - lag with the detrended target at "
    "time t. Lags inside the forecast horizon are unusable, because that text does not exist yet at "
    "the forecast origin, so the recommended `text_lag` is the strongest lag at or beyond the horizon. "
    "The noise level 2/sqrt(n) separates a real lead from chance."
))

cells.append(code(
    "TEXT_NAME = re.compile(r'(text|news|review|tweet|post|comment|article|doc|berita|ulasan|komentar)', re.I)\n"
    "\n"
    "def word_mean(s):\n"
    "    s = s.dropna().astype(str).head(2000)\n"
    "    return float(s.str.split().str.len().mean()) if len(s) else 0.0\n"
    "\n"
    "def find_text_file():\n"
    "    \"\"\"The configured text file, else a file in the dataset folder whose name looks like text (news.csv, reviews.csv).\"\"\"\n"
    "    if CFG.TEXT_PATH:\n"
    "        p = Path(CFG.TEXT_PATH)\n"
    "        return p if p.exists() else None\n"
    "    folder = Path(DATA_PATH).parent\n"
    "    skip = {Path(DATA_PATH).name, Path(TEST_PATH).name if TEST_PATH else ''}\n"
    "    cands = [f for f in sorted(folder.iterdir()) if f.is_file() and f.suffix.lower() in TABULAR_EXT\n"
    "             and f.name not in skip and file_role(f) is None and TEXT_NAME.search(f.stem)]\n"
    "    return cands[0] if cands else None\n"
    "\n"
    "def load_documents(main, ids):\n"
    "    \"\"\"Documents as (date, text, id columns): from a separate text file, else from a text column of the main table.\"\"\"\n"
    "    tf = find_text_file()\n"
    "    if tf is not None:\n"
    "        t = load_table(tf)\n"
    "        dcol = CFG.TEXT_DATE_COL if CFG.TEXT_DATE_COL != 'auto' else next(\n"
    "            (c for c in t.columns if any(k in c.lower() for k in ('date', 'time', 'tanggal', 'published', 'created'))), t.columns[0])\n"
    "        obj = [c for c in t.columns if c != dcol and not pd.api.types.is_numeric_dtype(t[c])]\n"
    "        tcol = CFG.TEXT_COL if CFG.TEXT_COL != 'auto' else max(obj, key=lambda c: word_mean(t[c]))\n"
    "        tid = [c for c in ids if c in t.columns] if CFG.TEXT_ID_COLS == 'auto' else list(CFG.TEXT_ID_COLS)\n"
    "        source = f'file {tf.name}'\n"
    "    else:\n"
    "        t = main\n"
    "        obj = [c for c in t.columns if c not in ids + ['series'] and not pd.api.types.is_numeric_dtype(t[c])\n"
    "               and not pd.api.types.is_datetime64_any_dtype(t[c])]\n"
    "        cands = [c for c in obj if word_mean(t[c]) >= 3]\n"
    "        tcol = CFG.TEXT_COL if CFG.TEXT_COL != 'auto' else (max(cands, key=lambda c: word_mean(t[c])) if cands else None)\n"
    "        if tcol is None or tcol not in t.columns:\n"
    "            return None, {}\n"
    "        dcol, tid, source = D, list(ids), f'column {tcol!r} of the main table'\n"
    "    docs = pd.DataFrame({'date': pd.to_datetime(t[dcol], errors='coerce').values, 'text': t[tcol].fillna('').astype(str).values})\n"
    "    for c in tid:\n"
    "        docs[c] = t[c].astype(str).values\n"
    "    docs = docs.dropna(subset=['date'])\n"
    "    docs = docs[docs['text'].str.strip().str.len() > 0].reset_index(drop=True)\n"
    "    info = {'text_source': source, 'text_col': tcol, 'text_date_col': dcol, 'text_id_cols': tid, 'text_n_docs': int(len(docs))}\n"
    "    print(f'Text: {len(docs):,} documents from {source}, keyed by {tid or \"date only (shared by every series)\"}, '\n"
    "          f'{docs[\"date\"].min().date()} to {docs[\"date\"].max().date()}')\n"
    "    return docs, info\n"
    "\n"
    "def to_bucket(dates, grid):\n"
    "    \"\"\"Map document dates to grid dates: the period that contains them (start- or end-labelled frequencies).\"\"\"\n"
    "    grid = np.asarray(np.sort(np.unique(grid)), dtype='datetime64[ns]')\n"
    "    d = np.asarray(dates, dtype='datetime64[ns]')\n"
    "    start_labelled = str(FREQ).upper().startswith(('MS', 'QS', 'YS', 'AS', 'BMS', 'BQS'))\n"
    "    idx = np.searchsorted(grid, d, side='right') - 1 if start_labelled else np.searchsorted(grid, d, side='left')\n"
    "    ok = (idx >= 0) & (idx < len(grid))\n"
    "    out = np.full(len(d), np.datetime64('NaT'), dtype='datetime64[ns]')\n"
    "    out[ok] = grid[idx[ok]]\n"
    "    return pd.to_datetime(out)"
))

cells.append(code(
    "TEXT_DECISIONS, TEXT_LAG_REC = {}, None\n"
    "DOCS, TEXT_INFO = load_documents(raw, IDS)\n"
    "if DOCS is None:\n"
    "    note('No text column (strings of three or more words) and no text file (news.csv, reviews.csv, ...) were found. '\n"
    "            'Set TEXT_COL or TEXT_*_PATH in Settings to add the text modality.')\n"
    "else:\n"
    "    hist_docs = DOCS[DOCS['date'] <= df[D].max()].reset_index(drop=True)\n"
    "    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True)\n"
    "    Xt = vec.fit_transform(hist_docs['text'])\n"
    "    k = max(1, min(CFG.TEXT_DIM, Xt.shape[1] - 1, len(hist_docs) - 1))\n"
    "    Z = TruncatedSVD(k, random_state=CFG.SEED).fit_transform(Xt)\n"
    "    comp_cols = [f'c{j}' for j in range(k)]\n"
    "    keys = [c for c in TEXT_INFO['text_id_cols'] if c in IDS]\n"
    "    doc_feats = pd.DataFrame(Z, columns=comp_cols).assign(n_docs=1, words=hist_docs['text'].str.split().str.len().values,\n"
    "                                                          bucket=to_bucket(hist_docs['date'], df[D].unique()))\n"
    "    for c in keys:\n"
    "        doc_feats[c] = hist_docs[c].values\n"
    "    agg = doc_feats.dropna(subset=['bucket']).groupby(keys + ['bucket']).agg(\n"
    "        n_docs=('n_docs', 'sum'), words=('words', 'mean'), **{c: (c, 'mean') for c in comp_cols}).reset_index()\n"
    "    ids_of = raw.drop_duplicates('series').set_index('series')[keys] if keys else None\n"
    "    tp = df[['series', D, Y]].copy()\n"
    "    for c in keys:\n"
    "        tp[c] = tp['series'].map(ids_of[c]).astype(str)\n"
    "    tp = tp.merge(agg.rename(columns={'bucket': D}), on=keys + [D], how='left')\n"
    "    tp['n_docs'] = tp['n_docs'].fillna(0)\n"
    "    tp = tp.sort_values(['series', D]).reset_index(drop=True)\n"
    "\n"
    "    g = tp.groupby('series')[Y]\n"
    "    trend = g.transform(lambda s: s.shift(1).rolling(4 * M, min_periods=1).mean())\n"
    "    spread = g.transform(lambda s: s.rolling(4 * M, min_periods=2).std()).replace(0, np.nan)\n"
    "    tp['z'] = (tp[Y] - trend) / spread\n"
    "    feats = ['n_docs'] + comp_cols\n"
    "    max_lag = int(min(max(3 * HORIZON, 2 * M), 180))\n"
    "    lags = list(range(0, max_lag + 1, max(1, max_lag // 60)))\n"
    "    corr = pd.DataFrame(index=lags, columns=feats, dtype=float)\n"
    "    for f in feats:\n"
    "        fs = tp.groupby('series')[f]\n"
    "        for L in lags:\n"
    "            corr.loc[L, f] = tp['z'].corr(fs.shift(L))\n"
    "    score = corr.abs().max(axis=1)\n"
    "    usable = score[score.index >= HORIZON]\n"
    "    n_obs = int(tp['z'].notna().sum())\n"
    "    threshold = 2 / np.sqrt(max(n_obs, 1))\n"
    "    TEXT_LAG_REC = int(usable.idxmax()) if len(usable) and usable.notna().any() else int(HORIZON)\n"
    "    signal = float(usable.max()) if len(usable) else float('nan')\n"
    "    useful = bool(signal > threshold)\n"
    "    save_table(corr.round(4), 'text_lag_correlation')\n"
    "    print(f'Strongest text-target link at a usable lag (>= horizon {HORIZON}): lag {TEXT_LAG_REC}, |r| = {signal:.3f} '\n"
    "          f'(noise level {threshold:.3f}) -> text {\"carries\" if useful else \"shows no clear\"} lead signal')\n"
    "\n"
    "    fig = plt.figure(figsize=(18, 9.5))\n"
    "    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])\n"
    "    ax0 = fig.add_subplot(gs[0, :])\n"
    "    vol = tp.groupby(D)['n_docs'].sum()\n"
    "    tot = tp.groupby(D)[Y].sum()\n"
    "    ax0.bar(vol.index, vol.values, width=1.0 if FREQ_KEY in ('D', 'h', 'min') else 5, color=SOFT, label='documents')\n"
    "    ax0.set_ylabel('documents per period')\n"
    "    axb = ax0.twinx()\n"
    "    axb.plot(tot.index, tot.rolling(max(1, M), min_periods=1).mean(), color=PRIMARY, lw=1.6, label=f'total {Y} (smoothed)')\n"
    "    axb.set_ylabel(Y)\n"
    "    axb.grid(False)\n"
    "    ax0.set_title('Text volume and the target over time', loc='left')\n"
    "    ax1 = fig.add_subplot(gs[1, 0])\n"
    "    sns.heatmap(corr.abs().T, cmap=CMAP_SEQ, ax=ax1, cbar_kws={'label': '|r|'})\n"
    "    ax1.set_xlabel('lag (periods): text at t - lag vs target at t')\n"
    "    ax1.set_title('Lead correlation by text feature and lag', loc='left')\n"
    "    ax2 = fig.add_subplot(gs[1, 1])\n"
    "    ax2.plot(score.index, score.values, color=PRIMARY, lw=2)\n"
    "    ax2.axvspan(0, HORIZON, color=MUTED, alpha=0.15, label='unusable: inside the horizon')\n"
    "    ax2.axhline(threshold, color=MUTED, ls='--', lw=1, label='noise level 2/sqrt(n)')\n"
    "    ax2.axvline(TEXT_LAG_REC, color=ACCENT, lw=2, label=f'recommended lag {TEXT_LAG_REC}')\n"
    "    ax2.set_xlabel('lag (periods)')\n"
    "    ax2.set_ylabel('max |r| over text features')\n"
    "    ax2.legend(loc='upper right', fontsize=8)\n"
    "    ax2.set_title('Best usable lead', loc='left')\n"
    "    suptitle(fig, 'Text signal', f'{len(hist_docs):,} documents, TF-IDF -> {k} SVD components, target detrended per series')\n"
    "    save_fig(fig, 'text_signal')\n"
    "    TEXT_DECISIONS = {**TEXT_INFO, 'text_lag': TEXT_LAG_REC, 'text_signal': signal, 'text_noise_level': float(threshold),\n"
    "                      'text_useful': useful, 'text_windows': sorted({1, int(M), int(4 * M)})}"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Does any text feature lead the target beyond the horizon, and by how "
    "many periods? A lead well above the noise level justifies the text modality in the pipeline; a "
    "flat curve means the text describes the past rather than anticipating it."
))

cells.append(section("EDA Summary and Pipeline Decisions", "17"))

cells.append(md(
    "All findings are condensed into a markdown summary and a machine-readable "
    "`eda_decisions.json` that the forecasting pipeline reads for its `'auto'` settings: frequency, "
    "horizon, seasonal periods, transform, lag and rolling features, gap filling, backtesting folds, "
    "metric, and model families. Each recommendation follows directly from a section above."
))

cells.append(code(
    "zero_share = float((df[Y] == 0).mean())\n"
    "inter_share = float(intermit['demand_class'].isin(['intermittent', 'lumpy']).mean()) if len(intermit) else 0.0\n"
    "gap_share = float(missing.values.sum() / inside.values.sum())\n"
    "FILL = 'zero' if zero_share > 0.05 else 'interpolate'\n"
    "metric = 'wape' if zero_share > 0.01 else 'smape'\n"
    "min_len = int(life['n_obs'].median())\n"
    "n_folds = int(max(1, min(5, (min_len - 2 * max(SEASONAL_PERIODS[0], 1) - H) // H)))\n"
    "families = ['seasonal_naive']\n"
    "if panel.shape[1] >= 10:\n"
    "    families += ['lightgbm_global']\n"
    "families += ['ets', 'arima'] if panel.shape[1] <= 200 else []\n"
    "if inter_share > 0.2:\n"
    "    families += ['croston_tsb']\n"
    "decisions = {\n"
    "    'source': str(DATA_PATH), 'date_col': D, 'target_col': Y, 'id_cols': IDS, 'freq': FREQ, 'freq_key': FREQ_KEY,\n"
    "    'horizon': HORIZON, 'horizon_source': H_SOURCE, 'seasonal_periods': [int(p) for p in SEASONAL_PERIODS],\n"
    "    'n_series': int(panel.shape[1]), 'median_length': min_len, 'transform': TRANSFORM, 'differencing_d': D_ORDER,\n"
    "    'fill_missing': FILL, 'gap_share': gap_share, 'zero_share': zero_share, 'intermittent_share': inter_share,\n"
    "    'lags_recursive': [int(l) for l in LAGS_ALL], 'lags_direct': [int(l) for l in LAGS_DIRECT], 'rolling_windows': [int(w) for w in ROLL_WINDOWS],\n"
    "    'exog_known_future': EXOG_KNOWN, 'exog_past_only': [c for c in EXOG if c not in EXOG_KNOWN],\n"
    "    'special_dates': [str(d_.date()) for d_ in SPECIAL.index], 'level_shifts': int(len(LEVEL_SHIFTS)),\n"
    "    'anomalies': int(len(anoms)), 'cv': {'scheme': 'expanding_window', 'n_folds': n_folds, 'step': H, 'horizon': H},\n"
    "    'metric': metric, 'secondary_metrics': ['mase', 'rmse'], 'best_naive': BEST_BENCH,\n"
    "    'best_naive_wape': float(summary_bench.loc[BEST_BENCH, 'WAPE']), 'model_families': families,\n"
    "    'global_model_recommended': bool(panel.shape[1] >= 10 or (not np.isnan(MEAN_CORR) and MEAN_CORR > 0.3)),\n"
    "    **TEXT_DECISIONS,\n"
    "}\n"
    "(CFG.OUTPUT_DIR / 'eda_decisions.json').write_text(json.dumps(decisions, indent=2, default=str), encoding='utf-8')\n"
    "\n"
    "lines = [f'# Forecasting EDA Summary: {DATA_PATH.name}', '', '## Structure',\n"
    "         f'- {panel.shape[1]} series of `{Y}` at frequency **{FREQ}**, {df[D].min().date()} to {df[D].max().date()} ({len(full_index):,} steps)',\n"
    "         f'- Horizon **{HORIZON}** ({H_SOURCE}); median series length {min_len}; {int(structure.loc[\"series starting late\", \"value\"])} series start late',\n"
    "         f'- Gaps {gap_share:.2%} of in-lifespan steps; zeros {zero_share:.1%}; intermittent or lumpy series {inter_share:.0%}', '',\n"
    "         '## Patterns',\n"
    "         f'- Transform: **{TRANSFORM}** (variance-level slope {slope:.2f}); differencing d = {D_ORDER}',\n"
    "         f'- Seasonal periods: **{SEASONAL_PERIODS}**; median trend strength {st[\"trend_strength\"].median():.2f}, seasonal strength {st[\"seasonal_strength\"].median():.2f}',\n"
    "         f'- Significant lags: {sorted(sig)}',\n"
    "         f'- Candidate special days: ' + ', '.join(str(d_.date()) for d_ in SPECIAL.index[:6]),\n"
    "         f'- Point anomalies: {len(anoms):,}; level shifts: {len(LEVEL_SHIFTS)}', '']\n"
    "if not exog_table.empty:\n"
    "    lines.append('## Drivers')\n"
    "    for c, r in exog_table.iterrows():\n"
    "        extra = f', uplift x{r[\"uplift_ratio\"]:.2f}' if 'uplift_ratio' in r and not pd.isna(r.get('uplift_ratio')) else ''\n"
    "        extra += f', elasticity {r[\"elasticity\"]:.2f}' if 'elasticity' in r and not pd.isna(r.get('elasticity')) else ''\n"
    "        lines.append(f'- `{c}`: corr {r[\"corr_lag0\"]:+.2f} at lag 0, {\"known in future\" if r[\"known_in_future\"] else \"past only\"}{extra}')\n"
    "    lines.append('')\n"
    "lines += ['## Benchmarks', f'- Best naive: **{BEST_BENCH}**, WAPE {summary_bench.loc[BEST_BENCH, \"WAPE\"]:.3f}, median MASE {summary_bench.loc[BEST_BENCH, \"median_MASE\"]:.3f}', '',\n"
    "          '## Pipeline decisions',\n"
    "          f'1. Fill gaps with **{FILL}**; apply **{TRANSFORM}** to the target and invert after forecasting',\n"
    "          f'2. Features: lags {LAGS_DIRECT} (direct) or {LAGS_ALL[:10]} (recursive), rolling windows {ROLL_WINDOWS}, calendar parts, special-day flags'\n"
    "          + (f', known covariates {EXOG_KNOWN}' if EXOG_KNOWN else ''),\n"
    "          f'3. Validation: expanding-window backtest, {n_folds} folds of {H} steps',\n"
    "          f'4. Metric: **{metric.upper()}** plus MASE (MAPE is unsafe with {zero_share:.0%} zeros)' if zero_share > 0 else f'4. Metric: **{metric.upper()}** plus MASE',\n"
    "          f'5. Models to try: {families}' + (' (global model across series recommended)' if decisions['global_model_recommended'] else ''),\n"
    "          '', '## Exported files', f'- Figures ({len(SAVED_FIGS)}): `{CFG.FIG_DIR}`', f'- Tables: `{CFG.TABLE_DIR}`', '- Decisions: `eda_decisions.json`']\n"
    "summary_md = '\\n'.join(lines)\n"
    "(CFG.OUTPUT_DIR / 'eda_summary.md').write_text(summary_md, encoding='utf-8')\n"
    "display(Markdown(summary_md))"
))

cells.append(md(
    "The final cell lists every exported artefact."
))

cells.append(code(
    "print(f'Figures in {CFG.FIG_DIR.resolve()}:')\n"
    "for f in SAVED_FIGS:\n"
    "    print('  ', f)\n"
    "print(f'\\nTables in {CFG.TABLE_DIR.resolve()}:')\n"
    "for f in sorted(CFG.TABLE_DIR.glob('*.csv')):\n"
    "    print('  ', f.name)\n"
    "print(f'\\nDecisions: {(CFG.OUTPUT_DIR / \"eda_decisions.json\").resolve()}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Summarise the frequency and horizon, the dominant seasonality and "
    "events, the transform, the benchmark to beat, and the planned model family, each linked to the "
    "figure that justifies it."
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
