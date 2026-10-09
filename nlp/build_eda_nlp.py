import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "eda_nlp.ipynb"
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
    "# NLP EDA Template\n\n"
    "*A dataset-agnostic exploratory analysis notebook for any text dataset (reviews, comments, "
    "lyrics, tweets, documents), with or without labels, in English or Indonesian. Every chart, "
    "including the word clouds, is exported to `eda-output/`.*"
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
    "3. [**Text Toolkit**](#3)\n"
    "4. [**Corpus Overview**](#4)\n"
    "5. [**Text Quality and Noise**](#5)\n"
    "6. [**Length Analysis**](#6)\n"
    "7. [**Vocabulary Statistics**](#7)\n"
    "8. [**Word Clouds**](#8)\n"
    "9. [**N-gram Analysis**](#9)\n"
    "10. [**Distinctive Words per Label**](#10)\n"
    "11. [**Sentiment Analysis**](#11)\n"
    "12. [**Topic Modeling**](#12)\n"
    "13. [**Semantic Map**](#13)\n"
    "14. [**Duplicates and Label Noise**](#14)\n"
    "15. [**Segment and Temporal Analysis**](#15)\n"
    "16. [**EDA Summary and Modelling Recommendations**](#16)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(md(
    "## Overview\n\n"
    "*Text data hides its problems: encoding noise, duplicates, mixed languages, label noise, and "
    "extreme length variation all silently hurt NLP models. This notebook profiles a text column "
    "end to end so that cleaning, tokenisation, sequence length, and model choices are all "
    "grounded in evidence.*"
))

cells.append(md(
    "## Aim\n\n"
    "*The notebook measures corpus size, quality, and noise, length distributions, vocabulary "
    "behaviour (Zipf, Heaps, coverage), frequent and distinctive words and phrases per label, "
    "lexicon sentiment, latent topics, and a 2-D semantic map. It closes with an automatic summary "
    "and concrete modelling recommendations (max sequence length, vocabulary size, cleaning steps, "
    "class weighting).*"
))

cells.append(md(
    "## How to Point the Notebook at a Dataset\n\n"
    "*All changes happen in the `Settings` class (section 2).*\n\n"
    "| Setting | Purpose | Example |\n"
    "|---|---|---|\n"
    "| `KAGGLE_PATH` / `COLAB_PATH` / `LOCAL_PATH` | Data file per environment (auto-detected) | `'/kaggle/input/comp'` (folder) or a file path |\n"
    "| `TEXT_COL` | Text column, or `'auto'` to pick the column with the longest texts | `'review'` |\n"
    "| `LABEL_COL` | Class label, numeric target (binned into quantiles), `None`, or `'auto'` (from sample_submission / test) | `'sentiment'` |\n"
    "| `GROUP_COL`, `TIME_COL` | Optional segment and time columns for section 15 | `'app'`, `'date'` |\n"
    "| `LANGUAGE` | Stopword set: `'en'`, `'id'` (Indonesian, including slang), or `'both'` | `'id'` |\n"
    "| `EXTRA_STOPWORDS`, `REMOVE_PATTERNS` | Dataset-specific stopwords and regex patterns to strip | `['app']`, `[r'\\[.*?\\]']` |\n\n"
    "*Every column setting defaults to `'auto'`, so the notebook runs on any text dataset as is. "
    "`REMOVE_PATTERNS` is the place for dataset-specific noise, for example `[r'\\[[^\\]]*\\]']` to strip "
    "`[Chorus]` style headers or `[r'@\\w+']` to strip user mentions.*"
))

cells.append(md(
    "## Dataset\n"
    "\n"
    "*Any text dataset works: one document per row, a text column, and optionally a label (a class, a numeric score, ordered levels, or several labels per document). Put the files in `data/` next to this notebook, or point `KAGGLE_PATH` / `COLAB_PATH` / `LOCAL_PATH` at the dataset folder.*\n"
    "\n"
    "```\n"
    "data/\n"
    "├── train.csv              (text + label)\n"
    "├── test.csv               (text only, optional)\n"
    "└── sample_submission.csv  (id + target columns, optional)\n"
    "```\n"
    "\n"
    "*With `'auto'` the text column is the string column with the longest documents and the label comes from `sample_submission.csv` (or is the only train column missing from the test file). Set `TEXT_COL` and `LABEL_COL` explicitly whenever the guess is wrong.*"
))

cells.append(md(
    "## Approach: Layered Text Profiling\n\n"
    "*The analysis moves from surface properties to meaning. Cheap checks run on the full corpus, "
    "while expensive ones (topic modeling, t-SNE, sentiment) run on a reproducible sample.*\n\n"
    "```\n"
    "Raw text\n"
    "    |\n"
    "    +-- Overview     : size, empties, label balance, examples\n"
    "    +-- Quality      : URLs, HTML, emoji, numbers, casing, repeats, script / language mix\n"
    "    +-- Length       : chars, words, sentences, by label\n"
    "    +-- Vocabulary   : Zipf, Heaps, hapax share, coverage curve\n"
    "    +-- Words        : word clouds, n-grams, log-odds distinctive words per label\n"
    "    +-- Meaning      : VADER sentiment, NMF topics, TF-IDF + SVD + t-SNE map\n"
    "    +-- Integrity    : exact and near duplicates, conflicting labels\n"
    "    +-- Segments     : group and time breakdowns\n"
    "         |\n"
    "    EDA summary + recommendations -> eda-output/ (PNG + CSV + MD)\n"
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
    "%pip install -q numpy pandas matplotlib seaborn scipy scikit-learn wordcloud vaderSentiment pyarrow openpyxl"
))

cells.append(md(
    "## Import Libraries\n\n"
    "`wordcloud` and `vaderSentiment` are imported defensively, so a missing package only skips its "
    "own section instead of stopping the notebook."
))

cells.append(code(
    "import os\n"
    "import re\n"
    "import json\n"
    "import random\n"
    "import warnings\n"
    "import unicodedata\n"
    "from collections import Counter\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib as mpl\n"
    "import matplotlib.pyplot as plt\n"
    "from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb\n"
    "import seaborn as sns\n"
    "from scipy import stats\n"
    "from sklearn.decomposition import NMF, TruncatedSVD\n"
    "from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer\n"
    "from sklearn.manifold import TSNE\n"
    "from sklearn.neighbors import NearestNeighbors\n"
    "from IPython.display import Markdown, display\n"
    "\n"
    "AVAILABLE = {}\n"
    "try:\n"
    "    from wordcloud import WordCloud\n"
    "    AVAILABLE['wordcloud'] = True\n"
    "except Exception:\n"
    "    AVAILABLE['wordcloud'] = False\n"
    "try:\n"
    "    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer\n"
    "    AVAILABLE['vader'] = True\n"
    "except Exception:\n"
    "    AVAILABLE['vader'] = False\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "pd.set_option('display.max_columns', 100)\n"
    "pd.set_option('display.width', 200)\n"
    "pd.set_option('display.max_colwidth', 120)\n"
    "print('Optional packages:', AVAILABLE)"
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
    "All paths, column roles, language options, and analysis knobs are centralised here. Switching "
    "between Kaggle, Colab, and local execution only requires editing the path lines in block 1."
))

cells.append(code(
    "class Settings:\n"
    "    SEED       = 42\n"
    "    _ON_KAGGLE = Path('/kaggle/input').exists()\n"
    "    _ON_COLAB  = Path('/content').exists() and not _ON_KAGGLE\n"
    "\n"
    "    # 1) Data location: a file or a dataset folder (train / test / sample_submission are found by name)\n"
    "    KAGGLE_PATH = '/kaggle/input/<dataset-slug>'\n"
    "    COLAB_PATH  = '/content/drive/MyDrive/<folder>'\n"
    "    LOCAL_PATH  = 'data'\n"
    "    READ_KWARGS = {}\n"
    "\n"
    "    # 2) Columns: LABEL_COL may be a class label, a numeric target, or None\n"
    "    TEXT_COL  = 'auto'\n"
    "    LABEL_COL = 'auto'\n"
    "    GROUP_COL = None\n"
    "    TIME_COL  = None\n"
    "    ID_COL    = None\n"
    "\n"
    "    # 3) Language and cleaning\n"
    "    LANGUAGE        = 'both'\n"
    "    EXTRA_STOPWORDS = []\n"
    "    REMOVE_PATTERNS = []\n"
    "    MIN_TOKEN_LEN   = 2\n"
    "\n"
    "    # 4) Analysis knobs\n"
    "    MAX_DOCS         = 30_000\n"
    "    MAX_LABELS       = 8\n"
    "    LABEL_BINS       = 4\n"
    "    CLASS_MAX        = 30\n"
    "    TOP_N            = 20\n"
    "    WORDCLOUD_WORDS  = 150\n"
    "    TOPIC_K          = 8\n"
    "    SENTIMENT_SAMPLE = 5_000\n"
    "    TSNE_SAMPLE      = 3_000\n"
    "    NEAR_DUP_SAMPLE  = 5_000\n"
    "    NEAR_DUP_SIM     = 0.90\n"
    "    TIME_RANGE       = (1950, 2025)\n"
    "\n"
    "    # 5) Outputs\n"
    "    DATA_PATH  = KAGGLE_PATH if _ON_KAGGLE else COLAB_PATH if _ON_COLAB else LOCAL_PATH\n"
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
    "print(f'Output dir  : {CFG.OUTPUT_DIR.resolve()}')"
))

cells.append(md(
    "## Load Dataset\n\n"
    "The path is resolved (with a Kaggle fallback to the largest tabular file), the text column is "
    "chosen, missing texts are kept as empty strings so they can be counted, and the analysis set "
    "is capped at `MAX_DOCS` rows with a fixed seed."
))

cells.append(code(
    "TABULAR_EXT = ('.csv', '.tsv', '.txt', '.parquet', '.pq', '.feather', '.xlsx', '.xls', '.json', '.jsonl', '.zip', '.gz')\n"
    "\n"
    "def load_table(path, **kwargs):\n"
    "    suffixes = [s.lower() for s in path.suffixes]\n"
    "    if '.parquet' in suffixes or '.pq' in suffixes:\n"
    "        return pd.read_parquet(path, **kwargs)\n"
    "    if '.feather' in suffixes:\n"
    "        return pd.read_feather(path, **kwargs)\n"
    "    if '.xlsx' in suffixes or '.xls' in suffixes:\n"
    "        return pd.read_excel(path, **kwargs)\n"
    "    if '.jsonl' in suffixes:\n"
    "        return pd.read_json(path, lines=True, **kwargs)\n"
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
    "DATA_PATH, TEST_PATH, SAMPLE_PATH = discover_files(CFG.DATA_PATH, None)\n"
    "assert DATA_PATH is not None, f'{CFG.DATA_PATH} not found and no train file was discovered. Edit the path lines in Settings.'\n"
    "raw = load_table(DATA_PATH, **CFG.READ_KWARGS)\n"
    "\n"
    "if CFG.TEXT_COL == 'auto' or CFG.TEXT_COL not in raw.columns:\n"
    "    obj = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]\n"
    "    TEXT_COL = max(obj, key=lambda c: raw[c].dropna().astype(str).str.len().mean())\n"
    "    print(f'[info] TEXT_COL auto-selected: {TEXT_COL}')\n"
    "else:\n"
    "    TEXT_COL = CFG.TEXT_COL\n"
    "if CFG.LABEL_COL == 'auto':\n"
    "    raw_test = load_table(TEST_PATH, **CFG.READ_KWARGS) if TEST_PATH else None\n"
    "    sample = load_table(SAMPLE_PATH) if SAMPLE_PATH else None\n"
    "    only_train = [c for c in raw.columns if raw_test is not None and c not in raw_test.columns and c != TEXT_COL]\n"
    "    if sample is not None and sample.shape[1] >= 2 and sample.columns[1] in raw.columns:\n"
    "        LABEL_COL = sample.columns[1]\n"
    "    else:\n"
    "        LABEL_COL = only_train[0] if len(only_train) == 1 else None\n"
    "    print(f'[info] LABEL_COL auto-selected: {LABEL_COL}' if LABEL_COL else\n"
    "          '[info] no label found (no sample_submission or test file): unlabeled EDA. Set LABEL_COL in Settings.')\n"
    "else:\n"
    "    LABEL_COL = CFG.LABEL_COL if CFG.LABEL_COL in raw.columns else None\n"
    "if CFG.LABEL_COL not in (None, 'auto') and LABEL_COL is None:\n"
    "    print(f'[warn] LABEL_COL={CFG.LABEL_COL!r} not found, running unlabeled EDA')\n"
    "\n"
    "df = raw.copy()\n"
    "df['text'] = df[TEXT_COL].fillna('').astype(str)\n"
    "N_TOTAL = len(df)\n"
    "if len(df) > CFG.MAX_DOCS:\n"
    "    df = df.sample(CFG.MAX_DOCS, random_state=CFG.SEED)\n"
    "df = df.reset_index(drop=True)\n"
    "print(f'Loaded {N_TOTAL:,} rows from {DATA_PATH}; analysing {len(df):,} rows')\n"
    "print(f'Text column: {TEXT_COL!r} | label column: {LABEL_COL!r}')"
))

# ---------------------------------------------------------------------------
# Section 3: Text Toolkit
# ---------------------------------------------------------------------------
cells.append(section("Text Toolkit", "3"))

cells.append(md(
    "The plotting identity matches the other templates. Each label gets a fixed colour from the "
    "palette, which is reused in every chart and in the per-label word clouds, so a label is "
    "recognisable across the whole report."
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
    "def make_grid(n, ncols=4, w=4.2, h=3.2):\n"
    "    ncols = max(1, min(ncols, n))\n"
    "    nrows = int(np.ceil(n / ncols))\n"
    "    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)\n"
    "    axes = axes.ravel()\n"
    "    for ax in axes[n:]:\n"
    "        ax.set_visible(False)\n"
    "    return fig, axes[:n]\n"
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

cells.append(md(
    "## Stopwords and Tokenisation\n\n"
    "Stopwords combine scikit-learn's English list, a built-in Indonesian list that includes common "
    "informal and slang forms (`gak`, `aja`, `yg`), apostrophe-free contractions, and "
    "`EXTRA_STOPWORDS`. Tokens are Unicode letter sequences, so accented and non-Latin words survive, "
    "while digits and punctuation are dropped. Two token streams are kept: all tokens (for vocabulary "
    "statistics) and content tokens without stopwords (for word clouds, n-grams, and topics)."
))

cells.append(code(
    "STOP_ID = set('''yang dan di ke dari ini itu dengan untuk tidak ada saya aku kamu kau dia mereka kita kami\n"
    "anda akan juga sudah telah belum bisa dapat harus atau tapi tetapi namun karena jadi jika kalau saat ketika\n"
    "seperti pada dalam oleh agar supaya sangat lebih paling sama lagi masih hanya saja pun bagi tentang antara\n"
    "setelah sebelum sejak hingga sampai mau ingin perlu pernah sedang apa siapa mana bagaimana kenapa mengapa\n"
    "begitu sini situ sana nya lah kah pula para sebuah seorang suatu semua setiap banyak sedikit hal adalah ialah\n"
    "yaitu yakni merupakan bahwa maka serta tanpa tak bukan jangan dong deh sih kok nih tuh loh lho kan ya yah\n"
    "gak ga nggak ngga enggak engga gk tdk aja doang udah udh sdh blm bgt banget yg dgn utk krn klo kalo trs terus\n"
    "gw gue lu lo elo aku ku mu nya min kak gan sis bro'''.split())\n"
    "CONTRACTIONS = set('dont im youre cant ive ill thats wont didnt doesnt isnt arent wasnt werent couldnt wouldnt '\n"
    "                   'shouldnt hes shes theyre weve youve youll itll lets aint id youd hed theyd whats theres'.split())\n"
    "\n"
    "STOPWORDS = set(CONTRACTIONS) | {w.lower() for w in CFG.EXTRA_STOPWORDS}\n"
    "if CFG.LANGUAGE in ('en', 'both'):\n"
    "    STOPWORDS |= set(ENGLISH_STOP_WORDS)\n"
    "if CFG.LANGUAGE in ('id', 'both'):\n"
    "    STOPWORDS |= STOP_ID\n"
    "\n"
    "URL_RE      = re.compile(r'https?://\\S+|www\\.\\S+')\n"
    "EMAIL_RE    = re.compile(r'\\b[\\w.+-]+@[\\w-]+\\.[\\w.]+\\b')\n"
    "MENTION_RE  = re.compile(r'(?<!\\w)@\\w+')\n"
    "HASHTAG_RE  = re.compile(r'(?<!\\w)#\\w+')\n"
    "HTML_RE     = re.compile(r'<[^>]+>|&[a-z]+;|&#\\d+;')\n"
    "NUMBER_RE   = re.compile(r'\\b\\d+(?:[.,]\\d+)?\\b')\n"
    "EMOJI_RE    = re.compile('[\\U0001F300-\\U0001FAFF\\U00002600-\\U000027BF\\U0001F1E6-\\U0001F1FF]')\n"
    "REPEAT_RE   = re.compile(r'([^\\W\\d_])\\1{2,}')\n"
    "TOKEN_RE    = re.compile(r\"[^\\W\\d_]+(?:'[^\\W\\d_]+)?\")\n"
    "SENT_RE     = re.compile(r'[.!?]+|\\n+')\n"
    "REMOVE_RES  = [re.compile(p) for p in CFG.REMOVE_PATTERNS]\n"
    "\n"
    "def clean(text):\n"
    "    for rx in REMOVE_RES:\n"
    "        text = rx.sub(' ', text)\n"
    "    text = URL_RE.sub(' ', text)\n"
    "    text = HTML_RE.sub(' ', text)\n"
    "    text = unicodedata.normalize('NFKC', text).lower()\n"
    "    return REPEAT_RE.sub(r'\\1\\1', text)\n"
    "\n"
    "def tokens_all(text):\n"
    "    return [t.replace(\"'\", '') for t in TOKEN_RE.findall(clean(text))]\n"
    "\n"
    "def content(tokens):\n"
    "    return [t for t in tokens if len(t) >= CFG.MIN_TOKEN_LEN and t not in STOPWORDS]\n"
    "\n"
    "df['clean'] = [clean(t) for t in df['text']]\n"
    "TOK_ALL = [[t.replace(\"'\", '') for t in TOKEN_RE.findall(c)] for c in df['clean']]\n"
    "TOK = [content(t) for t in TOK_ALL]\n"
    "df['n_tokens'] = [len(t) for t in TOK_ALL]\n"
    "df['n_content'] = [len(t) for t in TOK]\n"
    "print(f'{len(STOPWORDS):,} stopwords | {sum(df.n_tokens):,} tokens | {sum(df.n_content):,} content tokens')\n"
    "print('Example:', TOK[0][:25])"
))

cells.append(md(
    "## Label Preparation\n\n"
    "A categorical label is used as is. A numeric target with many distinct values is binned into "
    "`LABEL_BINS` quantile groups (Q1 is the lowest), so every per-label analysis also works for "
    "regression targets. Per-label charts show the `MAX_LABELS` most frequent labels and assign each "
    "one a fixed colour."
))

cells.append(code(
    "HAS_LABEL = LABEL_COL is not None\n"
    "LABEL_KIND = None\n"
    "if HAS_LABEL:\n"
    "    lab = df[LABEL_COL]\n"
    "    if pd.api.types.is_numeric_dtype(lab) and lab.nunique() > CFG.CLASS_MAX:\n"
    "        names = [f'Q{i + 1}' for i in range(CFG.LABEL_BINS)]\n"
    "        names[0], names[-1] = names[0] + ' (low)', names[-1] + ' (high)'\n"
    "        df['label'] = pd.qcut(lab.rank(method='first'), CFG.LABEL_BINS, labels=names).astype(str)\n"
    "        LABEL_KIND = 'binned numeric'\n"
    "        ORDER = names\n"
    "    else:\n"
    "        df['label'] = lab.astype(str).where(lab.notna(), '(missing)')\n"
    "        LABEL_KIND = 'categorical'\n"
    "        ORDER = df['label'].value_counts().index.tolist()\n"
    "    TOP_LABELS = ORDER[:CFG.MAX_LABELS] if LABEL_KIND == 'categorical' else ORDER\n"
    "    LABEL_COLOR = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(TOP_LABELS)}\n"
    "    df['label_top'] = df['label'].where(df['label'].isin(TOP_LABELS), '(other)')\n"
    "    print(f'Label kind: {LABEL_KIND} | {df.label.nunique()} labels | plotted: {TOP_LABELS}')\n"
    "else:\n"
    "    TOP_LABELS, LABEL_COLOR = [], {}\n"
    "    note('No label column, so per-label analyses are skipped.')"
))

# ---------------------------------------------------------------------------
# Section 4: Corpus Overview
# ---------------------------------------------------------------------------
cells.append(section("Corpus Overview", "4"))

cells.append(md(
    "The overview establishes the basic shape of the corpus: how many documents exist, how many "
    "are empty or duplicated, and how labels are distributed. Label imbalance decides the metric "
    "(macro F1 for skewed classes) and whether class weights are needed."
))

cells.append(code(
    "norm = df['clean'].str.split().str.join(' ')\n"
    "overview = pd.Series({\n"
    "    'documents (analysed)': len(df),\n"
    "    'documents (file total)': N_TOTAL,\n"
    "    'empty or whitespace': int((df['text'].str.strip() == '').sum()),\n"
    "    'no content tokens': int((df['n_content'] == 0).sum()),\n"
    "    'exact duplicate texts': int(df['text'].duplicated().sum()),\n"
    "    'duplicates after normalisation': int(norm[norm != ''].duplicated().sum()),\n"
    "    'labels': df['label'].nunique() if HAS_LABEL else 0,\n"
    "}, name='value').to_frame()\n"
    "overview['share_%'] = overview['value'] / len(df) * 100\n"
    "overview.loc[['documents (analysed)', 'documents (file total)', 'labels'], 'share_%'] = np.nan\n"
    "save_table(overview, 'corpus_overview')\n"
    "display(overview)"
))

cells.append(md(
    "## Label Distribution\n\n"
    "The chart shows document counts per label with their share. The imbalance ratio (largest "
    "label over smallest) is reported in the subtitle."
))

cells.append(code(
    "if HAS_LABEL:\n"
    "    vc = df['label'].value_counts()\n"
    "    if LABEL_KIND == 'binned numeric':\n"
    "        vc = vc.reindex(ORDER)\n"
    "    show = vc.head(30)[::-1]\n"
    "    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.3 * len(show) + 1.6)))\n"
    "    ax.barh([short(i, 26) for i in show.index], show.values, color=[LABEL_COLOR.get(i, MUTED) for i in show.index])\n"
    "    for i, v in enumerate(show.values):\n"
    "        ax.text(v, i, f'  {v:,} ({v / len(df):.1%})', va='center', fontsize=8, color='#4A5568')\n"
    "    ax.set_xlabel('documents')\n"
    "    extra = f', top 30 of {len(vc)} shown' if len(vc) > 30 else ''\n"
    "    suptitle(fig, f'Label distribution: {LABEL_COL}', f'imbalance ratio {vc.max() / vc.min():.1f}x{extra}; coloured = labels used in per-label charts')\n"
    "    save_fig(fig, 'label_distribution')\n"
    "    save_table(vc.rename('count').to_frame(), 'label_distribution')"
))

cells.append(md(
    "## Example Documents\n\n"
    "Reading raw examples is irreplaceable: it reveals formatting artefacts, boilerplate, and "
    "language that no statistic flags. Two random documents per plotted label are shown, truncated "
    "to 300 characters."
))

cells.append(code(
    "groups = TOP_LABELS if HAS_LABEL else [None]\n"
    "rows = []\n"
    "for g in groups:\n"
    "    pool = df[df['label'] == g] if g is not None else df\n"
    "    for _, r in pool.sample(min(2, len(pool)), random_state=CFG.SEED).iterrows():\n"
    "        rows.append({'label': g, 'words': r['n_tokens'], 'text': short(r['text'].replace('\\n', ' / '), 300)})\n"
    "display(pd.DataFrame(rows))"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* How large and how clean is the corpus? How imbalanced are the labels, "
    "and what does that imply for the metric and class weighting? What artefacts are visible in the "
    "raw examples (headers, signatures, templates, mixed languages)?"
))

# ---------------------------------------------------------------------------
# Section 5: Text Quality and Noise
# ---------------------------------------------------------------------------
cells.append(section("Text Quality and Noise", "5"))

cells.append(md(
    "Noise patterns determine the cleaning pipeline. The prevalence of URLs, mentions, hashtags, "
    "HTML, numbers, emoji, elongated words, and shouting is measured as the share of documents "
    "containing each pattern. The script and language mix is estimated from non-ASCII letters and "
    "from the share of English versus Indonesian stopwords in each document."
))

cells.append(code(
    "patterns = {'url': URL_RE, 'email': EMAIL_RE, 'mention (@user)': MENTION_RE, 'hashtag': HASHTAG_RE,\n"
    "            'html / entity': HTML_RE, 'number': NUMBER_RE, 'emoji': EMOJI_RE, 'elongated word (sooo)': REPEAT_RE}\n"
    "for rx in REMOVE_RES:\n"
    "    patterns[f'custom: {rx.pattern}'] = rx\n"
    "rows = []\n"
    "for name, rx in patterns.items():\n"
    "    counts = df['text'].map(lambda t: len(rx.findall(t)))\n"
    "    rows.append({'pattern': name, 'docs_%': (counts > 0).mean() * 100, 'mean_per_doc': counts.mean()})\n"
    "\n"
    "letters = df['text'].str.count(r'[^\\W\\d_]').clip(lower=1)\n"
    "upper = df['text'].str.count(r'[A-Z]')\n"
    "non_ascii = df['text'].str.count(r'[^\\x00-\\x7f]')\n"
    "rows.append({'pattern': 'shouting (>50% caps)', 'docs_%': ((upper / letters) > 0.5).mean() * 100, 'mean_per_doc': np.nan})\n"
    "rows.append({'pattern': 'non-ASCII heavy (>20%)', 'docs_%': ((non_ascii / letters) > 0.2).mean() * 100, 'mean_per_doc': np.nan})\n"
    "rows.append({'pattern': 'excess punctuation (!!! ???)', 'docs_%': df['text'].str.contains(r'[!?]{2,}').mean() * 100, 'mean_per_doc': np.nan})\n"
    "noise = pd.DataFrame(rows).set_index('pattern').sort_values('docs_%', ascending=False)\n"
    "save_table(noise, 'noise_patterns')\n"
    "display(noise.round(3))\n"
    "\n"
    "en_sw, id_sw = set(ENGLISH_STOP_WORDS) | CONTRACTIONS, STOP_ID\n"
    "def guess_language(tokens):\n"
    "    if len(tokens) < 3:\n"
    "        return 'too short'\n"
    "    en = sum(t in en_sw for t in tokens) / len(tokens)\n"
    "    idn = sum(t in id_sw for t in tokens) / len(tokens)\n"
    "    if max(en, idn) < 0.08:\n"
    "        return 'other / unknown'\n"
    "    if en > 1.5 * idn:\n"
    "        return 'English'\n"
    "    if idn > 1.5 * en:\n"
    "        return 'Indonesian'\n"
    "    return 'mixed'\n"
    "df['lang_guess'] = [guess_language(t) for t in TOK_ALL]\n"
    "lang = df['lang_guess'].value_counts()\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(16, max(4.5, 0.36 * len(noise) + 1.8)), gridspec_kw={'width_ratios': [1.6, 1]})\n"
    "n = noise['docs_%'][::-1]\n"
    "axes[0].barh(n.index, n.values, color=[ACCENT if v > 5 else PALETTE[3] if v > 1 else PALETTE[2] for v in n.values])\n"
    "bar_labels(axes[0], '{:.1f}%')\n"
    "axes[0].set_xlabel('% of documents')\n"
    "axes[0].set_title('Noise pattern prevalence (green < 1% < yellow < 5% < orange)', loc='left')\n"
    "lp = (lang / lang.sum() * 100).sort_values()\n"
    "axes[1].barh(lp.index, lp.values, color=[PALETTE[i % 10] for i in range(len(lp))][::-1])\n"
    "bar_labels(axes[1], '{:.1f}%')\n"
    "axes[1].set_xlabel('% of documents')\n"
    "axes[1].set_title('Language guess (stopword heuristic)', loc='left')\n"
    "suptitle(fig, 'Text quality and noise', 'decide which patterns to strip, normalise, or keep as features')\n"
    "save_fig(fig, 'noise_and_language')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Which patterns are frequent enough to need handling (above 1% of "
    "documents)? Emoji and elongations often carry sentiment and are better converted than deleted. "
    "Is the corpus monolingual, and does the language mix call for a multilingual model?"
))

# ---------------------------------------------------------------------------
# Section 6: Length Analysis
# ---------------------------------------------------------------------------
cells.append(section("Length Analysis", "6"))

cells.append(md(
    "Length drives the maximum sequence length of transformer models, the cost of training, and "
    "sometimes the label itself (complaints are often longer than praise). Characters, words, "
    "sentences, average word length, and lexical diversity (type-token ratio) are profiled."
))

cells.append(code(
    "df['n_chars'] = df['text'].str.len()\n"
    "df['n_sentences'] = [max(1, len([s for s in SENT_RE.split(t) if s.strip()])) for t in df['text']]\n"
    "df['avg_word_len'] = [np.mean([len(w) for w in t]) if t else 0 for t in TOK_ALL]\n"
    "df['ttr'] = [len(set(t)) / len(t) if t else 0 for t in TOK_ALL]\n"
    "len_cols = ['n_chars', 'n_tokens', 'n_content', 'n_sentences', 'avg_word_len', 'ttr']\n"
    "len_stats = df[len_cols].describe(percentiles=[0.05, 0.5, 0.9, 0.95, 0.99]).T\n"
    "save_table(len_stats, 'length_stats')\n"
    "display(len_stats.round(2))\n"
    "\n"
    "titles = {'n_chars': 'Characters', 'n_tokens': 'Words', 'n_content': 'Content words (no stopwords)',\n"
    "          'n_sentences': 'Sentences / lines', 'avg_word_len': 'Average word length', 'ttr': 'Type-token ratio'}\n"
    "fig, axes = make_grid(len(len_cols), ncols=3, w=5.4, h=3.6)\n"
    "for i, (ax, c) in enumerate(zip(axes, len_cols)):\n"
    "    x = df[c].clip(upper=df[c].quantile(0.99))\n"
    "    sns.histplot(x, bins=50, color=PALETTE[i], edgecolor='white', alpha=0.85, ax=ax)\n"
    "    for q, ls in ((0.5, '--'), (0.95, ':')):\n"
    "        ax.axvline(df[c].quantile(q), color=DARK, ls=ls, lw=1.2)\n"
    "    ax.set_title(f'{titles[c]}  (median {df[c].median():,.1f}, p95 {df[c].quantile(0.95):,.1f})', loc='left', fontsize=10)\n"
    "    ax.set_xlabel('')\n"
    "    ax.set_ylabel('')\n"
    "suptitle(fig, 'Document length profile', 'clipped at p99; dashed = median, dotted = p95')\n"
    "save_fig(fig, 'length_profile')"
))

cells.append(md(
    "## Length by Label\n\n"
    "Box plots compare word counts across labels, and a Kruskal-Wallis test with epsilon-squared "
    "effect size checks whether length alone separates the labels. A large effect means length is "
    "a useful feature and a possible shortcut that a model may exploit."
))

cells.append(code(
    "if HAS_LABEL:\n"
    "    d = df[df['label'].isin(TOP_LABELS)]\n"
    "    groups = [g['n_tokens'].values for _, g in d.groupby('label')]\n"
    "    H, p = stats.kruskal(*groups) if len(groups) > 1 else (np.nan, np.nan)\n"
    "    eps2 = H / (len(d) - 1) if len(groups) > 1 else np.nan\n"
    "    order = d.groupby('label')['n_tokens'].median().sort_values().index.tolist() if LABEL_KIND == 'categorical' else TOP_LABELS\n"
    "    fig, axes = plt.subplots(1, 2, figsize=(17, max(4.5, 0.42 * len(order) + 1.8)))\n"
    "    sns.boxplot(data=d, y='label', x='n_tokens', order=order, palette=LABEL_COLOR, ax=axes[0], showfliers=False, width=0.6)\n"
    "    axes[0].set_xlabel('words')\n"
    "    axes[0].set_ylabel('')\n"
    "    axes[0].set_title(f'Words per document (Kruskal p = {p:.2g}, eps2 = {eps2:.3f})', loc='left')\n"
    "    sns.boxplot(data=d, y='label', x='ttr', order=order, palette=LABEL_COLOR, ax=axes[1], showfliers=False, width=0.6)\n"
    "    axes[1].set_ylabel('')\n"
    "    axes[1].set_xlabel('type-token ratio')\n"
    "    axes[1].set_title('Lexical diversity (higher = less repetitive)', loc='left')\n"
    "    suptitle(fig, f'Length and diversity by {LABEL_COL}', 'outliers hidden; labels sorted by median length')\n"
    "    save_fig(fig, 'length_by_label')\n"
    "    if LABEL_KIND == 'binned numeric':\n"
    "        rho, pr = stats.spearmanr(df['n_tokens'], df[LABEL_COL], nan_policy='omit')\n"
    "        print(f'Spearman rho between word count and {LABEL_COL}: {rho:+.3f} (p = {pr:.2g})')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* What are the median and p95 lengths? These set the truncation "
    "length for transformers. Does length or repetitiveness differ by label enough to be a feature?"
))

# ---------------------------------------------------------------------------
# Section 7: Vocabulary Statistics
# ---------------------------------------------------------------------------
cells.append(section("Vocabulary Statistics", "7"))

cells.append(md(
    "Natural language vocabularies follow Zipf's law (frequency is inversely proportional to rank) "
    "and Heaps' law (vocabulary grows sub-linearly with corpus size). Deviations reveal templated "
    "or synthetic text. The coverage curve answers a practical question: how many distinct words "
    "are needed to cover 90% or 95% of all tokens, which sets the vocabulary size for bag-of-words "
    "models."
))

cells.append(code(
    "COUNT_ALL = Counter(t for doc in TOK_ALL for t in doc)\n"
    "COUNT = Counter(t for doc in TOK for t in doc)\n"
    "freq = np.array(sorted(COUNT_ALL.values(), reverse=True), dtype=float)\n"
    "ranks = np.arange(1, len(freq) + 1)\n"
    "fit_mask = (ranks >= 10) & (ranks <= min(10_000, len(ranks)))\n"
    "zipf_slope = np.polyfit(np.log(ranks[fit_mask]), np.log(freq[fit_mask]), 1)[0] if fit_mask.sum() > 10 else np.nan\n"
    "\n"
    "seen, growth_x, growth_y, total = set(), [], [], 0\n"
    "step = max(1, len(TOK_ALL) // 200)\n"
    "for i, doc in enumerate(TOK_ALL):\n"
    "    seen.update(doc)\n"
    "    total += len(doc)\n"
    "    if i % step == 0:\n"
    "        growth_x.append(total)\n"
    "        growth_y.append(len(seen))\n"
    "gx, gy = np.array(growth_x[1:], dtype=float), np.array(growth_y[1:], dtype=float)\n"
    "gx, gy = gx[(gx > 0) & (gy > 0)], gy[(gx > 0) & (gy > 0)]\n"
    "heaps_beta = np.polyfit(np.log(gx), np.log(gy), 1)[0] if len(gx) > 5 else np.nan\n"
    "\n"
    "cov_all = np.cumsum(freq) / freq.sum()\n"
    "freq_c = np.array(sorted(COUNT.values(), reverse=True), dtype=float)\n"
    "cov_c = np.cumsum(freq_c) / freq_c.sum()\n"
    "coverage = pd.DataFrame({'all_tokens': [int(np.searchsorted(cov_all, q) + 1) for q in (0.5, 0.8, 0.9, 0.95, 0.99)],\n"
    "                         'content_tokens': [int(np.searchsorted(cov_c, q) + 1) for q in (0.5, 0.8, 0.9, 0.95, 0.99)]},\n"
    "                        index=['50%', '80%', '90%', '95%', '99%'])\n"
    "hapax = sum(1 for v in COUNT_ALL.values() if v == 1)\n"
    "vocab_stats = pd.Series({'tokens': int(freq.sum()), 'vocabulary': len(COUNT_ALL), 'content vocabulary': len(COUNT),\n"
    "                         'hapax legomena': hapax, 'hapax share of vocab': hapax / len(COUNT_ALL),\n"
    "                         'zipf slope': zipf_slope, 'heaps beta': heaps_beta}, name='value')\n"
    "save_table(vocab_stats.to_frame(), 'vocabulary_stats')\n"
    "save_table(coverage, 'vocabulary_coverage')\n"
    "display(vocab_stats.to_frame())\n"
    "display(coverage.rename_axis('tokens covered'))\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(19, 5))\n"
    "axes[0].loglog(ranks, freq, color=PRIMARY, lw=2)\n"
    "axes[0].loglog(ranks, freq[0] * ranks ** -1.0, color=ACCENT, ls='--', lw=1.2, label='ideal Zipf (slope -1)')\n"
    "axes[0].set_xlabel('rank')\n"
    "axes[0].set_ylabel('frequency')\n"
    "axes[0].legend()\n"
    "axes[0].set_title(f'Zipf plot (fitted slope {zipf_slope:.2f})', loc='left')\n"
    "axes[1].plot(gx, gy, color=PALETTE[2], lw=2.2)\n"
    "axes[1].set_xlabel('tokens seen')\n"
    "axes[1].set_ylabel('distinct words')\n"
    "axes[1].set_title(f'Heaps growth (beta {heaps_beta:.2f})', loc='left')\n"
    "axes[2].semilogx(np.arange(1, len(cov_all) + 1), cov_all * 100, color=PRIMARY, lw=2, label='all tokens')\n"
    "axes[2].semilogx(np.arange(1, len(cov_c) + 1), cov_c * 100, color=ACCENT, lw=2, label='content tokens')\n"
    "for q in (90, 95):\n"
    "    axes[2].axhline(q, color=MUTED, ls=':', lw=1)\n"
    "axes[2].set_xlabel('top-N words (log)')\n"
    "axes[2].set_ylabel('% of tokens covered')\n"
    "axes[2].legend(loc='lower right')\n"
    "axes[2].set_title(f'Coverage: {coverage.loc[\"95%\", \"content_tokens\"]:,} content words cover 95%', loc='left')\n"
    "suptitle(fig, 'Vocabulary statistics', f'{int(freq.sum()):,} tokens, {len(COUNT_ALL):,} distinct words, {hapax / len(COUNT_ALL):.0%} appear once')\n"
    "save_fig(fig, 'vocabulary_statistics')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Does the corpus follow Zipf (slope near -1)? A large hapax share means "
    "many typos, names, or rare words, which favours subword tokenisers. How many words give 95% "
    "coverage, which is a sensible `max_features` for TF-IDF?"
))

# ---------------------------------------------------------------------------
# Section 8: Word Clouds
# ---------------------------------------------------------------------------
cells.append(section("Word Clouds", "8"))

cells.append(md(
    "Word clouds are the fastest way to communicate what a corpus talks about, which makes them "
    "ideal for the presentation. Word size is proportional to frequency after stopword removal. "
    "The overall cloud uses the full palette, the bigram cloud shows recurring phrases, and the "
    "per-label clouds reuse each label's colour. Section 10 adds clouds of *distinctive* words, "
    "which are usually more informative than raw frequency."
))

cells.append(code(
    "def shades(color, n=6):\n"
    "    base, dark = np.array(to_rgb(color)), np.array(to_rgb(DARK))\n"
    "    return [to_hex(base * f + dark * (1 - f)) for f in np.linspace(0.55, 1.0, n)]\n"
    "\n"
    "def color_func_from(colors):\n"
    "    rng = random.Random(CFG.SEED)\n"
    "    return lambda *args, **kwargs: rng.choice(colors)\n"
    "\n"
    "def draw_cloud(ax, freqs, colors, title, max_words=None):\n"
    "    freqs = {k: float(v) for k, v in freqs.items() if v > 0}\n"
    "    if not freqs:\n"
    "        ax.set_visible(False)\n"
    "        return\n"
    "    wc = WordCloud(width=1600, height=900, background_color='white', max_words=max_words or CFG.WORDCLOUD_WORDS,\n"
    "                   prefer_horizontal=0.9, relative_scaling=0.5, min_font_size=8, random_state=CFG.SEED,\n"
    "                   color_func=color_func_from(colors), collocations=False, margin=4)\n"
    "    wc.generate_from_frequencies(freqs)\n"
    "    ax.imshow(wc, interpolation='bilinear')\n"
    "    ax.set_title(title, loc='left', fontsize=13)\n"
    "    ax.axis('off')\n"
    "\n"
    "BIGRAMS = Counter(f'{a} {b}' for doc in TOK for a, b in zip(doc, doc[1:]) if a != b)\n"
    "\n"
    "if not AVAILABLE['wordcloud']:\n"
    "    note('wordcloud is not installed, so word clouds are skipped (pip install wordcloud).')\n"
    "else:\n"
    "    fig, axes = plt.subplots(1, 2, figsize=(22, 7))\n"
    "    draw_cloud(axes[0], dict(COUNT.most_common(2000)), PALETTE, 'Most frequent words')\n"
    "    draw_cloud(axes[1], dict(BIGRAMS.most_common(1000)), PALETTE[::-1], 'Most frequent bigrams', max_words=100)\n"
    "    suptitle(fig, 'Corpus word clouds', f'{len(df):,} documents, stopwords removed (language: {CFG.LANGUAGE})')\n"
    "    save_fig(fig, 'wordcloud_overall')"
))

cells.append(md(
    "## Word Clouds per Label\n\n"
    "Each panel shows the most frequent content words for one label in that label's colour. "
    "Words that dominate every panel are corpus-wide and are good candidates for "
    "`EXTRA_STOPWORDS`."
))

cells.append(code(
    "LABEL_COUNTS = {}\n"
    "if HAS_LABEL:\n"
    "    lab_arr = df['label'].values\n"
    "    for l in TOP_LABELS:\n"
    "        LABEL_COUNTS[l] = Counter(t for doc, lb in zip(TOK, lab_arr) if lb == l for t in doc)\n"
    "if HAS_LABEL and AVAILABLE['wordcloud']:\n"
    "    fig, axes = make_grid(len(TOP_LABELS), ncols=2 if len(TOP_LABELS) <= 4 else 3, w=8.5, h=5.8)\n"
    "    for ax, l in zip(axes, TOP_LABELS):\n"
    "        n_docs = int((df['label'] == l).sum())\n"
    "        draw_cloud(ax, dict(LABEL_COUNTS[l].most_common(1500)), shades(LABEL_COLOR[l]), f'{short(l, 30)}  ({n_docs:,} docs)', max_words=100)\n"
    "    suptitle(fig, f'Word clouds by {LABEL_COL}', 'size = frequency within the label')\n"
    "    save_fig(fig, 'wordcloud_by_label')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* What themes dominate the corpus overall and per label? Which words "
    "appear everywhere and should be added to `EXTRA_STOPWORDS` before re-running?"
))

# ---------------------------------------------------------------------------
# Section 9: N-gram Analysis
# ---------------------------------------------------------------------------
cells.append(section("N-gram Analysis", "9"))

cells.append(md(
    "Bar charts give exact counts where word clouds give impressions. Unigrams show vocabulary, "
    "while bigrams and trigrams surface fixed phrases, negations, and templates (for example "
    "'not working' in reviews or a repeated chorus line in lyrics). Counts are document "
    "frequencies, so one very repetitive document cannot dominate."
))

cells.append(code(
    "def doc_freq(docs, n):\n"
    "    c = Counter()\n"
    "    for doc in docs:\n"
    "        grams = set(zip(*[doc[i:] for i in range(n)])) if n > 1 else set(doc)\n"
    "        c.update(' '.join(g) if n > 1 else g for g in grams)\n"
    "    return c\n"
    "\n"
    "NGRAMS = {n: doc_freq(TOK, n) for n in (1, 2, 3)}\n"
    "fig, axes = plt.subplots(1, 3, figsize=(20, 0.33 * CFG.TOP_N + 2.4))\n"
    "for ax, n, col in zip(axes, (1, 2, 3), (PRIMARY, ACCENT, PALETTE[2])):\n"
    "    top = pd.Series(dict(NGRAMS[n].most_common(CFG.TOP_N))).sort_values()\n"
    "    ax.barh([short(i, 30) for i in top.index], top.values / len(df) * 100, color=col)\n"
    "    bar_labels(ax, '{:.1f}%', size=7)\n"
    "    ax.set_xlabel('% of documents')\n"
    "    ax.set_title(['Unigrams', 'Bigrams', 'Trigrams'][n - 1], loc='left')\n"
    "    save_table(top[::-1].rename('documents').to_frame(), f'top_{n}grams')\n"
    "suptitle(fig, 'Most common n-grams', 'document frequency after stopword removal')\n"
    "save_fig(fig, 'ngrams_overall')"
))

cells.append(md(
    "## Bigrams per Label\n\n"
    "The most common phrases per label, shown as document share within that label, reveal how "
    "each group expresses itself."
))

cells.append(code(
    "if HAS_LABEL:\n"
    "    fig, axes = make_grid(len(TOP_LABELS), ncols=4, w=4.8, h=0.28 * 10 + 1.4)\n"
    "    for ax, l in zip(axes, TOP_LABELS):\n"
    "        docs = [d for d, lb in zip(TOK, df['label'].values) if lb == l]\n"
    "        top = pd.Series(dict(doc_freq(docs, 2).most_common(10))).sort_values() / max(len(docs), 1) * 100\n"
    "        ax.barh([short(i, 24) for i in top.index], top.values, color=LABEL_COLOR[l])\n"
    "        ax.set_title(short(l, 28), loc='left', fontsize=10)\n"
    "        ax.tick_params(axis='y', labelsize=8)\n"
    "        ax.set_xlabel('% of docs in label', fontsize=8)\n"
    "    suptitle(fig, f'Top bigrams by {LABEL_COL}', 'document share within each label')\n"
    "    save_fig(fig, 'bigrams_by_label')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Which phrases recur, and do any look like boilerplate that should be "
    "removed? Which phrases are characteristic of specific labels?"
))

# ---------------------------------------------------------------------------
# Section 10: Distinctive Words per Label
# ---------------------------------------------------------------------------
cells.append(section("Distinctive Words per Label", "10"))

cells.append(md(
    "Raw frequency is dominated by words that are common everywhere. Distinctiveness is measured "
    "with the weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi and "
    "Quinn, 2008), which compares each label against all other documents and shrinks rare words "
    "toward the corpus background, so a word seen twice cannot top the list. Counts are document "
    "frequencies (how many documents use the word), so one document that repeats a word dozens of "
    "times cannot make it look characteristic. The z-score is the evidence that a word is over-used "
    "by the label."
))

cells.append(md(
    "$$\\delta_w^{(i)} = \\log\\frac{y_w^{(i)} + \\alpha_w}{n^{(i)} + \\alpha_0 - y_w^{(i)} - \\alpha_w} "
    "- \\log\\frac{y_w^{(j)} + \\alpha_w}{n^{(j)} + \\alpha_0 - y_w^{(j)} - \\alpha_w}, \\qquad "
    "z_w = \\frac{\\delta_w^{(i)}}{\\sqrt{\\frac{1}{y_w^{(i)} + \\alpha_w} + \\frac{1}{y_w^{(j)} + \\alpha_w}}}$$\n\n"
    "where $y_w^{(i)}$ is the number of documents in label $i$ containing word $w$, $j$ is the rest of the corpus, and the "
    "prior $\\alpha_w$ is proportional to the background frequency of $w$."
))

cells.append(code(
    "DISTINCT = {}\n"
    "if not HAS_LABEL:\n"
    "    note('No label column, so distinctive-word analysis is skipped.')\n"
    "else:\n"
    "    cv = CountVectorizer(analyzer=lambda d: d, min_df=5, max_features=30_000, binary=True)\n"
    "    Xc = cv.fit_transform(TOK)\n"
    "    vocab = cv.get_feature_names_out()\n"
    "    total = np.asarray(Xc.sum(axis=0)).ravel().astype(float)\n"
    "    a0 = 10_000.0\n"
    "    alpha = total / total.sum() * a0\n"
    "    rows = []\n"
    "    for l in TOP_LABELS:\n"
    "        m = (df['label'] == l).values\n"
    "        yi = np.asarray(Xc[m].sum(axis=0)).ravel()\n"
    "        yj = total - yi\n"
    "        ni, nj = yi.sum(), yj.sum()\n"
    "        delta = np.log((yi + alpha) / (ni + a0 - yi - alpha)) - np.log((yj + alpha) / (nj + a0 - yj - alpha))\n"
    "        z = delta / np.sqrt(1 / (yi + alpha) + 1 / (yj + alpha))\n"
    "        DISTINCT[l] = pd.Series(z, index=vocab).sort_values(ascending=False)\n"
    "        for w in DISTINCT[l].index[:30]:\n"
    "            rows.append({'label': l, 'word': w, 'z': DISTINCT[l][w], 'docs_in_label': int(yi[cv.vocabulary_[w]])})\n"
    "    distinct_table = pd.DataFrame(rows)\n"
    "    save_table(distinct_table, 'distinctive_words', index=False)\n"
    "\n"
    "    fig, axes = make_grid(len(TOP_LABELS), ncols=4, w=4.6, h=0.27 * 12 + 1.4)\n"
    "    for ax, l in zip(axes, TOP_LABELS):\n"
    "        top = DISTINCT[l].head(12)[::-1]\n"
    "        ax.barh(top.index, top.values, color=LABEL_COLOR[l])\n"
    "        ax.set_title(short(l, 28), loc='left', fontsize=10)\n"
    "        ax.tick_params(axis='y', labelsize=8)\n"
    "        ax.set_xlabel('log-odds z-score', fontsize=8)\n"
    "    suptitle(fig, f'Most distinctive words by {LABEL_COL}', 'weighted log-odds with informative Dirichlet prior, each label vs the rest')\n"
    "    save_fig(fig, 'distinctive_words')"
))

cells.append(md(
    "## Distinctive Word Clouds\n\n"
    "The same z-scores drive a second set of word clouds where size reflects how characteristic a "
    "word is rather than how frequent. These are usually the clearest visual for a slide about "
    "'the lyrical profile of each genre' or 'what each customer segment complains about'."
))

cells.append(code(
    "if DISTINCT and AVAILABLE['wordcloud']:\n"
    "    fig, axes = make_grid(len(TOP_LABELS), ncols=2 if len(TOP_LABELS) <= 4 else 3, w=8.5, h=5.8)\n"
    "    for ax, l in zip(axes, TOP_LABELS):\n"
    "        z = DISTINCT[l]\n"
    "        draw_cloud(ax, z[z > 1.96].head(150).to_dict(), shades(LABEL_COLOR[l]), f'{short(l, 30)}: distinctive words', max_words=100)\n"
    "    suptitle(fig, f'Distinctive word clouds by {LABEL_COL}', 'size = log-odds z-score (only z > 1.96, i.e. significant over-use)')\n"
    "    save_fig(fig, 'wordcloud_distinctive')"
))

cells.append(md(
    "## Two-Label Comparison\n\n"
    "For the two largest labels, each word is placed by its frequency in label A (x) and label B (y). "
    "Words far from the diagonal belong to one side. The most distinctive words on each side are "
    "annotated, in the spirit of a Scattertext plot."
))

cells.append(code(
    "if HAS_LABEL and len(TOP_LABELS) >= 2:\n"
    "    A, B = TOP_LABELS[0], TOP_LABELS[1]\n"
    "    ca, cb = LABEL_COUNTS[A], LABEL_COUNTS[B]\n"
    "    words = [w for w, _ in (ca + cb).most_common(3000)]\n"
    "    na, nb = sum(ca.values()), sum(cb.values())\n"
    "    fa = np.array([(ca[w] + 1) / na * 1e4 for w in words])\n"
    "    fb = np.array([(cb[w] + 1) / nb * 1e4 for w in words])\n"
    "    lr = np.log2(fa / fb)\n"
    "    fig, ax = plt.subplots(figsize=(11, 10))\n"
    "    sc = ax.scatter(fa, fb, c=lr, cmap=LinearSegmentedColormap.from_list('ab', [LABEL_COLOR[B], '#E2E8F0', LABEL_COLOR[A]]),\n"
    "                    vmin=-3, vmax=3, s=12, alpha=0.75)\n"
    "    ax.set_xscale('log')\n"
    "    ax.set_yscale('log')\n"
    "    lims = [min(fa.min(), fb.min()), max(fa.max(), fb.max())]\n"
    "    ax.plot(lims, lims, color=MUTED, ls='--', lw=1)\n"
    "    weight = np.log1p(fa + fb)\n"
    "    for idx in list(np.argsort(-lr * weight)[:12]) + list(np.argsort(lr * weight)[:12]):\n"
    "        ax.annotate(words[idx], (fa[idx], fb[idx]), fontsize=8, color=DARK, xytext=(3, 3), textcoords='offset points')\n"
    "    ax.set_xlabel(f'frequency in {A} (per 10k tokens)')\n"
    "    ax.set_ylabel(f'frequency in {B} (per 10k tokens)')\n"
    "    fig.colorbar(sc, ax=ax, shrink=0.6, label=f'log2 ratio ({A} / {B})')\n"
    "    suptitle(fig, f'{A} vs {B}', 'top 3,000 words; annotated = most one-sided frequent words')\n"
    "    save_fig(fig, 'two_label_comparison')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Which words define each label? These distinctive vocabularies are "
    "the core of a 'profile per category' narrative and also predict which labels a bag-of-words "
    "model will separate easily (strong, unique vocabulary) or confuse (overlapping vocabulary)."
))

# ---------------------------------------------------------------------------
# Section 11: Sentiment Analysis
# ---------------------------------------------------------------------------
cells.append(section("Sentiment Analysis", "11"))

cells.append(md(
    "VADER is a rule-based sentiment lexicon tuned for short, informal English text. It needs no "
    "training and returns a compound score from -1 (most negative) to +1 (most positive), with "
    "+/-0.05 as the conventional neutral band. It is applied to a sample of `SENTIMENT_SAMPLE` "
    "documents. For Indonesian text a lexicon is unreliable, and a fine-tuned IndoBERT sentiment "
    "model in the NLP modelling notebook is the right tool."
))

cells.append(code(
    "SENT = None\n"
    "if not AVAILABLE['vader']:\n"
    "    note('vaderSentiment is not installed, so sentiment analysis is skipped.')\n"
    "elif CFG.LANGUAGE == 'id':\n"
    "    note('LANGUAGE is Indonesian, and VADER is English-only, so lexicon sentiment is skipped.')\n"
    "else:\n"
    "    vader = SentimentIntensityAnalyzer()\n"
    "    s_idx = df.sample(min(CFG.SENTIMENT_SAMPLE, len(df)), random_state=CFG.SEED).index\n"
    "    def strip_patterns(t):\n"
    "        for rx in REMOVE_RES:\n"
    "            t = rx.sub(' ', t)\n"
    "        return ' '.join(t.split())\n"
    "    texts = [strip_patterns(t) for t in df.loc[s_idx, 'text']]\n"
    "    SENT = pd.DataFrame([vader.polarity_scores(t[:3000]) for t in texts], index=s_idx)\n"
    "    SENT['polarity'] = pd.cut(SENT['compound'], [-1.01, -0.05, 0.05, 1.01], labels=['negative', 'neutral', 'positive'])\n"
    "    if HAS_LABEL:\n"
    "        SENT['label'] = df.loc[s_idx, 'label']\n"
    "    save_table(SENT, 'vader_scores')\n"
    "    share = SENT['polarity'].value_counts(normalize=True).reindex(['negative', 'neutral', 'positive']) * 100\n"
    "    print('Polarity share (%):', share.round(1).to_dict())\n"
    "\n"
    "    pol_color = {'negative': ACCENT, 'neutral': '#C9D1DB', 'positive': PALETTE[2]}\n"
    "    ncols = 2 if HAS_LABEL else 1\n"
    "    fig, axes = plt.subplots(1, ncols, figsize=(8.5 * ncols, 5.2), squeeze=False)\n"
    "    ax = axes[0, 0]\n"
    "    sns.histplot(SENT['compound'], bins=40, color=PRIMARY, edgecolor='white', ax=ax)\n"
    "    ax.axvspan(-0.05, 0.05, color='#E2E8F0', alpha=0.8)\n"
    "    ax.set_title(f'Compound score: {share[\"positive\"]:.0f}% pos, {share[\"neutral\"]:.0f}% neu, {share[\"negative\"]:.0f}% neg', loc='left')\n"
    "    if HAS_LABEL:\n"
    "        d = SENT[SENT['label'].isin(TOP_LABELS)]\n"
    "        ct = pd.crosstab(d['label'], d['polarity'], normalize='index').reindex(columns=['negative', 'neutral', 'positive']).fillna(0)\n"
    "        ct = ct.sort_values('positive')\n"
    "        left = np.zeros(len(ct))\n"
    "        for pcol in ct.columns:\n"
    "            axes[0, 1].barh([short(i, 22) for i in ct.index], ct[pcol], left=left, color=pol_color[pcol], label=pcol, edgecolor='white')\n"
    "            left += ct[pcol].values\n"
    "        axes[0, 1].set_xlim(0, 1)\n"
    "        axes[0, 1].legend(loc='lower right', fontsize=8)\n"
    "        axes[0, 1].set_title('Polarity mix per label (sorted by positive share)', loc='left')\n"
    "        save_table(ct, 'polarity_by_label')\n"
    "    suptitle(fig, 'Lexicon sentiment (VADER)', f'{len(SENT):,} sampled documents, first 3,000 characters each')\n"
    "    save_fig(fig, 'sentiment')\n"
    "    if HAS_LABEL and LABEL_KIND == 'binned numeric':\n"
    "        rho, p = stats.spearmanr(SENT['compound'], df.loc[s_idx, LABEL_COL], nan_policy='omit')\n"
    "        print(f'Spearman rho between compound sentiment and {LABEL_COL}: {rho:+.3f} (p = {p:.2g})')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* What is the overall tone of the corpus, and which labels are the most "
    "positive or negative? Remember that lexicons miss sarcasm and domain meaning (for example "
    "'killer' as praise), so treat these as indicative and validate with examples."
))

# ---------------------------------------------------------------------------
# Section 12: Topic Modeling
# ---------------------------------------------------------------------------
cells.append(section("Topic Modeling", "12"))

cells.append(md(
    "Non-negative Matrix Factorisation on TF-IDF decomposes the corpus into `TOPIC_K` additive "
    "topics, each a weighted list of words, and gives every document a mixture of topics. NMF is "
    "fast, deterministic with a fixed seed, and produces more coherent topics than LDA on short or "
    "medium texts. Words that appear in more than half of the documents are excluded so topics are "
    "not dominated by generic vocabulary."
))

cells.append(code(
    "t_idx = df.sample(min(20_000, len(df)), random_state=CFG.SEED).index\n"
    "tfv = TfidfVectorizer(analyzer=lambda d: d, min_df=5, max_df=0.5, max_features=20_000, sublinear_tf=True)\n"
    "Xt = tfv.fit_transform([TOK[i] for i in t_idx])\n"
    "nmf = NMF(n_components=CFG.TOPIC_K, init='nndsvda', random_state=CFG.SEED, max_iter=400)\n"
    "W = nmf.fit_transform(Xt)\n"
    "H = nmf.components_\n"
    "terms = tfv.get_feature_names_out()\n"
    "TOPIC_WORDS = [[terms[j] for j in np.argsort(-H[k])[:10]] for k in range(CFG.TOPIC_K)]\n"
    "TOPIC_NAMES = [f'T{k + 1}: ' + ', '.join(w[:3]) for k, w in enumerate(TOPIC_WORDS)]\n"
    "Wn = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)\n"
    "topics = pd.DataFrame({'topic': TOPIC_NAMES, 'top_words': [', '.join(w) for w in TOPIC_WORDS],\n"
    "                       'prevalence_%': Wn.mean(axis=0) * 100}).set_index('topic')\n"
    "save_table(topics, 'nmf_topics')\n"
    "display(topics.round(2))\n"
    "\n"
    "fig, axes = make_grid(CFG.TOPIC_K, ncols=4, w=4.3, h=3.5)\n"
    "for k, ax in enumerate(axes):\n"
    "    idx = np.argsort(-H[k])[:10][::-1]\n"
    "    ax.barh(terms[idx], H[k][idx], color=PALETTE[k % 10])\n"
    "    ax.set_title(f'Topic {k + 1} ({topics[\"prevalence_%\"].iloc[k]:.1f}%)', loc='left', fontsize=10)\n"
    "    ax.tick_params(axis='y', labelsize=8)\n"
    "    ax.set_xticks([])\n"
    "suptitle(fig, f'NMF topics (k = {CFG.TOPIC_K})', f'top 10 words per topic; % = mean topic share across {len(t_idx):,} documents')\n"
    "save_fig(fig, 'nmf_topics')"
))

cells.append(md(
    "## Topic Mix per Label\n\n"
    "The heatmap shows the average topic share for each label, normalised so each row sums to 1. "
    "Bright cells mark the themes a label over-represents."
))

cells.append(code(
    "if HAS_LABEL:\n"
    "    labs = df.loc[t_idx, 'label'].values\n"
    "    rows = {l: Wn[labs == l].mean(axis=0) for l in TOP_LABELS if (labs == l).any()}\n"
    "    mix = pd.DataFrame(rows, index=TOPIC_NAMES).T\n"
    "    lift = mix / Wn.mean(axis=0)\n"
    "    save_table(mix, 'topic_mix_by_label')\n"
    "    fig, ax = plt.subplots(figsize=(max(10, 1.4 * CFG.TOPIC_K + 3), 0.5 * len(mix) + 2.4))\n"
    "    sns.heatmap(lift, cmap=CMAP_HEAT, annot=mix * 100, fmt='.0f', linewidths=0.5, linecolor='white',\n"
    "                cbar_kws={'label': 'lift vs corpus average', 'shrink': 0.7}, ax=ax,\n"
    "                xticklabels=[short(t, 24) for t in TOPIC_NAMES], yticklabels=[short(l, 22) for l in mix.index])\n"
    "    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)\n"
    "    ax.set_ylabel('')\n"
    "    suptitle(fig, f'Topic mix by {LABEL_COL}', 'cell text = % topic share, colour = lift over the corpus average')\n"
    "    save_fig(fig, 'topic_mix_by_label')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Name each topic in plain language. Which labels are dominated by "
    "which themes? If topics look incoherent, try another `TOPIC_K` or add generic words to "
    "`EXTRA_STOPWORDS`."
))

# ---------------------------------------------------------------------------
# Section 13: Semantic Map
# ---------------------------------------------------------------------------
cells.append(section("Semantic Map", "13"))

cells.append(md(
    "A 2-D map shows whether labels occupy distinct regions of meaning. Documents are embedded with "
    "TF-IDF, compressed to 50 dimensions with truncated SVD (latent semantic analysis), and "
    "projected with t-SNE. Well-separated clusters predict an easy classification task, while "
    "heavy overlap predicts confusion between labels."
))

cells.append(code(
    "m_idx = df[df['n_content'] > 0].sample(min(CFG.TSNE_SAMPLE, int((df['n_content'] > 0).sum())), random_state=CFG.SEED).index\n"
    "Xm = TfidfVectorizer(analyzer=lambda d: d, min_df=3, max_df=0.5, sublinear_tf=True).fit_transform([TOK[i] for i in m_idx])\n"
    "Z = TruncatedSVD(n_components=min(50, Xm.shape[1] - 1), random_state=CFG.SEED).fit_transform(Xm)\n"
    "Y = TSNE(n_components=2, perplexity=30, init='pca', random_state=CFG.SEED).fit_transform(Z)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(12, 9))\n"
    "if HAS_LABEL:\n"
    "    labs = df.loc[m_idx, 'label_top'].values\n"
    "    other = labs == '(other)'\n"
    "    ax.scatter(Y[other, 0], Y[other, 1], s=6, color='#D5DBE3', alpha=0.5, label='(other)')\n"
    "    for l in TOP_LABELS:\n"
    "        m = labs == l\n"
    "        ax.scatter(Y[m, 0], Y[m, 1], s=9, color=LABEL_COLOR[l], alpha=0.75, label=short(l, 20))\n"
    "    ax.legend(markerscale=2.5, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))\n"
    "else:\n"
    "    dom = nmf.transform(tfv.transform([TOK[i] for i in m_idx])).argmax(axis=1)\n"
    "    for k in range(CFG.TOPIC_K):\n"
    "        m = dom == k\n"
    "        ax.scatter(Y[m, 0], Y[m, 1], s=9, color=PALETTE[k % 10], alpha=0.75, label=short(TOPIC_NAMES[k], 26))\n"
    "    ax.legend(markerscale=2.5, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))\n"
    "ax.set_xticks([])\n"
    "ax.set_yticks([])\n"
    "colour_by = LABEL_COL if HAS_LABEL else 'dominant NMF topic'\n"
    "suptitle(fig, 'Semantic map (TF-IDF -> SVD -> t-SNE)', f'{len(m_idx):,} sampled documents coloured by {colour_by}')\n"
    "save_fig(fig, 'semantic_map')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Do labels form separate clusters or one mixed cloud? Overlap at the "
    "bag-of-words level suggests that contextual embeddings (transformers) will be needed to "
    "separate them."
))

# ---------------------------------------------------------------------------
# Section 14: Duplicates and Label Noise
# ---------------------------------------------------------------------------
cells.append(section("Duplicates and Label Noise", "14"))

cells.append(md(
    "Duplicate texts inflate validation scores when copies fall into both training and validation "
    "folds, and identical texts with different labels cap the achievable accuracy. Exact duplicates "
    "are found after normalisation. Near duplicates (reposts, small edits, remixes) are found with "
    "cosine similarity of TF-IDF word and bigram vectors on a sample."
))

cells.append(code(
    "norm = df['clean'].str.split().str.join(' ')\n"
    "valid = norm != ''\n"
    "grp = df[valid].groupby(norm[valid])\n"
    "sizes = grp.size()\n"
    "dup_groups = sizes[sizes > 1]\n"
    "print(f'Exact duplicate groups: {len(dup_groups):,} covering {dup_groups.sum():,} documents ({dup_groups.sum() / len(df):.1%})')\n"
    "CONFLICTS = 0\n"
    "if HAS_LABEL and len(dup_groups):\n"
    "    n_lab = grp['label'].nunique()\n"
    "    conflict_keys = n_lab[n_lab > 1].index\n"
    "    CONFLICTS = len(conflict_keys)\n"
    "    print(f'Duplicate groups with conflicting labels: {CONFLICTS:,} ({CONFLICTS / max(len(dup_groups), 1):.1%} of duplicate groups)')\n"
    "    ex = [{'text': short(k, 90), 'copies': int(sizes[k]), 'labels': ', '.join(sorted(df.loc[grp.groups[k], 'label'].unique()))[:80]}\n"
    "          for k in list(conflict_keys[:8])]\n"
    "    if ex:\n"
    "        display(pd.DataFrame(ex))\n"
    "\n"
    "n_idx = df[valid].sample(min(CFG.NEAR_DUP_SAMPLE, int(valid.sum())), random_state=CFG.SEED).index\n"
    "Xn = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(norm[n_idx])\n"
    "nn = NearestNeighbors(n_neighbors=2, metric='cosine').fit(Xn)\n"
    "dist, nbr = nn.kneighbors(Xn)\n"
    "sim = 1 - dist[:, 1]\n"
    "exact = norm[n_idx].values == norm[n_idx].values[nbr[:, 1]]\n"
    "near = (sim >= CFG.NEAR_DUP_SIM) & ~exact\n"
    "NEAR_SHARE = near.mean()\n"
    "print(f'Near duplicates (cosine >= {CFG.NEAR_DUP_SIM}, not exact) in sample: {near.sum():,} ({NEAR_SHARE:.1%})')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(11, 4.4))\n"
    "sns.histplot(sim, bins=50, color=PRIMARY, edgecolor='white', ax=ax)\n"
    "ax.axvline(CFG.NEAR_DUP_SIM, color=ACCENT, ls='--', lw=1.5, label=f'near-duplicate threshold {CFG.NEAR_DUP_SIM}')\n"
    "ax.set_xlabel('cosine similarity to nearest other document')\n"
    "ax.legend()\n"
    "suptitle(fig, 'Nearest-neighbour similarity', f'{len(n_idx):,} sampled documents; mass near 1.0 = duplicated content')\n"
    "save_fig(fig, 'near_duplicates')\n"
    "\n"
    "pairs = [{'similarity': round(float(sim[i]), 3), 'text_a': short(norm[n_idx[i]], 70), 'text_b': short(norm[n_idx[nbr[i, 1]]], 70),\n"
    "          **({'label_a': df.loc[n_idx[i], 'label'], 'label_b': df.loc[n_idx[nbr[i, 1]], 'label']} if HAS_LABEL else {})}\n"
    "         for i in np.where(near)[0][:8]]\n"
    "if pairs:\n"
    "    display(pd.DataFrame(pairs))"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* How much of the corpus is duplicated, and do duplicates carry "
    "conflicting labels? Duplicates must be removed or kept together with a group-aware split. "
    "Conflicting labels set an upper bound on accuracy and should be reported as label noise."
))

# ---------------------------------------------------------------------------
# Section 15: Segment and Temporal Analysis
# ---------------------------------------------------------------------------
cells.append(section("Segment and Temporal Analysis", "15"))

cells.append(md(
    "When a segment column (`GROUP_COL`) or a time column (`TIME_COL`) is available, text properties "
    "are tracked across them: document volume, length, the usage of the most frequent words, and "
    "the label mix. This reveals drift, for example reviews getting longer after an app update or "
    "vocabulary shifting over the years."
))

cells.append(code(
    "HAS_TIME = bool(CFG.TIME_COL) and CFG.TIME_COL in df.columns\n"
    "if HAS_TIME:\n"
    "    tc = df[CFG.TIME_COL]\n"
    "    if pd.api.types.is_numeric_dtype(tc):\n"
    "        period = tc.where(tc.between(*CFG.TIME_RANGE)).round()\n"
    "        if period.nunique() > 30:\n"
    "            period = (period // 5 * 5)\n"
    "            p_label = f'{CFG.TIME_COL} (5-year bins)'\n"
    "        else:\n"
    "            p_label = CFG.TIME_COL\n"
    "    else:\n"
    "        dt = pd.to_datetime(tc, errors='coerce')\n"
    "        span = (dt.max() - dt.min()).days\n"
    "        freq = 'D' if span <= 90 else 'W' if span <= 730 else 'M' if span <= 3650 else 'Y'\n"
    "        period = dt.dt.to_period(freq).dt.to_timestamp()\n"
    "        p_label = f'{CFG.TIME_COL} ({freq})'\n"
    "    df['period'] = period\n"
    "    d = df.dropna(subset=['period'])\n"
    "    agg = d.groupby('period').agg(docs=('text', 'size'), median_words=('n_tokens', 'median'),\n"
    "                                  q25=('n_tokens', lambda s: s.quantile(0.25)), q75=('n_tokens', lambda s: s.quantile(0.75)))\n"
    "    agg = agg[agg['docs'] >= 10]\n"
    "    top_words = [w for w, _ in COUNT.most_common(6)]\n"
    "    rate = {}\n"
    "    for p_, g in d.groupby('period'):\n"
    "        if len(g) < 10:\n"
    "            continue\n"
    "        c = Counter(t for i in g.index for t in TOK[i])\n"
    "        n_tok = max(sum(c.values()), 1)\n"
    "        rate[p_] = {w: c[w] / n_tok * 1000 for w in top_words}\n"
    "    rate = pd.DataFrame(rate).T.sort_index()\n"
    "\n"
    "    ncols = 3 if HAS_LABEL else 2\n"
    "    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))\n"
    "    width = 3.5 if p_label.endswith('bins)') else 0.8\n"
    "    if pd.api.types.is_datetime64_any_dtype(agg.index):\n"
    "        width = 20\n"
    "    axes[0].bar(agg.index, agg['docs'], color=SOFT, width=width)\n"
    "    ax2 = axes[0].twinx()\n"
    "    ax2.fill_between(agg.index, agg['q25'], agg['q75'], color=ACCENT, alpha=0.15)\n"
    "    ax2.plot(agg.index, agg['median_words'], color=ACCENT, marker='o', ms=3, lw=2)\n"
    "    ax2.set_ylabel('median words (band = IQR)', color=ACCENT)\n"
    "    ax2.grid(False)\n"
    "    axes[0].set_ylabel('documents')\n"
    "    axes[0].set_title('Volume and length over time', loc='left')\n"
    "    for i, w in enumerate(top_words):\n"
    "        axes[1].plot(rate.index, rate[w], marker='o', ms=3, lw=1.8, color=PALETTE[i % 10], label=w)\n"
    "    axes[1].set_ylabel('uses per 1,000 content tokens')\n"
    "    axes[1].legend(fontsize=8)\n"
    "    axes[1].set_title('Top words over time', loc='left')\n"
    "    if HAS_LABEL:\n"
    "        ct = pd.crosstab(d['period'], d['label_top'], normalize='index')\n"
    "        ct = ct.loc[ct.index.isin(agg.index)]\n"
    "        cols = [l for l in TOP_LABELS if l in ct.columns] + (['(other)'] if '(other)' in ct.columns else [])\n"
    "        axes[2].stackplot(ct.index, ct[cols].T.values, colors=[LABEL_COLOR.get(c, '#D5DBE3') for c in cols],\n"
    "                          labels=[short(c, 18) for c in cols], alpha=0.9)\n"
    "        axes[2].set_ylim(0, 1)\n"
    "        axes[2].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))\n"
    "        axes[2].set_title('Label mix over time', loc='left')\n"
    "    for ax in axes:\n"
    "        ax.set_xlabel(p_label)\n"
    "    suptitle(fig, f'Text over {CFG.TIME_COL}', 'periods with fewer than 10 documents are hidden')\n"
    "    save_fig(fig, 'temporal_text')\n"
    "    save_table(agg.join(rate), 'text_over_time')\n"
    "else:\n"
    "    note('TIME_COL is not set or not found, so temporal analysis is skipped.')\n"
    "\n"
    "if CFG.GROUP_COL and CFG.GROUP_COL in df.columns:\n"
    "    g = df.groupby(CFG.GROUP_COL)\n"
    "    seg = g.agg(docs=('text', 'size'), median_words=('n_tokens', 'median'), ttr=('ttr', 'median')).sort_values('docs', ascending=False).head(20)\n"
    "    save_table(seg, 'text_by_group')\n"
    "    display(seg)\n"
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 0.36 * len(seg) + 2))\n"
    "    axes[0].barh([short(i, 22) for i in seg.index[::-1]], seg['docs'][::-1], color=PRIMARY)\n"
    "    axes[0].set_title('Documents per group', loc='left')\n"
    "    axes[1].barh([short(i, 22) for i in seg.index[::-1]], seg['median_words'][::-1], color=ACCENT)\n"
    "    axes[1].set_title('Median words per group', loc='left')\n"
    "    suptitle(fig, f'Text by {CFG.GROUP_COL}', 'top 20 groups by size')\n"
    "    save_fig(fig, 'text_by_group')"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Does the language or the label mix drift over time or across "
    "segments? If so, a random split overstates performance on future data and a time-based or "
    "group-based split is more honest."
))

# ---------------------------------------------------------------------------
# Section 16: EDA Summary and Modelling Recommendations
# ---------------------------------------------------------------------------
cells.append(section("EDA Summary and Modelling Recommendations", "16"))

cells.append(md(
    "The key findings are compiled automatically and translated into concrete modelling settings: "
    "the transformer `max_length` (p95 of words times about 1.3 subword tokens per word), the "
    "TF-IDF vocabulary size (words for 95% coverage), the cleaning steps implied by the noise "
    "audit, and the handling of imbalance and duplicates. The summary is saved as "
    "`eda-output/eda_summary.md`."
))

cells.append(code(
    "p95 = df['n_tokens'].quantile(0.95)\n"
    "max_len = int(min(512, 2 ** np.ceil(np.log2(max(p95 * 1.3, 16)))))\n"
    "lines = [f'# NLP EDA Summary: {DATA_PATH.name} (`{TEXT_COL}`)', '', '## Corpus',\n"
    "         f'- {len(df):,} documents analysed (file has {N_TOTAL:,}); median {df.n_tokens.median():.0f} words, p95 {p95:.0f}, max {df.n_tokens.max():,}',\n"
    "         f'- Empty documents: {int(overview.loc[\"empty or whitespace\", \"value\"]):,}; exact duplicates: {int(overview.loc[\"duplicates after normalisation\", \"value\"]):,} '\n"
    "         f'({overview.loc[\"duplicates after normalisation\", \"share_%\"]:.1f}%); near duplicates in sample: {NEAR_SHARE:.1%}',\n"
    "         f'- Language guess: ' + ', '.join(f'{k} {v / len(df):.0%}' for k, v in lang.items()),\n"
    "         f'- Vocabulary: {len(COUNT_ALL):,} words, {hapax / len(COUNT_ALL):.0%} hapax, Zipf slope {zipf_slope:.2f}', '']\n"
    "if HAS_LABEL:\n"
    "    vc = df['label'].value_counts()\n"
    "    lines += ['## Labels', f'- `{LABEL_COL}` ({LABEL_KIND}): {len(vc)} labels, imbalance {vc.max() / vc.min():.1f}x',\n"
    "              f'- Duplicate groups with conflicting labels: {CONFLICTS:,}']\n"
    "    for l in TOP_LABELS[:8]:\n"
    "        if l in DISTINCT:\n"
    "            lines.append(f'- **{l}** distinctive words: ' + ', '.join(DISTINCT[l].index[:8]))\n"
    "    lines.append('')\n"
    "if SENT is not None:\n"
    "    sh = SENT['polarity'].value_counts(normalize=True)\n"
    "    lines += ['## Sentiment (VADER)', f'- positive {sh.get(\"positive\", 0):.0%}, neutral {sh.get(\"neutral\", 0):.0%}, negative {sh.get(\"negative\", 0):.0%}', '']\n"
    "lines += ['## Topics (NMF)'] + [f'- {t}: {w}' for t, w in topics['top_words'].items()] + ['']\n"
    "\n"
    "recs = [f'Transformer `max_length` = {max_len} tokens (p95 = {p95:.0f} words x 1.3 subwords/word, capped at 512)'\n"
    "        + ('; consider head+tail truncation or chunking for long documents' if p95 * 1.3 > 512 else ''),\n"
    "        f'TF-IDF `max_features` around {int(coverage.loc[\"95%\", \"content_tokens\"]):,} (95% content-token coverage)']\n"
    "heavy = noise[noise['docs_%'] > 1].index.tolist()\n"
    "if heavy:\n"
    "    recs.append('Handle frequent noise patterns: ' + ', '.join(heavy))\n"
    "if HAS_LABEL:\n"
    "    vc = df['label'].value_counts()\n"
    "    if vc.max() / vc.min() > 3:\n"
    "        recs.append(f'Imbalance {vc.max() / vc.min():.1f}x: use stratified folds, class weights, and macro F1')\n"
    "if overview.loc['duplicates after normalisation', 'share_%'] > 1 or NEAR_SHARE > 0.01:\n"
    "    recs.append('Deduplicate, or split with groups of identical / near-identical texts, to avoid leakage')\n"
    "if CONFLICTS:\n"
    "    recs.append(f'{CONFLICTS:,} duplicate texts carry different labels: treat as label noise (resolve, drop, or use multi-label)')\n"
    "other_lang = 1 - (lang.get('English', 0) if CFG.LANGUAGE == 'en' else lang.get('Indonesian', 0) if CFG.LANGUAGE == 'id' else lang.get('English', 0) + lang.get('Indonesian', 0)) / len(df)\n"
    "if other_lang > 0.1:\n"
    "    recs.append(f'{other_lang:.0%} of documents are outside the configured language: prefer a multilingual encoder (e.g. XLM-R, mE5)')\n"
    "lines += ['## Modelling recommendations'] + [f'{i + 1}. {r}' for i, r in enumerate(recs)]\n"
    "lines += ['', '## Exported files', f'- Figures ({len(SAVED_FIGS)}): `{CFG.FIG_DIR}`', f'- Tables: `{CFG.TABLE_DIR}`']\n"
    "summary_md = '\\n'.join(lines)\n"
    "(CFG.OUTPUT_DIR / 'eda_summary.md').write_text(summary_md, encoding='utf-8')\n"
    "display(Markdown(summary_md))"
))

cells.append(md(
    "## Decisions File for the Pipeline\n\n"
    "The key decisions are also written as machine-readable JSON to `eda-output/eda_decisions.json`. "
    "The NLP pipeline notebook reads this file and uses it for every setting left on `'auto'` "
    "(language and model family, `max_length`, TF-IDF size, cleaning patterns, class weighting, "
    "and duplicate handling), so the modelling choices are traceable to EDA evidence."
))

cells.append(code(
    "lang_share = (lang / len(df)).to_dict()\n"
    "en_s, id_s = lang_share.get('English', 0), lang_share.get('Indonesian', 0)\n"
    "dominant = 'id' if id_s >= 0.6 else 'en' if en_s >= 0.6 else 'multi'\n"
    "family = {'id': 'indobert', 'en': 'deberta-v3', 'multi': 'xlm-roberta'}[dominant]\n"
    "imb = float(df['label'].value_counts().max() / df['label'].value_counts().min()) if HAS_LABEL else None\n"
    "decisions = {\n"
    "    'source': str(DATA_PATH), 'text_col': TEXT_COL, 'label_col': LABEL_COL, 'label_kind': LABEL_KIND,\n"
    "    'language': dominant, 'language_share': {k: round(float(v), 4) for k, v in lang_share.items()},\n"
    "    'recommended_model': family, 'max_length': max_len,\n"
    "    'words_p50': float(df['n_tokens'].median()), 'words_p95': float(p95),\n"
    "    'tfidf_max_features': int(coverage.loc['95%', 'content_tokens']),\n"
    "    'remove_patterns': list(CFG.REMOVE_PATTERNS), 'extra_stopwords': list(CFG.EXTRA_STOPWORDS),\n"
    "    'noise_patterns': heavy, 'imbalance_ratio': imb, 'class_weights': bool(imb and imb > 3),\n"
    "    'duplicate_share': float(overview.loc['duplicates after normalisation', 'share_%'] / 100),\n"
    "    'near_duplicate_share': float(NEAR_SHARE), 'conflicting_label_groups': int(CONFLICTS),\n"
    "    'group_identical_texts': bool(overview.loc['duplicates after normalisation', 'share_%'] > 1 or NEAR_SHARE > 0.01),\n"
    "}\n"
    "(CFG.OUTPUT_DIR / 'eda_decisions.json').write_text(json.dumps(decisions, indent=2), encoding='utf-8')\n"
    "print(json.dumps(decisions, indent=2))"
))

cells.append(md(
    "The final cell lists every exported figure and table."
))

cells.append(code(
    "print(f'Figures in {CFG.FIG_DIR.resolve()}:')\n"
    "for f in SAVED_FIGS:\n"
    "    print('  ', f)\n"
    "print(f'\\nTables in {CFG.TABLE_DIR.resolve()}:')\n"
    "for f in sorted(CFG.TABLE_DIR.glob('*.csv')):\n"
    "    print('  ', f.name)"
))

cells.append(md(
    "#### Insights\n\n"
    "> *Fill in after running.* Summarise the three to five findings that shape preprocessing, "
    "the model choice, validation, and the business narrative, each linked to its figure."
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
