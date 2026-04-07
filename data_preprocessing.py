#   SQL:
#     Train.csv + Validation.csv  →  SQL training pool
#     Test.csv                    →  SQL internal test
#     Modified_SQL_Dataset.csv    →  Cross-dataset held-out test (NEVER seen
#                                    during training, val, or internal test)
#
#   XSS:
#       70% → XSS train pool
#       15% → XSS val pool
#       15% → XSS internal test
#     (Only one XSS dataset available; cross-dataset validation for XSS
#      is noted as a limitation in the paper.)
#
#   Final unified splits:
#     X_train  = SQL train pool  + XSS train pool
#     X_val    = SQL val*        + XSS val
#     X_test   = SQL Test.csv    + XSS internal test   ← internal benchmark
#     X_cross  = Modified_SQL_Dataset.csv              ← cross-dataset benchmark
#


from config import GLOBAL_CONFIG
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
import gc

warnings.filterwarnings('ignore')



SQL_KEYWORDS = [
    'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'UNION', 'WHERE', 'FROM', 'JOIN', 'AND', 'OR', 'NOT', 'NULL',
    'ORDER', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'AS', 'ON',
    'EXEC', 'EXECUTE', 'DECLARE', 'TABLE', 'DATABASE', 'COLUMN',
    'BETWEEN', 'LIKE', 'IN', 'EXISTS', 'CASE', 'WHEN', 'THEN',
    'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX'
]

XSS_KEYWORDS = [
    'script', 'iframe', 'object', 'embed', 'applet', 'meta', 'link',
    'style', 'img', 'svg', 'video', 'audio', 'canvas', 'input',
    'button', 'form', 'body', 'html', 'onerror', 'onload', 'onclick',
    'onmouseover', 'onfocus', 'onblur', 'alert', 'prompt', 'confirm',
    'eval', 'expression', 'javascript', 'vbscript', 'document',
    'window', 'location', 'cookie', 'localstorage', 'sessionstorage'
]

class ContentMatchingPreprocessor:
    """Implements content matching preprocessing"""

    def __init__(self):
        self.sql_keywords = set([kw.lower() for kw in SQL_KEYWORDS])
        self.xss_keywords = set([kw.lower() for kw in XSS_KEYWORDS])
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            import nltk
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

    def digital_generalization(self, text):
        return re.sub(r'\d+', ' NUM ', text)

    def url_replacement(self, text):
        url_pattern = r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        return re.sub(url_pattern, ' URL ', text)

    def preserve_keywords(self, text):
        for keyword in self.sql_keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            text = pattern.sub(f' SQL_{keyword.upper()} ', text)
        for keyword in self.xss_keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            text = pattern.sub(f' XSS_{keyword.upper()} ', text)
        return text

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def special_character_mapping(self, text):
        char_mappings = [
            ('--', ' COMMENT '),
            ('/*', ' BLOCKCOMMENT_START '),
            ('*/', ' BLOCKCOMMENT_END '),
            ("'", ' SQUOTE '),
            ('"', ' DQUOTE '),
            (';', ' SEMICOLON '),
            ('=', ' EQUALS '),
            ('<', ' LT '),
            ('>', ' GT '),
            ('(', ' LPAREN '),
            (')', ' RPAREN '),
            ('{', ' LBRACE '),
            ('}', ' RBRACE '),
            ('[', ' LBRACKET '),
            (']', ' RBRACKET ')
        ]
        for char, token in char_mappings:
            text = text.replace(char, token)
        return text

    def preprocess(self, text, apply_stemming=False, remove_stopwords=False):
        if not isinstance(text, str):
            text = str(text)
        text = self.digital_generalization(text)
        text = self.url_replacement(text)
        text = self.preserve_keywords(text)   # Before lowercase!
        text = self.normalize_text(text)
        text = self.special_character_mapping(text)
        text = re.sub(r'\s+', ' ', text).strip()

        if apply_stemming or remove_stopwords:
            tokens = word_tokenize(text)
            if remove_stopwords:
                tokens = [t for t in tokens if t not in self.stop_words]
            if apply_stemming:
                tokens = [self.stemmer.stem(t) for t in tokens]
            text = ' '.join(tokens)
        return text


class DataLoader:
    """
    Load datasets and produce source-aware splits.

    Key principle: datasets are split BEFORE merging so that no sample from a
    held-out source can appear in the training set.
    """

    def __init__(self, data_dir='/content', config=None):
        self.data_dir = data_dir
        self.preprocessor = ContentMatchingPreprocessor()
        self.config = config if config is not None else GLOBAL_CONFIG


    def _load_csv(self, filename, text_col=None, label_col='Label'):
        """Load one CSV and return a clean (text, label, dataset_source) DataFrame."""
        filepath = f"{self.data_dir}/{filename}"
        print(f"  Loading {filename} ...", end=' ')

        try:
            df = pd.read_csv(filepath, engine='python')

            # Auto-detect text column
            if text_col is None:
                for col in ['Sentence', 'Query', 'text', 'Text', 'payload', 'Payload']:
                    if col in df.columns:
                        text_col = col
                        break
            if text_col is None:
                text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            df = df[['text', 'label']].dropna(subset=['label'])
            df['text'] = df['text'].fillna('')
            df['label'] = df['label'].astype(int)
            print(f"{len(df)} rows  |  label dist: {df['label'].value_counts().to_dict()}")
            return df

        except Exception as e:
            print(f"FAILED — {e}")
            return None

    def _preprocess_df(self, df):
        """Apply content-matching preprocessing to a DataFrame in-place."""
        df = df.copy()
        df['text'] = df['text'].apply(self.preprocessor.preprocess)
        return df

    def _stratified_split(self, df, test_size, seed):
        """Split a DataFrame into two stratified parts."""
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df['label']
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _remove_duplicates(self, df, label='dataset'):
        before = len(df)
        df = df.drop_duplicates(subset=['text', 'label'], keep='first')
        removed = before - len(df)
        if removed:
            print(f"  [{label}] Removed {removed} exact duplicates.")
        return df.reset_index(drop=True)


    def load_and_split(self):
        """
        Load all datasets and produce source-aware splits.

        Returns
        -------
        splits : dict with keys
            X_train, y_train       — unified training set
            X_val,   y_val         — unified validation set
            X_test,  y_test        — unified internal test set
            X_cross, y_cross       — cross-dataset held-out test (SQL only)
        source_meta : dict
            Per-split source breakdown for reporting in the paper.
        """
        seed = self.config['random_seed']

        print("\n" + "="*70)
        print("LOADING DATASETS")
        print("="*70)


        print("\n[SQL — training pool]")
        sql_train_raw = self._load_csv('Train.csv', text_col='Query')
        sql_val_raw   = self._load_csv('Validation.csv', text_col='Query')

        print("\n[SQL — internal test]")
        sql_test_raw  = self._load_csv('Test.csv', text_col='Query')

        print("\n[SQL — cross-dataset held-out]")
        sql_cross_raw = self._load_csv('Modified_SQL_Dataset.csv', text_col='Query')


        print("\n[XSS — splitting before merge]")
        xss_raw = self._load_csv('XSS_dataset.csv', text_col='Sentence')


        if sql_train_raw is None or xss_raw is None:
            raise ValueError("Cannot proceed without SQL Train.csv and XSS_dataset.csv.")

        xss_trainval, xss_test = self._stratified_split(xss_raw, test_size=0.15, seed=seed)
        xss_train, xss_val     = self._stratified_split(xss_trainval, test_size=0.15/0.85, seed=seed)

        print(f"  XSS train: {len(xss_train)}  |  val: {len(xss_val)}  |  test: {len(xss_test)}")

        sql_pool_parts = [p for p in [sql_train_raw, sql_val_raw] if p is not None]
        sql_pool = pd.concat(sql_pool_parts, ignore_index=True)
        sql_pool = self._remove_duplicates(sql_pool, 'SQL pool')

        sql_train, sql_val = self._stratified_split(sql_pool, test_size=0.15, seed=seed)
        print(f"\n  SQL train: {len(sql_train)}  |  val: {len(sql_val)}")

        if sql_test_raw is not None:
            sql_test = self._remove_duplicates(sql_test_raw, 'SQL test')
            print(f"  SQL internal test: {len(sql_test)}")
        else:

            print("  WARNING: Test.csv not found — carving test from pool.")
            sql_train, sql_test = self._stratified_split(sql_train, test_size=0.15/0.85, seed=seed)

        if sql_cross_raw is not None:
            sql_cross = self._remove_duplicates(sql_cross_raw, 'SQL cross')
            print(f"  SQL cross-dataset held-out: {len(sql_cross)}")
        else:
            sql_cross = None
            print("  WARNING: Modified_SQL_Dataset.csv not found — cross-dataset test unavailable.")

        for df, tag in [
            (sql_train, 'sql_train'), (sql_val, 'sql_val'),
            (sql_test,  'sql_test'),
            (xss_train, 'xss_train'), (xss_val,  'xss_val'),
            (xss_test,  'xss_test'),
        ]:
            df['source'] = tag

        if sql_cross is not None:
            sql_cross['source'] = 'sql_cross'


        train_df = pd.concat([sql_train, xss_train], ignore_index=True)
        val_df   = pd.concat([sql_val,   xss_val],   ignore_index=True)
        test_df  = pd.concat([sql_test,  xss_test],  ignore_index=True)


        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        val_df   = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        test_df  = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)


        print("\n" + "="*70)
        print("PREPROCESSING SPLITS")
        print("="*70)
        print("Preprocessing train set ...")
        train_df = self._preprocess_df(train_df)
        print("Preprocessing val set ...")
        val_df   = self._preprocess_df(val_df)
        print("Preprocessing internal test set ...")
        test_df  = self._preprocess_df(test_df)

        cross_df = None
        if sql_cross is not None:
            print("Preprocessing cross-dataset test set ...")
            cross_df = self._preprocess_df(sql_cross)

        print("\n" + "="*70)
        print("SPLIT SUMMARY")
        print("="*70)
        print(f"  Train   : {len(train_df):>7}  "
              f"(SQL: {(train_df.source=='sql_train').sum()}, "
              f"XSS: {(train_df.source=='xss_train').sum()})")
        print(f"  Val     : {len(val_df):>7}  "
              f"(SQL: {(val_df.source=='sql_val').sum()}, "
              f"XSS: {(val_df.source=='xss_val').sum()})")
        print(f"  Test    : {len(test_df):>7}  "
              f"(SQL: {(test_df.source=='sql_test').sum()}, "
              f"XSS: {(test_df.source=='xss_test').sum()})")
        if cross_df is not None:
            print(f"  Cross   : {len(cross_df):>7}  (SQL only — Modified_SQL_Dataset)")
        print()

        splits = {
            'X_train': train_df['text'].values,
            'y_train': train_df['label'].values,
            'X_val':   val_df['text'].values,
            'y_val':   val_df['label'].values,
            'X_test':  test_df['text'].values,
            'y_test':  test_df['label'].values,
        }

        if cross_df is not None:
            splits['X_cross'] = cross_df['text'].values
            splits['y_cross'] = cross_df['label'].values

        source_meta = {
            'train_sources': train_df['source'].value_counts().to_dict(),
            'val_sources':   val_df['source'].value_counts().to_dict(),
            'test_sources':  test_df['source'].value_counts().to_dict(),
        }
        if cross_df is not None:
            source_meta['cross_sources'] = cross_df['source'].value_counts().to_dict()

        return splits, source_meta

    def load_all_datasets(self):
        """Deprecated shim — use load_and_split() instead."""
        raise NotImplementedError(
            "load_all_datasets() has been replaced by load_and_split() "
            "to implement proper source-aware splits. "
            "Please update master_runner.py to call load_and_split()."
        )

    def preprocess_data(self, df):
        """Deprecated shim."""
        return self._preprocess_df(df)

    def create_splits(self, df, config=None):
        """Deprecated shim."""
        raise NotImplementedError(
            "create_splits() has been replaced by load_and_split(). "
            "Merging before splitting causes distribution leakage."
        )


if __name__ == "__main__":
    from config import GLOBAL_CONFIG, set_seed
    from pathlib import Path

    set_seed(GLOBAL_CONFIG['random_seed'])

    loader = DataLoader(data_dir=GLOBAL_CONFIG['data_dir'], config=GLOBAL_CONFIG)
    splits, source_meta = loader.load_and_split()

    print("Source breakdown:")
    for k, v in source_meta.items():
        print(f"  {k}: {v}")

    save_dir = f"{GLOBAL_CONFIG['base_dir']}/data"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for key, val in splits.items():
        np.save(f"{save_dir}/{key}.npy", val)
        print(f"  Saved {key}.npy  shape={val.shape}")

    print("\nData preprocessing complete!")
