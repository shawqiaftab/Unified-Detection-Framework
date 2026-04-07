
from config import GLOBAL_CONFIG
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle
import warnings
import gc

warnings.filterwarnings('ignore')


class UniEmbedExtractor:
    """Extract UniEmbed features with dynamic dimensions"""

    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        self.word2vec_model = None
        self.fasttext_model = None
        self.use_model = None

    def tokenize_texts(self, texts):
        """Tokenize texts for training word embedding models"""
        return [text.split() for text in texts]

    def train_word2vec(self, texts):
        """Train Word2Vec model"""
        print("Training Word2Vec model...")
        tokenized = self.tokenize_texts(texts)

        self.word2vec_model = Word2Vec(
            sentences=tokenized,
            vector_size=self.config['word2vec_dim'],
            window=5,
            min_count=1,
            workers=4,
            sg=0,
            seed=self.config['random_seed']
        )
        print(f"  Word2Vec trained: {len(self.word2vec_model.wv)} tokens")
        return self.word2vec_model

    def train_fasttext(self, texts):
        """Train FastText model"""
        print("Training FastText model...")
        tokenized = self.tokenize_texts(texts)

        self.fasttext_model = FastText(
            sentences=tokenized,
            vector_size=self.config['fasttext_dim'],
            window=5,
            min_count=1,
            workers=4,
            min_n=3,
            max_n=6,
            seed=self.config['random_seed']
        )
        print(f"  FastText trained: {len(self.fasttext_model.wv)} tokens")
        return self.fasttext_model

    def load_use_model(self):
        """Load Universal Sentence Encoder"""
        if self.config.get('skip_use', False):
            print("Skipping USE (disabled in config)")
            return None

        print("Loading Universal Sentence Encoder...")
        try:
            self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            print("  USE loaded successfully")
        except Exception as e:
            print(f"  Failed to load USE: {e}")
            self.use_model = None

        return self.use_model

    def get_word2vec_embedding(self, text):
        """Get Word2Vec embedding for a text"""
        tokens = text.split()
        vectors = []
        for token in tokens:
            if token in self.word2vec_model.wv:
                vectors.append(self.word2vec_model.wv[token])

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.config['word2vec_dim'])

    def get_fasttext_embedding(self, text):
        """Get FastText embedding for a text"""
        tokens = text.split()
        vectors = []
        for token in tokens:
            if token in self.fasttext_model.wv:
                vectors.append(self.fasttext_model.wv[token])

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.config['fasttext_dim'])

    def get_use_embedding(self, texts):
        """Get USE embeddings for texts"""
        if self.use_model is None:
            return np.zeros((len(texts), self.config['use_dim']))

        embeddings = self.use_model(texts)
        return embeddings.numpy()

    def extract_uniembed_features(self, texts, batch_size=1000):
        """Extract complete UniEmbed features"""
        w2v_dim = self.config['word2vec_dim']
        ft_dim = self.config['fasttext_dim']
        use_dim = self.config['use_dim']

        if self.config.get('skip_use', False) or self.use_model is None:
            total_dim = w2v_dim + ft_dim
            use_enabled = False
        else:
            total_dim = w2v_dim + ft_dim + use_dim
            use_enabled = True

        print(f"\nEXTRACTING UNIEMBED FEATURES ({total_dim}D)")

        n_samples = len(texts)
        features = np.zeros((n_samples, total_dim))

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_texts = texts[i:batch_end]

            for j, text in enumerate(batch_texts):
                idx = i + j
                features[idx, :w2v_dim] = self.get_word2vec_embedding(text)
                features[idx, w2v_dim:w2v_dim+ft_dim] = self.get_fasttext_embedding(text)

            if use_enabled:
                features[i:batch_end, w2v_dim+ft_dim:] = self.get_use_embedding(batch_texts)

            if (batch_end % 10000 == 0) or (batch_end == n_samples):
                print(f"  Processed {batch_end}/{n_samples} samples")

        return features

    def fit_transform(self, train_texts, val_texts, test_texts):
        """Train models on train set and transform all sets"""
        self.train_word2vec(train_texts)
        self.train_fasttext(train_texts)
        self.load_use_model()

        print("\nExtracting features for Training set...")
        X_train = self.extract_uniembed_features(train_texts)

        print("\nExtracting features for Validation set...")
        X_val = self.extract_uniembed_features(val_texts)

        print("\nExtracting features for Test set...")
        X_test = self.extract_uniembed_features(test_texts)

        return X_train, X_val, X_test

    def save(self, save_dir):
        """Save trained models"""
        print(f"\nSaving UniEmbed models to {save_dir}...")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.word2vec_model.save(f"{save_dir}/word2vec.model")
        self.fasttext_model.save(f"{save_dir}/fasttext.model")
        print("  Models saved successfully!")


class TFIDFExtractor:
    """Extract TF-IDF features"""

    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config

        self.vectorizer = TfidfVectorizer(
            max_features=config['tfidf_max_features'],
            ngram_range=config['tfidf_ngram_range'],
            analyzer='word',
            lowercase=True,
            stop_words='english' 
        )

    def fit_transform(self, train_texts, val_texts, test_texts):
        """Fit on train and transform all sets"""
        print(f"\nEXTRACTING TF-IDF FEATURES ({self.config['tfidf_max_features']}D)")

        print("Fitting TF-IDF vectorizer...")
        X_train = self.vectorizer.fit_transform(train_texts)

        print("Transforming validation set...")
        X_val = self.vectorizer.transform(val_texts)

        print("Transforming test set...")
        X_test = self.vectorizer.transform(test_texts)

        print(f"  Train shape: {X_train.shape}")

        return X_train.toarray(), X_val.toarray(), X_test.toarray()

    def save(self, save_dir):
        """Save vectorizer"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("  TF-IDF vectorizer saved!")


def extract_all_features(splits, config=None):
    """Extract all feature representations"""
    if config is None:
        config = GLOBAL_CONFIG

    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']

    feature_dir = f"{config['base_dir']}/features"
    Path(feature_dir).mkdir(parents=True, exist_ok=True)
    features = {}

    print("\n" + "="*70)
    print("STAGE 1: UNIEMBED FEATURES")
    print("="*70)

    uni_extractor = UniEmbedExtractor(config)
    X_train_uni, X_val_uni, X_test_uni = uni_extractor.fit_transform(X_train, X_val, X_test)
    uni_extractor.save(feature_dir)

    np.save(f"{feature_dir}/X_train_uniembed.npy", X_train_uni)
    np.save(f"{feature_dir}/X_val_uniembed.npy", X_val_uni)
    np.save(f"{feature_dir}/X_test_uniembed.npy", X_test_uni)

    del X_train_uni, X_val_uni, X_test_uni
    gc.collect()

    print("\n" + "="*70)
    print("STAGE 2: TF-IDF FEATURES")
    print("="*70)

    tfidf_extractor = TFIDFExtractor(config)
    X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf_extractor.fit_transform(X_train, X_val, X_test)
    tfidf_extractor.save(feature_dir)

    np.save(f"{feature_dir}/X_train_tfidf.npy", X_train_tfidf)
    np.save(f"{feature_dir}/X_val_tfidf.npy", X_val_tfidf)
    np.save(f"{feature_dir}/X_test_tfidf.npy", X_test_tfidf)

    del X_train_tfidf, X_val_tfidf, X_test_tfidf
    gc.collect()

    print("\nAll features extracted and saved!")
    return features

def extract_cross_features(config=None):
    """
    Extract TF-IDF and UniEmbed features for the cross-dataset held-out split
    (X_cross / y_cross) using the ALREADY-FITTED vectorizers from training.
    Must be called AFTER extract_all_features() so the fitted models exist.
    """
    if config is None:
        config = GLOBAL_CONFIG

    data_dir    = f"{config['base_dir']}/data"
    feature_dir = f"{config['base_dir']}/features"

    cross_path = f"{data_dir}/X_cross.npy"
    if not os.path.exists(cross_path):
        print("X_cross.npy not found — skipping cross-dataset feature extraction.")
        return

    print("\n" + "="*70)
    print("EXTRACTING FEATURES FOR CROSS-DATASET HELD-OUT SPLIT")
    print("="*70)

    X_cross = np.load(cross_path, allow_pickle=True)
    print(f"  Cross-dataset samples: {len(X_cross)}")


    tfidf_vectorizer_path = f"{feature_dir}/tfidf_vectorizer.pkl"
    if os.path.exists(tfidf_vectorizer_path):
        with open(tfidf_vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        X_cross_tfidf = tfidf_vectorizer.transform(X_cross).toarray()
        np.save(f"{feature_dir}/X_cross_tfidf.npy", X_cross_tfidf)
        print(f"  Saved X_cross_tfidf.npy  shape={X_cross_tfidf.shape}")
        del X_cross_tfidf
        gc.collect()
    else:
        print("  WARNING: tfidf_vectorizer.pkl not found — skipping TF-IDF for cross split.")

    w2v_path = f"{feature_dir}/word2vec.model"
    ft_path  = f"{feature_dir}/fasttext.model"
    if os.path.exists(w2v_path) and os.path.exists(ft_path):
        from gensim.models import Word2Vec, FastText as FastTextModel

        uni = UniEmbedExtractor(config)
        uni.word2vec_model = Word2Vec.load(w2v_path)
        uni.fasttext_model = FastTextModel.load(ft_path)

        if not config.get('skip_use', False):
            uni.load_use_model()

        X_cross_uni = uni.extract_uniembed_features(X_cross)
        np.save(f"{feature_dir}/X_cross_uniembed.npy", X_cross_uni)
        print(f"  Saved X_cross_uniembed.npy  shape={X_cross_uni.shape}")
        del X_cross_uni
        gc.collect()
    else:
        print("  WARNING: Word2Vec/FastText models not found — skipping UniEmbed for cross split.")

    print("Cross-dataset feature extraction complete!")

if __name__ == "__main__":
    from config import GLOBAL_CONFIG, set_seed

    set_seed(GLOBAL_CONFIG['random_seed'])
    data_dir = f"{GLOBAL_CONFIG['base_dir']}/data"

    splits = {
        'X_train': np.load(f"{data_dir}/X_train.npy", allow_pickle=True),
        'X_val': np.load(f"{data_dir}/X_val.npy", allow_pickle=True),
        'X_test': np.load(f"{data_dir}/X_test.npy", allow_pickle=True),
        'y_train': np.load(f"{data_dir}/y_train.npy"), # Just for completeness
    }

    extract_all_features(splits, GLOBAL_CONFIG)
