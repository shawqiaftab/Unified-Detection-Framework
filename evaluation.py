

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef
)
from scipy.stats import chi2
import time
import json
from pathlib import Path
import joblib
import tensorflow as tf
import torch
import pickle
import gc

from config import (
    GLOBAL_CONFIG, set_seed,
    CLASSICAL_ML_MODELS, DEEP_LEARNING_MODELS,
    TRANSFORMER_MODELS, GNN_MODELS,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification ## OK
from torch_geometric.loader import DataLoader       # OK KAJ KORE
from models_gnn import GCN, GAT, GraphConstructor # OK



def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95, seed=42):
    """
    Bootstrap confidence interval for a classification metric.

    Resamples (y_true, y_pred) with replacement n_bootstrap times, computes
    the metric each time, and returns the (ci/2)th and (100-ci/2)th
    percentiles as lower and upper bounds.

    Parameters
    ----------
    y_true, y_pred : array-like
    metric_fn      : callable(y_true, y_pred) -> float
    n_bootstrap    : int   (1000 is standard for publications)
    ci             : float (95 = 95% CI)
    seed           : int

    Returns
    -------
    lower, upper : float, float
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (100 - ci) / 2
    return float(np.percentile(scores, alpha)), float(np.percentile(scores, 100 - alpha))


def mcnemar_test(y_pred_a, y_pred_b, y_true):
    """
    McNemar's test with Edwards' continuity correction.

    Compares two classifiers by examining the samples where one model is
    correct and the other is not.

    H0: both classifiers err at the same rate.
    p < 0.05 means the difference is statistically significant.

    Returns
    -------
    statistic : float
    p_value   : float
    b01       : int — A wrong, B correct
    b10       : int — A correct, B wrong
    """
    ca  = np.asarray(y_pred_a) == np.asarray(y_true)
    cb  = np.asarray(y_pred_b) == np.asarray(y_true)
    b01 = int(np.sum(~ca & cb))
    b10 = int(np.sum(ca & ~cb))
    denom = b01 + b10
    if denom == 0:
        return 0.0, 1.0, b01, b10     # identical predictions
    stat = (abs(b01 - b10) - 1) ** 2 / denom
    p    = float(1 - chi2.cdf(stat, df=1))
    return float(stat), p, b01, b10


def run_all_mcnemar_tests(predictions_dict, y_true, reference_model='XGBoost'):
    """
    Run McNemar's tests: every model vs the reference, plus every unique pair.
    Returns a list of dicts ready for pd.DataFrame().
    """
    results = []
    names   = list(predictions_dict.keys())

    if reference_model in predictions_dict:
        ref = predictions_dict[reference_model]
        for name, pred in predictions_dict.items():
            if name == reference_model:
                continue
            stat, p, b01, b10 = mcnemar_test(pred, ref, y_true)
            results.append({
                'comparison':          f'{name}_vs_{reference_model}',
                'model_a':             name,
                'model_b':             reference_model,
                'mcnemar_statistic':   round(stat, 4),
                'p_value':             round(p, 6),
                'significant_p05':     p < 0.05,
                'b_a_wrong_b_correct': b01,
                'b_a_correct_b_wrong': b10,
                'interpretation': (
                    f'{reference_model} is significantly better'
                    if (p < 0.05 and b01 > b10)
                    else f'{name} is significantly better'
                    if (p < 0.05 and b10 > b01)
                    else 'No significant difference'
                ),
            })

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            stat, p, b01, b10 = mcnemar_test(
                predictions_dict[a], predictions_dict[b], y_true)
            results.append({
                'comparison':          f'{a}_vs_{b}',
                'model_a':             a,
                'model_b':             b,
                'mcnemar_statistic':   round(stat, 4),
                'p_value':             round(p, 6),
                'significant_p05':     p < 0.05,
                'b_a_wrong_b_correct': b01,
                'b_a_correct_b_wrong': b10,
                'interpretation': (
                    f'{b} is significantly better'
                    if (p < 0.05 and b01 > b10)
                    else f'{a} is significantly better'
                    if (p < 0.05 and b10 > b01)
                    else 'No significant difference'
                ),
            })
    return results


class ComprehensiveEvaluator:
    """Evaluate all 17 models with comprehensive metrics and statistical tests."""

    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config

    def compute_error_metrics(self, y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        sq   = (y_true - y_pred) ** 2
        ab   = np.abs(y_true - y_pred)
        sse  = float(np.sum(sq))
        mse  = float(np.mean(sq))
        u, c = np.unique(y_true, return_counts=True)
        cw   = {int(cls): len(y_true) / cnt for cls, cnt in zip(u, c)}
        w    = np.array([cw.get(int(yt), 1.0) for yt in y_true])
        wsq  = w * sq
        return {
            'sse': sse, 'mse': mse, 'rmse': float(np.sqrt(mse)),
            'mae': float(np.mean(ab)),
            'wsse': float(np.sum(wsq)), 'wmse': float(np.mean(wsq)),
        }


    def compute_all_metrics(self, y_true, y_pred, y_pred_proba=None,
                            compute_ci=True, n_bootstrap=1000):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        m = {
            'accuracy':  float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score':  float(f1_score(y_true, y_pred, zero_division=0)),
            'mcc':       float(matthews_corrcoef(y_true, y_pred)),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        }

        if y_pred_proba is not None:
            try:
                ys = (y_pred_proba[:, 1]
                      if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2
                      else y_pred_proba)
                m['roc_auc'] = float(roc_auc_score(y_true, ys))
                m['pr_auc']  = float(average_precision_score(y_true, ys))
            except Exception:
                m['roc_auc'] = None
                m['pr_auc']  = None
        else:
            m['roc_auc'] = None
            m['pr_auc']  = None

        m.update(self.compute_error_metrics(y_true, y_pred))

        if compute_ci:
            try:
                a_lo, a_hi = bootstrap_ci(y_true, y_pred, accuracy_score,
                                          n_bootstrap=n_bootstrap)
                f_lo, f_hi = bootstrap_ci(
                    y_true, y_pred,
                    lambda yt, yp: f1_score(yt, yp, zero_division=0),
                    n_bootstrap=n_bootstrap)
                m.update({'acc_ci_low': a_lo, 'acc_ci_high': a_hi,
                          'f1_ci_low':  f_lo, 'f1_ci_high':  f_hi})
            except Exception as e:
                print(f"    Warning: Bootstrap CI failed — {e}")
                for k in ('acc_ci_low', 'acc_ci_high', 'f1_ci_low', 'f1_ci_high'):
                    m[k] = None
        return m


    def evaluate_classical_ml(self, model_name, X_test, y_test, compute_ci=True):
        print(f"\n  Evaluating {model_name}...")
        model  = joblib.load(
            f"{self.config['models_dir']}/classical_ml/{model_name}.pkl")
        t0     = time.time()
        y_pred = model.predict(X_test)
        inf_t  = time.time() - t0
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        m = self.compute_all_metrics(y_test, y_pred, y_prob, compute_ci=compute_ci)
        m['total_inference_time']      = float(inf_t)
        m['inference_time_per_sample'] = float(inf_t / len(X_test) * 1000)
        m['throughput']                = float(len(X_test) / inf_t)
        ci = (f" [95%CI: {m['acc_ci_low']:.4f}–{m['acc_ci_high']:.4f}]"
              if compute_ci and m.get('acc_ci_low') is not None else "")
        print(f"    Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | "
              f"MSE: {m['mse']:.6f}{ci}")
        return m, y_pred, y_prob

    def evaluate_deep_learning(self, model_name, X_test, y_test, compute_ci=True):
        print(f"\n  Evaluating {model_name}...")
        model = tf.keras.models.load_model(
            f"{self.config['models_dir']}/deep_learning/{model_name}.h5")
        if model_name == 'CNN':
            Xp = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        elif model_name in ['RNN', 'LSTM', 'BiLSTM', 'CNN_LSTM', 'CNN_RNN']:
            nf = X_test.shape[1]
            ts = min(nf, 50)
            fp = nf // ts
            Xp = X_test[:, :ts * fp].reshape(X_test.shape[0], ts, fp)
        else:
            Xp = X_test
        t0     = time.time()
        y_prob = model.predict(Xp, verbose=0)
        inf_t  = time.time() - t0
        y_pred = (y_prob > 0.5).astype(int).flatten()
        m = self.compute_all_metrics(y_test, y_pred, y_prob, compute_ci=compute_ci)
        m['total_inference_time']      = float(inf_t)
        m['inference_time_per_sample'] = float(inf_t / len(X_test) * 1000)
        m['throughput']                = float(len(X_test) / inf_t)
        m['model_parameters']          = int(model.count_params())
        ci = (f" [95%CI: {m['acc_ci_low']:.4f}–{m['acc_ci_high']:.4f}]"
              if compute_ci and m.get('acc_ci_low') is not None else "")
        print(f"    Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | "
              f"MSE: {m['mse']:.6f}{ci}")
        return m, y_pred, y_prob

    def evaluate_transformer(self, model_name, X_text, y_test, compute_ci=True):
        print(f"\n  Evaluating {model_name}...")
        model_dir = f"{self.config['models_dir']}/transformers/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device).eval()
        y_pred_l, y_prob_l = [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(X_text), 32):
                batch = list(X_text[i:i + 32])
                enc   = tokenizer(batch, padding=True, truncation=True,
                                  max_length=self.config['max_seq_length'],
                                  return_tensors='pt')
                enc   = {k: v.to(device) for k, v in enc.items()}
                out   = model(**enc)
                y_prob_l.extend(torch.softmax(out.logits, dim=1).cpu().numpy())
                y_pred_l.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        inf_t  = time.time() - t0
        y_pred = np.array(y_pred_l)
        y_prob = np.array(y_prob_l)
        m = self.compute_all_metrics(y_test, y_pred, y_prob, compute_ci=compute_ci)
        m['total_inference_time']      = float(inf_t)
        m['inference_time_per_sample'] = float(inf_t / len(X_text) * 1000)
        m['throughput']                = float(len(X_text) / inf_t)
        m['model_parameters']          = int(sum(p.numel() for p in model.parameters()))
        ci = (f" [95%CI: {m['acc_ci_low']:.4f}–{m['acc_ci_high']:.4f}]"
              if compute_ci and m.get('acc_ci_low') is not None else "")
        print(f"    Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | "
              f"MSE: {m['mse']:.6f}{ci}")
        return m, y_pred, y_prob

    def evaluate_gnn(self, model_name, test_graphs, compute_ci=True):
        print(f"\n  Evaluating {model_name}...")
        input_dim = test_graphs[0].x.shape[1]
        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_name == 'GCN':
            model = GCN(input_dim=input_dim,
                        hidden_dims=self.config['gnn_hidden_dims'],
                        num_classes=2, dropout=self.config['gnn_dropout'])
        elif model_name == 'GAT':
            model = GAT(input_dim=input_dim,
                        hidden_dims=self.config['gnn_hidden_dims'],
                        num_classes=2, heads=self.config['gat_heads'],
                        dropout=self.config['gnn_dropout'])
        else:
            raise ValueError(f"Unknown GNN model: {model_name}")
        model.load_state_dict(torch.load(
            f"{self.config['models_dir']}/gnn/{model_name}.pt",
            map_location=device))
        model.to(device).eval()
        loader = DataLoader(test_graphs,
                            batch_size=self.config['gnn_batch_size'],
                            shuffle=False)
        preds, labels = [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out   = model(batch.x, batch.edge_index, batch.batch)
                preds.extend(out.argmax(dim=1).cpu().numpy())
                labels.extend(batch.y.cpu().numpy())
        inf_t  = time.time() - t0
        y_pred = np.array(preds)
        y_true = np.array(labels)
        # GNNs output hard labels — no probability estimates available
        m = self.compute_all_metrics(y_true, y_pred, y_pred_proba=None,
                                     compute_ci=compute_ci)
        m['roc_auc'] = None
        m['pr_auc']  = None
        m['total_inference_time']      = float(inf_t)
        m['inference_time_per_sample'] = float(inf_t / len(y_true) * 1000)
        m['throughput']                = float(len(y_true) / inf_t)
        m['model_parameters']          = int(sum(p.numel() for p in model.parameters()))
        ci = (f" [95%CI: {m['acc_ci_low']:.4f}–{m['acc_ci_high']:.4f}]"
              if compute_ci and m.get('acc_ci_low') is not None else "")
        print(f"    Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | "
              f"MSE: {m['mse']:.6f}{ci}")
        return m, y_pred, None


    def _build_gnn_graphs_for_cross(self, X_cross_text, y_cross):
        """
        Build GNN graphs for ALL cross-dataset samples.

        The 6,000-sample limit applied only to GNN *training* — a Kaggle
        runtime constraint during backpropagation. Inference is forward-pass
        only and has no such constraint. This method processes all 30,907
        cross-dataset samples, exactly as the internal test set evaluation
        already processes all 34,741 test graphs.

        Loads the Word2Vec model saved during feature extraction (FastText is
        not needed — only Word2Vec embeddings are used for node features and
        semantic edges in GraphConstructor).
        """
        from gensim.models import Word2Vec

        w2v_path = f"{self.config['base_dir']}/features/word2vec.model"
        if not Path(w2v_path).exists():
            raise FileNotFoundError(
                f"Word2Vec model not found at {w2v_path}. "
                "Run feature extraction before cross-dataset GNN evaluation.")

        print(f"  Loading Word2Vec from {w2v_path} ...")
        w2v = Word2Vec.load(w2v_path)

        print(f"  Building graphs for all {len(X_cross_text)} cross-dataset samples ...")
        print(f"  (Forward-pass only — no training cap applies here)")

        constructor = GraphConstructor(word2vec_model=w2v)
        graphs = constructor.texts_to_graphs(
            np.array(X_cross_text), np.array(y_cross))

        del w2v
        gc.collect()
        return graphs


    def _run_and_save_statistical_tests(self, all_preds, y_true,
                                        filename_prefix='statistical_tests'):
        print("\n" + "=" * 70)
        print("STATISTICAL ANALYSIS — McNEMAR'S TESTS")
        print("=" * 70)
        ref = 'XGBoost' if 'XGBoost' in all_preds else list(all_preds.keys())[0]
        print(f"  Reference: {ref}  |  alpha=0.05  |  Edwards continuity correction")

        results  = run_all_mcnemar_tests(all_preds, y_true, reference_model=ref)
        stats_df = pd.DataFrame(results)

        vs_ref = stats_df[stats_df['model_b'] == ref]
        if len(vs_ref) > 0:
            print(f"\n  Results vs {ref}:")
            print(f"  {'Model':<26} {'p-value':>10}  {'Sig.':>12}  Interpretation")
            print(f"  {'-'*26} {'-'*10}  {'-'*12}  {'-'*35}")
            for _, row in vs_ref.iterrows():
                sig = 'Yes (p<0.05)' if row['significant_p05'] else 'No'
                print(f"  {row['model_a']:<26} {row['p_value']:>10.6f}  "
                      f"{sig:>12}  {row['interpretation']}")

        n_sig = stats_df['significant_p05'].sum()
        print(f"\n  Total pairs: {len(stats_df)}  |  "
              f"Significantly different (p<0.05): {n_sig}")

        sdir = f"{self.config['results_dir']}/metrics"
        Path(sdir).mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(f"{sdir}/{filename_prefix}.csv", index=False)
        with open(f"{sdir}/{filename_prefix}.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {sdir}/{filename_prefix}.csv")


    def evaluate_all_models(self, test_data, run_statistical_tests=True):
        """
        Evaluate all 17 models on the internal held-out test set.

        test_data must contain:
            X_test_tfidf, X_test_uniembed, X_test_text, y_test,
            (optional) test_graphs
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE MODEL EVALUATION (WITH STATISTICAL ANALYSIS)")
        print("=" * 70)

        all_results, all_preds = [], {}
        y_test = test_data['y_test']

        print("\n" + "-" * 70)
        print("EVALUATING CLASSICAL ML MODELS")
        print("-" * 70)
        for mn in CLASSICAL_ML_MODELS:
            try:
                m, yp, yb = self.evaluate_classical_ml(
                    mn, test_data['X_test_tfidf'], y_test)
                m.update({'model_name': mn, 'model_type': 'Classical ML'})
                all_results.append(m)
                all_preds[mn] = yp
                self.save_predictions(mn, yp, yb, y_test)
            except Exception as e:
                print(f"    Error evaluating {mn}: {e}")

        print("\n" + "-" * 70)
        print("EVALUATING DEEP LEARNING MODELS")
        print("-" * 70)
        for mn in DEEP_LEARNING_MODELS:
            try:
                m, yp, yb = self.evaluate_deep_learning(
                    mn, test_data['X_test_uniembed'], y_test)
                m.update({'model_name': mn, 'model_type': 'Deep Learning'})
                all_results.append(m)
                all_preds[mn] = yp
                self.save_predictions(mn, yp, yb, y_test)
            except Exception as e:
                print(f"    Error evaluating {mn}: {e}")

        print("\n" + "-" * 70)
        print("EVALUATING TRANSFORMER MODELS")
        print("-" * 70)
        for mn in TRANSFORMER_MODELS:
            try:
                m, yp, yb = self.evaluate_transformer(
                    mn, test_data['X_test_text'], y_test)
                m.update({'model_name': mn, 'model_type': 'Transformer'})
                all_results.append(m)
                all_preds[mn] = yp
                self.save_predictions(mn, yp, yb, y_test)
            except Exception as e:
                print(f"    Error evaluating {mn}: {e}")

        print("\n" + "-" * 70)
        print("EVALUATING GNN MODELS")
        print("-" * 70)
        if 'test_graphs' in test_data:
            tg = test_data['test_graphs']
            print(f"  Loaded {len(tg)} test graphs")
            for mn in ['GCN', 'GAT']:
                if not Path(f"{self.config['models_dir']}/gnn/{mn}.pt").exists():
                    print(f"  {mn} checkpoint not found — skipping.")
                    continue
                try:
                    m, yp, _ = self.evaluate_gnn(mn, tg)
                    m.update({'model_name': mn, 'model_type': 'GNN'})
                    all_results.append(m)
                    all_preds[mn] = yp
                    yg = np.array([g.y.item() for g in tg])
                    self.save_predictions(mn, yp, None, yg)
                except Exception as e:
                    print(f"    Error evaluating {mn}: {e}")
        else:
            print("  No test_graphs provided — skipping GNN on internal test set")

        results_df = pd.DataFrame(all_results)
        if len(results_df) > 0 and 'f1_score' in results_df.columns:
            results_df = results_df.sort_values('f1_score', ascending=False)

        sp = f"{self.config['results_dir']}/metrics/all_models_metrics.csv"
        Path(sp).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(sp, index=False)
        print(f"\nResults saved: {sp}")

        if run_statistical_tests and len(all_preds) >= 2:
            self._run_and_save_statistical_tests(all_preds, y_test,
                                                  'statistical_tests')

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY — ALL MODELS")
        print("=" * 70)
        if len(results_df) > 0:
            cols = ['model_name', 'model_type', 'accuracy',
                    'acc_ci_low', 'acc_ci_high',
                    'f1_score', 'f1_ci_low', 'f1_ci_high',
                    'mse', 'roc_auc']
            cols = [c for c in cols if c in results_df.columns]
            print(results_df[cols].to_string(index=False))

        return results_df

    def evaluate_cross_dataset(self, cross_data, run_statistical_tests=True):
        """
        Evaluate all 17 models on Modified_SQL_Dataset (30,907 samples,
        never seen in training).

        GCN and GAT are evaluated on ALL cross-dataset samples — no cap.
        The 6,000 limit was a training constraint (Kaggle runtime during
        backprop). Inference here is forward-pass only.

        XSS is not present in this split by design. See module header for
        the full justification of why XSS cross-dataset validation is not
        feasible in this study.

        cross_data must contain:
            X_cross_tfidf, X_cross_uniembed, X_cross_text, y_cross
        """
        n_cross = len(cross_data['y_cross'])
        print("\n" + "-" * 70)
        print("CROSS-DATASET EVALUATION  "
              "(Modified_SQL_Dataset — never seen in training)")
        print("-" * 70)
        print(f"  Samples: {n_cross}  |  SQL only  "
              "|  XSS: not applicable (see paper §3.2)")
        print(f"  GNN evaluation: ALL {n_cross} samples "
              "(no cap — inference only, no training constraint)")
        print("\n" + "=" * 70)
        print("COMPREHENSIVE MODEL EVALUATION (WITH ERROR METRICS)")
        print("=" * 70)

        all_results, all_preds = [], {}
        y_cross = cross_data['y_cross']

        # Classical ML
        print("\n" + "-" * 70)
        print("EVALUATING CLASSICAL ML MODELS")
        print("-" * 70)
        for mn in CLASSICAL_ML_MODELS:
            try:
                m, yp, yb = self.evaluate_classical_ml(
                    mn, cross_data['X_cross_tfidf'], y_cross, compute_ci=False)
                m.update({'model_name': mn, 'model_type': 'Classical ML'})
                all_results.append(m)
                all_preds[mn] = yp
            except Exception as e:
                print(f"    Error evaluating {mn}: {e}")

        # Deep Learning
        print("\n" + "-" * 70)
        print("EVALUATING DEEP LEARNING MODELS")
        print("-" * 70)
        for mn in DEEP_LEARNING_MODELS:
            try:
                m, yp, yb = self.evaluate_deep_learning(
                    mn, cross_data['X_cross_uniembed'], y_cross, compute_ci=False)
                m.update({'model_name': mn, 'model_type': 'Deep Learning'})
                all_results.append(m)
                all_preds[mn] = yp
            except Exception as e:
                print(f"    Error evaluating {mn}: {e}")

        # Transformers
        if 'X_cross_text' in cross_data:
            print("\n" + "-" * 70)
            print("EVALUATING TRANSFORMER MODELS")
            print("-" * 70)
            for mn in TRANSFORMER_MODELS:
                try:
                    m, yp, yb = self.evaluate_transformer(
                        mn, cross_data['X_cross_text'], y_cross, compute_ci=False)
                    m.update({'model_name': mn, 'model_type': 'Transformer'})
                    all_results.append(m)
                    all_preds[mn] = yp
                except Exception as e:
                    print(f"    Error evaluating {mn}: {e}")

        # GNN — build graphs from ALL X_cross samples
        print("\n" + "-" * 70)
        print(f"EVALUATING GNN MODELS (cross-dataset, all {n_cross} samples)")
        print("-" * 70)
        gnn_ok = False
        try:
            cross_graphs = self._build_gnn_graphs_for_cross(
                cross_data['X_cross_text'], y_cross)

            for mn in ['GCN', 'GAT']:
                pt = f"{self.config['models_dir']}/gnn/{mn}.pt"
                if not Path(pt).exists():
                    print(f"  {mn} checkpoint not found — skipping.")
                    continue
                try:
                    m, yp, _ = self.evaluate_gnn(mn, cross_graphs, compute_ci=False)
                    m.update({'model_name': mn, 'model_type': 'GNN',
                              'gnn_cross_n_samples': len(cross_graphs)})
                    all_results.append(m)
                    all_preds[mn] = yp
                    gnn_ok = True
                except Exception as e:
                    print(f"    Error evaluating {mn}: {e}")

            del cross_graphs
            gc.collect()

        except FileNotFoundError as e:
            print(f"  Skipping GNN cross-dataset: {e}")
        except Exception as e:
            print(f"  GNN graph build failed: {e}")
            import traceback
            traceback.print_exc()

        if not gnn_ok:
            print("  GNN cross-dataset evaluation could not be completed.")

        cross_df = pd.DataFrame(all_results)
        if len(cross_df) > 0 and 'f1_score' in cross_df.columns:
            cross_df = cross_df.sort_values('f1_score', ascending=False)

        cp = f"{self.config['results_dir']}/metrics/cross_dataset_metrics.csv"
        Path(cp).parent.mkdir(parents=True, exist_ok=True)
        cross_df.to_csv(cp, index=False)
        print(f"\nResults saved: {cp}")

        if run_statistical_tests and len(all_preds) >= 2:
            self._run_and_save_statistical_tests(
                all_preds, y_cross,
                filename_prefix='cross_dataset_statistical_tests')

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY — ALL MODELS")
        print("=" * 70)
        if len(cross_df) > 0:
            cols = ['model_name', 'model_type', 'accuracy', 'f1_score',
                    'mse', 'rmse', 'roc_auc']
            cols = [c for c in cols if c in cross_df.columns]
            print(cross_df[cols].to_string(index=False))

        print("\nCross-dataset evaluation complete! "
              "Results saved to cross_dataset_metrics.csv")
        return cross_df


    def save_predictions(self, model_name, y_pred, y_pred_proba, y_test):
        pred_dir = f"{self.config['results_dir']}/predictions"
        Path(pred_dir).mkdir(parents=True, exist_ok=True)
        data = {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist(),
                'correct': (y_pred == y_test).tolist()}
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                data['y_pred_proba_class_1'] = y_pred_proba[:, 1].tolist()
            else:
                data['y_pred_proba_class_1'] = y_pred_proba.flatten().tolist()
        with open(f"{pred_dir}/{model_name}_predictions.json", 'w') as f:
            json.dump(data, f, indent=2)



if __name__ == "__main__":
    set_seed(GLOBAL_CONFIG['random_seed'])
    feature_dir = f"{GLOBAL_CONFIG['base_dir']}/features"
    data_dir    = f"{GLOBAL_CONFIG['base_dir']}/data"

    test_data = {
        'X_test_tfidf':    np.load(f"{feature_dir}/X_test_tfidf.npy"),
        'X_test_uniembed': np.load(f"{feature_dir}/X_test_uniembed.npy"),
        'X_test_text':     np.load(f"{data_dir}/X_test.npy", allow_pickle=True),
        'y_test':          np.load(f"{data_dir}/y_test.npy"),
    }
    gnn_pkl = f"{data_dir}/test_graphs.pkl"
    if Path(gnn_pkl).exists():
        with open(gnn_pkl, 'rb') as f:
            test_data['test_graphs'] = pickle.load(f)

    ev = ComprehensiveEvaluator(GLOBAL_CONFIG)
    results_df = ev.evaluate_all_models(test_data, run_statistical_tests=True)

    if (Path(f"{feature_dir}/X_cross_tfidf.npy").exists() and
            Path(f"{data_dir}/y_cross.npy").exists()):
        cross_data = {
            'X_cross_tfidf':    np.load(f"{feature_dir}/X_cross_tfidf.npy"),
            'X_cross_uniembed': np.load(f"{feature_dir}/X_cross_uniembed.npy"),
            'X_cross_text':     np.load(f"{data_dir}/X_cross.npy", allow_pickle=True),
            'y_cross':          np.load(f"{data_dir}/y_cross.npy"),
        }
        ev.evaluate_cross_dataset(cross_data, run_statistical_tests=True)
    else:
        print("\nCross-dataset features not found — skipping.")

    print(f"\nEvaluation complete! Evaluated {len(results_df)} models.")
