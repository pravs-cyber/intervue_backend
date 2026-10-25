"""
Feature extraction for the answer classifier.

Three groups of hand-crafted features (all dense, all explainable) plus the answer's own TF-IDF:

  surface   : how the answer is written (length, sentences, repetition, function-word ratio, "I don't know")
  relevance : how the answer relates to the QUESTION (similarity, keyword overlap, how much is just copied)
  reference : how the answer relates to the REFERENCE answer (similarity, key-term coverage, length ratio)
  semantic  : LSA similarity (TF-IDF -> TruncatedSVD), a classical "meaning" space that matches related
              terms even when the exact words differ

  embedding : (experimental, v3) cosine similarity of IDF-weighted averaged pretrained GloVe vectors
              (answer vs question, answer vs reference). Unlike LSA it is learned from a large external corpus,
              so 'shares memory' and 'same address space' can be close without sharing words.

Reference features are optional: questions from the bank or the LLM may have no reference answer, so the
extractor can be used with or without them (see FEATURE_GROUPS and the 'noref' model).
"""
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#'-]*")
IDK = re.compile(r"\b(i\s*(do\s*n[o']?t|dont|don't)\s*know|no idea|not sure|haven'?t studied|have not studied|cannot explain|can't explain)\b", re.I)
STOP = set(ENGLISH_STOP_WORDS)

SURFACE = ["log_words", "n_sentences", "avg_sentence_len", "unique_ratio", "stopword_ratio", "repeat_ratio", "idk_flag"]
RELEVANCE = ["q_cosine", "q_keyword_overlap", "copied_from_question"]
REFERENCE = ["ref_cosine", "ref_coverage", "len_ratio_ref"]
SEMANTIC_Q = ["q_lsa"]
SEMANTIC_REF = ["ref_lsa"]
EMBED_Q = ["q_emb"]
EMBED_REF = ["ref_emb"]
EMB_PATH = Path(__file__).resolve().parent / "embeddings" / "glove100_100k.npz"


class Embedder:
    """IDF-weighted average of pretrained word vectors. Loaded lazily and NOT pickled (rebuilt from EMB_PATH)."""
    _table = None

    @classmethod
    def table(cls):
        if cls._table is None:
            if not EMB_PATH.exists():
                raise RuntimeError("Embeddings missing. Run: python ml/build_embeddings.py")
            z = np.load(EMB_PATH)
            cls._table = ({w: i for i, w in enumerate(z["words"])}, z["vectors"].astype(np.float32))
        return cls._table

    @classmethod
    def encode(cls, texts, idf):
        index, vecs = cls.table()
        out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [w for w in content(t) if w in index]
            if toks:
                wts = np.array([idf.get(w, 1.0) for w in toks], dtype=np.float32)
                out[i] = (vecs[[index[w] for w in toks]] * wts[:, None]).sum(0) / wts.sum()
        return out


def words(text):
    return [w.lower() for w in WORD.findall(text or "")]


def content(text):
    return [w for w in words(text) if w not in STOP and len(w) > 2]


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


class FeatureExtractor:
    """Fit on TRAINING questions + references only, then transform any (question, answer, reference)."""

    def __init__(self, lsa_dims=100, seed=42):
        self.tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, ngram_range=(1, 1))
        self.svd = TruncatedSVD(n_components=lsa_dims, random_state=seed)

    def fit(self, questions, references):
        corpus = list(questions) + list(references)
        X = self.tfidf.fit_transform(corpus)
        self.svd.fit(X)
        return self

    def _idf(self):
        if not hasattr(self, "_idf_map"):
            self._idf_map = dict(zip(self.tfidf.get_feature_names_out(), self.tfidf.idf_))
        return self._idf_map

    def _vecs(self, texts):
        X = self.tfidf.transform(texts)
        return X.toarray(), self.svd.transform(X)

    def transform(self, questions, answers, references=None, groups=("surface", "relevance", "reference", "semantic")):
        n = len(answers)
        refs = list(references) if references is not None else [""] * n
        q_t, q_l = self._vecs(questions)
        a_t, a_l = self._vecs(answers)
        r_t, r_l = self._vecs(refs)
        names, rows = feature_names(groups), []
        if "embedding" in groups:
            q_e, a_e, r_e = (Embedder.encode(x, self._idf()) for x in (questions, answers, refs))
        for i in range(n):
            q, a, r = questions[i], answers[i], refs[i]
            aw, ac, qc, rc = words(a), content(a), set(content(q)), set(content(r))
            sents = [s for s in re.split(r"(?<=[.!?])\s+", a.strip()) if s]
            f = {}
            if "surface" in groups:
                f["log_words"] = np.log1p(len(aw))
                f["n_sentences"] = len(sents)
                f["avg_sentence_len"] = len(aw) / max(1, len(sents))
                f["unique_ratio"] = len(set(aw)) / max(1, len(aw))
                f["stopword_ratio"] = sum(w in STOP for w in aw) / max(1, len(aw))
                tri = list(zip(aw, aw[1:], aw[2:]))
                f["repeat_ratio"] = 1 - len(set(tri)) / max(1, len(tri))
                f["idk_flag"] = float(bool(IDK.search(a)))
            if "relevance" in groups:
                f["q_cosine"] = _cos(q_t[i], a_t[i])
                f["q_keyword_overlap"] = len(qc & set(ac)) / max(1, len(qc))
                f["copied_from_question"] = sum(w in qc for w in ac) / max(1, len(ac))
            if "reference" in groups:
                f["ref_cosine"] = _cos(r_t[i], a_t[i])
                f["ref_coverage"] = len(rc & set(ac)) / max(1, len(rc))
                f["len_ratio_ref"] = min(3.0, len(aw) / max(1, len(words(r))))
            if "semantic" in groups:
                f["q_lsa"] = _cos(q_l[i], a_l[i])
                if "reference" in groups:
                    f["ref_lsa"] = _cos(r_l[i], a_l[i])
            if "embedding" in groups:
                f["q_emb"] = _cos(q_e[i], a_e[i])
                if "reference" in groups:
                    f["ref_emb"] = _cos(r_e[i], a_e[i])
            rows.append([f[k] for k in names])
        return np.array(rows, dtype=float), names


def feature_names(groups):
    names = []
    if "surface" in groups:
        names += SURFACE
    if "relevance" in groups:
        names += RELEVANCE
    if "reference" in groups:
        names += REFERENCE
    if "semantic" in groups:
        names += SEMANTIC_Q + (SEMANTIC_REF if "reference" in groups else [])
    if "embedding" in groups:
        names += EMBED_Q + (EMBED_REF if "reference" in groups else [])
    return names


def answer_tfidf():
    """The original lexical representation of the answer text alone (ablation A)."""
    return TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2, sublinear_tf=True, norm=None)
