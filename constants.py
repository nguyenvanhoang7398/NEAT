import os
from nltk.corpus import stopwords
import string

STOP_WORDS = set(stopwords.words('english'))
PUNCTUATIONS = string.punctuation
USELESS_WORDS = ["become", "turn", "really", "very", "often", "like", "certain", "feeling", "feel"]
EXCLUDED_WORDS = list(STOP_WORDS) + list(PUNCTUATIONS) + USELESS_WORDS
POD_DIR = "data/pod"
AUTHOR_DOC_REVIEW_PATH = os.path.join(POD_DIR, "Author-Doc-Review.tsv")
EXPERT_SE_PATH = os.path.join(POD_DIR, "Expert-Drug-SideEffects.tsv")
AUTHOR_DOC_SYMPTOMS_PATH = os.path.join(POD_DIR, "Author-Doc-Symtpoms.tsv")
AUTHOR_DRUG_DOCS_PATH = os.path.join(POD_DIR, "Author-Drug-Docs.tsv")
AUTHOR_DETAIL_PATH = os.path.join(POD_DIR, "Author-Details.tsv")

OUTPUT_CORPUS_DIR = "output/corpus"
OUTPUT_DRUG_DOC_PATH = os.path.join(OUTPUT_CORPUS_DIR, "drug_docs.json")
OUTPUT_VOCAB_PATH = os.path.join(OUTPUT_CORPUS_DIR, "fold", "vocab.txt")
OUTPUT_USER_STYLE_MATRIX_PATH = os.path.join(OUTPUT_CORPUS_DIR, "fold", "user_style.pickle")
OUTPUT_IDF_PATH = os.path.join(OUTPUT_CORPUS_DIR, "idf.json")
OUTPUT_IDF_META_PATH = os.path.join(OUTPUT_CORPUS_DIR, "idf_meta.json")
MIN_VOCAB_COUNT = 2
MAX_VOCAB_SIZE = 20000
GLOVE_PATH = "data/glove.840B.300d.txt"
GLOVE_TMP_PATH = "data/glove_w2v.840B.300d.txt"
ADDITIONAL_NOUN_LIST_PATH = "data/noun_list.txt"
BODY_PART_LIST_PATH = "data/body_parts.txt"

OUTPUT_USER_PATH = os.path.join(OUTPUT_CORPUS_DIR, "fold", "users.txt")
OUTPUT_USER_EXPERTISE_PATH = os.path.join(OUTPUT_CORPUS_DIR, "fold", "user_expertise.pickle")
MIN_USER_COUNT = 1
MAX_USER_SIZE = 100000

OUTPUT_SINGLE_SE_VOCAB_PATH = os.path.join(OUTPUT_CORPUS_DIR, "custom_single_se_vocab.txt")
OUTPUT_SE_VOCAB_PATH = os.path.join(OUTPUT_CORPUS_DIR, "se_vocab.txt")
OUTPUT_STANDARD_SE_MAP_PATH = os.path.join(OUTPUT_CORPUS_DIR, "standard_se_map.json")
OUTPUT_CLUSTER_SE_VOCAB_PATH = os.path.join(OUTPUT_CORPUS_DIR, "cluster_se_vocab.txt")
OUTPUT_VALID_DOC_PATH = os.path.join(OUTPUT_CORPUS_DIR, "valid_docs.txt")
OUTPUT_USER_DOC_MAP_PATH = os.path.join(OUTPUT_CORPUS_DIR, "user_docs.json")
OUTPUT_DRUG_SE_MAP_PATH = os.path.join(OUTPUT_CORPUS_DIR, "drug_ses.json")
OUTPUT_USER_MENTIONED_DRUG_MAP_PATH = os.path.join(OUTPUT_CORPUS_DIR, "fold", "user_mentioned_drugs.json")
OUTPUT_USER_PARTICIPATED_DRUG_MAP_PATH = os.path.join(OUTPUT_CORPUS_DIR, "fold", "user_participated_drugs.json")
OUTPUT_DOC_DRUG_MAP_PATH = os.path.join(OUTPUT_CORPUS_DIR, "fold", "doc_drugs.json")
OUTPUT_DOC_SE_MAP_PATH = os.path.join(OUTPUT_CORPUS_DIR, "doc_ses.json")
OUTPUT_USER_MENTIONED_SE_MAP_PATH = os.path.join(OUTPUT_CORPUS_DIR, "fold", "user_mentioned_ses.json")
OUTPUT_USER_PARTICIPATED_SE_MAP_PATH = os.path.join(OUTPUT_CORPUS_DIR, "fold", "user_participated_ses.json")

OUTPUT_CLUSTERING_DIR = "output/clustering/fold"
OUTPUT_K_MEANS_DIR = os.path.join(OUTPUT_CLUSTERING_DIR, "k_means")
OUTPUT_USER_MTX_FORMAT_PATH = os.path.join(OUTPUT_CLUSTERING_DIR, "user_mtx_{}.pickle")

OUTPUT_USER_PCA_DIR = "output/pca/fold/"
OUTPUT_USER_PCA_PATH = os.path.join(OUTPUT_USER_PCA_DIR, "pca_{}.pickle")
OUTPUT_USER_PCA_RESULT_PATH = os.path.join(OUTPUT_USER_PCA_DIR, "reduced_user_embedding_{}.pickle")

MIN_SE_VOCAB_COUNT = 1
MAX_SE_VOCAB_SIZE = 50000

ADR_DIR = "data/adr"
META_PATH = os.path.join(ADR_DIR, "meta.json")
ADR_MODEL_PATH = os.path.join(ADR_DIR, "model.ckpt")

OUTPUT_DOC_DIR = os.path.join(POD_DIR, "docs")
OUTPUT_META_DIR = os.path.join(POD_DIR, "meta")
OUTPUT_USER_EXPERTISE_META_PATH = os.path.join(OUTPUT_META_DIR, "user_expertise.pickle")
OUTPUT_USER_STYLE_META_PATH = os.path.join(OUTPUT_META_DIR, "user_style.pickle")
OUTPUT_CACHE_DIR = os.path.join(POD_DIR, "cache")
OUTPUT_SE_DOC_DIR = os.path.join(POD_DIR, "se_docs")
OUTPUT_UMLS_DOC_DIR = os.path.join(POD_DIR, "umls_docs/")
OUTPUT_REVIEW_SENT_DIR = os.path.join(POD_DIR, "review_sents/")
SE_TYPES = ["more common", "less common"]

OUTPUT_BENCHMARK_DIR = "output/benchmark/"
OUTPUT_ADR_RESULT_PATH = os.path.join(OUTPUT_BENCHMARK_DIR, "adr_docs_se.json")
OUTPUT_UMLS_RESULT_PATH = os.path.join(OUTPUT_BENCHMARK_DIR, "umls_docs_se.json")

OUTPUT_BOW_DIR = os.path.join(POD_DIR, "bow")
OUTPUT_BOW_FEATURE = os.path.join(OUTPUT_BOW_DIR, "bow_feature_{}.pickle")
OUTPUT_BOW_LABEL = os.path.join(OUTPUT_BOW_DIR, "bow_label_{}.pickle")
OUTPUT_BOW_DOCS = os.path.join(OUTPUT_BOW_DIR, "docs_{}.txt")
OUTPUT_BOW_SVM = os.path.join(OUTPUT_BOW_DIR, "svm.pickle")
OUTPUT_BOW_RF = os.path.join(OUTPUT_BOW_DIR, "rf.pickle")
OUTPUT_BOW_SCALER = os.path.join(OUTPUT_BOW_DIR, "bow.pickle")
OUTPUT_BOW_TRAIN_X = os.path.join(OUTPUT_BOW_DIR, "bow_x.pickle")
OUTPUT_BOW_TRAIN_Y = os.path.join(OUTPUT_BOW_DIR, "bow_y.pickle")
OUTPUT_BOW_PCA = os.path.join(OUTPUT_BOW_DIR, "pca.pickle")

OUTPUT_USER_CREDIBILITY_DIR = "output/credibility"
OUTPUT_USER_LEARNED_CREDIBILITY_PATH = os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "user_credibility.tsv")
OUTPUT_USER_FEATURE_PATH = os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "user_features.json")
OUTPUT_USER_PRECISION_PATH = os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "user_precision.json")
OUTPUT_USER_CREDIBILITY_LIST_PATH = os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "metadata.tsv")
OUTPUT_USER_CREDIBILITY_SCORE_PATH = os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "tensors.tsv")

OUTPUT_ATTENTION_DIR = os.path.join(OUTPUT_BENCHMARK_DIR, "attention")
OUTPUT_TRUE_CREDIBILITY_RANKING = os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "user_ranking.json")
