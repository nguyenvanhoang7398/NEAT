from constants import *
from utils import *
from nltk.tokenize import TweetTokenizer
# from nltk import pos_tag
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def build_docs_vocab():
    if os.path.exists(OUTPUT_VOCAB_PATH):
        print("Vocab exists in {}".format(OUTPUT_VOCAB_PATH))
        return load_text_as_list(OUTPUT_VOCAB_PATH)
    input_valid_docs = read_json("data/pod/meta/train_test.json")["train"]
    tokenizer = TweetTokenizer()
    corpus = []
    for file_name in tqdm(os.listdir(OUTPUT_DOC_DIR), "Building vocabulary"):
        doc_id = file_name[:-5]
        if doc_id in input_valid_docs:
            doc_data = read_json(os.path.join(OUTPUT_DOC_DIR, file_name))
            for post in doc_data.values():
                tokens = [t for t in tokenizer.tokenize(post["t"].lower()) if t not in EXCLUDED_WORDS]
                corpus.extend(tokens)
    word_count = Counter(corpus).most_common(MAX_VOCAB_SIZE)
    vocab = [w[0] for w in word_count if w[1] >= MIN_VOCAB_COUNT]
    save_list_as_text(vocab, OUTPUT_VOCAB_PATH)
    print("Create vocab at {}".format(OUTPUT_VOCAB_PATH))
    return vocab


def build_docs_users():
    if os.path.exists(OUTPUT_USER_PATH):
        print("Users exists in {}".format(OUTPUT_USER_PATH))
        return load_text_as_list(OUTPUT_USER_PATH)
    user_corpus = []
    for file_name in tqdm(os.listdir(OUTPUT_DOC_DIR), "Building users"):
        doc_data = read_json(os.path.join(OUTPUT_DOC_DIR, file_name))
        for post in doc_data.values():
            user_corpus.append(post["a"])
    word_count = Counter(user_corpus).most_common(MAX_USER_SIZE)
    users = [w[0] for w in word_count if w[1] >= MIN_USER_COUNT]
    save_list_as_text(users, OUTPUT_USER_PATH)
    return users


def build_expertise_vectors():
    if os.listdir(OUTPUT_USER_PCA_DIR):
        print("User expertise exists in {}".format(OUTPUT_USER_PCA_DIR))
        return
    users = build_docs_users()
    user_participated_se_map = read_json(OUTPUT_USER_PARTICIPATED_SE_MAP_PATH)

    user_ses_list = [user_participated_se_map[user] if user in user_participated_se_map else []
                     for user in users]
    mlb = MultiLabelBinarizer()
    user_one_hot_mtx = mlb.fit_transform(user_ses_list)

    print("Fit scaler")
    scaler = MinMaxScaler()
    scaler_batch_size = 1024
    i = 0
    while i < len(user_one_hot_mtx):
        scaler.partial_fit(user_one_hot_mtx[i: i + scaler_batch_size])
        i += scaler_batch_size
    scaled_user_embedding = []
    i = 0
    while i < len(user_one_hot_mtx):
        scaled_user_embedding_batch = scaler.transform(user_one_hot_mtx[i: i + scaler_batch_size])
        scaled_user_embedding.extend(scaled_user_embedding_batch)
        i += scaler_batch_size
    unk_user_embedding = np.zeros(len(scaled_user_embedding[0]))
    scaled_user_embedding = np.vstack([unk_user_embedding] + scaled_user_embedding)

    for num_components in tqdm(range(5, 300, 5), desc="Building PCA for user expertise"):
        pca = PCA(n_components=num_components)
        reduced_user_embedding = pca.fit_transform(scaled_user_embedding)
        save_to_pickle(reduced_user_embedding, OUTPUT_USER_PCA_RESULT_PATH.format(num_components))
        save_to_pickle(pca, OUTPUT_USER_PCA_PATH.format(num_components))


def plot_explained_variance():
    pca_data = []
    for file_name in os.listdir(OUTPUT_USER_PCA_DIR):
        if file_name.startswith("pca"):
            pca = load_from_pickle(os.path.join(OUTPUT_USER_PCA_DIR, file_name))
            explained_variance = pca.explained_variance_ratio_
            summed_explained_variance = np.sum(explained_variance)
            pca_data.append((pca.n_components_, summed_explained_variance))

    sorted_pca_data = sorted(pca_data, key=lambda x: x[0])
    percentage_explained_var_list = [x[1] * 100 for x in sorted_pca_data]

    num_component_list = [x[0] for x in sorted_pca_data]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(num_component_list, percentage_explained_var_list, 'b*-')
    ax.set_ylim((0, 100))
    plt.grid(True)
    plt.xlabel('Number of principle components')
    plt.ylabel('Percentage of variance explained (%)')
    plt.savefig("pca.png")


def analyse_all_class_distribution():
    doc_ses = read_json(OUTPUT_DOC_SE_MAP_PATH)
    train_test_path = os.path.join(OUTPUT_META_DIR, "train_test.json")
    train_test = read_json(train_test_path)
    standard_se_map = read_json(OUTPUT_STANDARD_SE_MAP_PATH)
    side_effects = set(standard_se_map.values())
    print(len(side_effects))
    train_docs = train_test["train"]
    val_docs = train_test["val"]
    test_docs = train_test["test"]
    print("Train")
    analyse_class_distribution(doc_ses, train_docs, side_effects)
    print("Val")
    analyse_class_distribution(doc_ses, val_docs, side_effects)
    print("Test")
    analyse_class_distribution(doc_ses, test_docs, side_effects)


def analyse_class_distribution(doc_ses, docs, side_effects, k=5):
    class_corpus = []
    [class_corpus.extend(doc_ses[doc]) for doc in docs]
    class_counter = Counter(class_corpus).most_common()
    class_percentage = [(cls, cnt/len(docs)) for cls, cnt in class_counter]
    all_classes = set([x[0] for x in class_counter])
    top_k_classes = class_percentage[:k]
    bottom_k_classes = class_percentage[::-1][:k]
    missing_classes = side_effects.difference(all_classes)
    print("Top {} classes: ".format(top_k_classes))
    print("Bottom {} classes: ".format(bottom_k_classes))
    print("Missing classes: {}, {}".format(missing_classes, len(missing_classes)))


NEG = "NEG"
FCONJ = "FCONJ"
CCONJ = 'CCONJ'
ICONJ = 'ICONJ'
SMODAL = 'SMODAL'
WMODAL = 'WMODAL'
COND = 'COND'
ADVERB = 'ADVERB'
ADJECTIVE = 'ADJECTIVE'
PNOUN = 'PNOUN'
CNOUN = 'CNOUN'
FPERSON = 'FPERSON'
SPERSON = 'SPERSON'
TPERSON = 'TPERSON'
DDET = 'DDET'
IDET = 'IDET'
OTHER = 'OTHER'

ALL_TAGS = [NEG, FCONJ, CCONJ, ICONJ, SMODAL, WMODAL, COND, ADVERB, ADJECTIVE,
            PNOUN, CNOUN, FPERSON, SPERSON, TPERSON, DDET, IDET, OTHER]
TAG2IDX = {tag: i for i, tag in enumerate(ALL_TAGS)}


def classify_pos_tag(token, tag):
    if token in ["no", "not", "neither", "nor", "never"]:
        return NEG
    if token in ["but", "however", "nevertheless", "otherwise", "yet", "still", "nonetheless"]:
        return FCONJ
    if token in ["till", "until", "despite", "inspite", "though", "although"]:
        return CCONJ
    if token in ["therefore", "furthermore", "consequently", "thus", "as", "subsequently", "eventually", "hence"]:
        return ICONJ
    if token in ["might", "could", "can", "would", "may"]:
        return SMODAL
    if token in ["should", "ought", "need", "shall", "will", "must"]:
        return WMODAL
    if token in ["if"]:
        return COND
    if tag[:2] == "RB":
        return ADVERB
    if tag[:2] == "JJ":
        return ADJECTIVE
    if tag in ["NNP", "NNPS"]:
        return PNOUN
    if tag in ["NN", "NNS"]:
        return CNOUN
    if token in ["i", "we", "me", "us", "my", "mine", "our", "ours"]:
        return FPERSON
    if token in ["you", "your", "yours"]:
        return SPERSON
    if token in ["he", "she", "him", "her", "his", "it", "its", "hers", "they", "them", "their", "theirs"]:
        return TPERSON
    if tag[:2] == "DT" and token[:2] == "th":
        return DDET
    elif tag[:2] == "DT":
        return IDET
    return OTHER


# def pos_tag_post(post_tokens):
#     tagged_tokens = pos_tag(post_tokens)
#     return [classify_pos_tag(token, tag) for token, tag in tagged_tokens]
#
#
# def build_style_vector():
#     if os.path.exists(OUTPUT_USER_STYLE_MATRIX_PATH):
#         print("User style matrix exists in {}".format(OUTPUT_USER_STYLE_MATRIX_PATH))
#         return load_from_pickle(OUTPUT_USER_STYLE_MATRIX_PATH)
#     input_valid_docs = read_json("data/pod/meta/train_test.json")["train"]
#     all_users = build_docs_users()
#     tokenizer = TweetTokenizer()
#     user_style_corpus = {}
#     for file_name in tqdm(os.listdir(OUTPUT_DOC_DIR), "Building user style vector"):
#         doc_id = file_name.replace(".json", "")
#         if doc_id in input_valid_docs:
#             doc_data = read_json(os.path.join(OUTPUT_DOC_DIR, file_name))
#             for post in doc_data.values():
#                 tokens = [t for t in tokenizer.tokenize(post["t"].lower()) if t not in EXCLUDED_WORDS]
#                 user = post["a"]
#                 styles = pos_tag_post(tokens)
#                 if user not in user_style_corpus:
#                     user_style_corpus[user] = []
#                 user_style_corpus[user].extend(styles)
#     user_style_mtx = []
#     for user in all_users:
#         user_style_vector = np.zeros(len(ALL_TAGS))
#         if user in user_style_corpus:
#             style_corpus = user_style_corpus[user]
#             corpus_size = len(style_corpus)
#             style_count = Counter(style_corpus).most_common()
#             for style, cnt in style_count:
#                 user_style_vector[TAG2IDX[style]] += (float(cnt) / corpus_size)
#         user_style_mtx.append(user_style_vector)
#     unk_user_embedding = np.zeros(len(ALL_TAGS))
#     user_style_mtx = np.vstack([unk_user_embedding] + user_style_mtx)
#     save_to_pickle(user_style_mtx, OUTPUT_USER_STYLE_MATRIX_PATH)
#     min_max_scaler = MinMaxScaler()
#     user_style_mtx = min_max_scaler.fit_transform(user_style_mtx)
#     return user_style_mtx


if __name__ == "__main__":
    build_docs_vocab()
    build_docs_users()
    build_expertise_vectors()
    # build_style_vector()
    plot_explained_variance()
    # analyse_all_class_distribution()
