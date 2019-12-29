from constants import *
from utils import *
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def normalize(mtx):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mtx.sum(axis=1))
    sum_inv = np.power(row_sum, -1).flatten()
    sum_inv[np.isinf(sum_inv)] = 0
    row_mtx_inv = np.diag(sum_inv)
    return row_mtx_inv.dot(mtx)


def preprocess_pod(valid_docs=None):
    if len(os.listdir(OUTPUT_DOC_DIR)) > 0:
        print("POD docs exist in {}".format(OUTPUT_DOC_DIR))
        return
    author_review = read_csv(AUTHOR_DOC_REVIEW_PATH, True, delimiter="\t", quotechar=None)
    pod_data = {}
    batch_size = 10000
    all_doc_ids = set()
    for row in tqdm(author_review, desc="Preprocessing POD"):
        author_id, doc_id, doc_text = row[0], row[1], row[2]
        if valid_docs is not None and doc_id not in valid_docs:
            continue
        if doc_id not in pod_data:
            pod_data[doc_id] = {}
        pod_data[doc_id]["{}".format(len(pod_data[doc_id]))] = {
            "a": author_id,
            "t": doc_text
        }
        if len(pod_data) == batch_size:
            dump_doc_data(pod_data)
            pod_data = {}
        all_doc_ids.add(doc_id)
    dump_doc_data(pod_data)


def dump_doc_data(pod_data, output_dir=OUTPUT_DOC_DIR):
    for doc_id, posts in pod_data.items():
        doc_path = os.path.join(output_dir, "{}.json".format(doc_id))
        if os.path.exists(doc_path):
            doc_data = read_json(doc_path)
            for post_id, post_data in posts.items():
                doc_data[post_id] = post_data
        else:
            doc_data = posts
        write_json(doc_data, doc_path)


def build_user_doc_map():
    if os.path.exists(OUTPUT_USER_DOC_MAP_PATH):
        print("User doc map exists in {}".format(OUTPUT_USER_DOC_MAP_PATH))
        return read_json(OUTPUT_USER_DOC_MAP_PATH)
    preprocess_pod()
    user_doc_map = {}
    for doc_file in tqdm(os.listdir(OUTPUT_DOC_DIR), desc="Building user doc map"):
        doc_data = read_json(os.path.join(OUTPUT_DOC_DIR, doc_file))
        doc_id = doc_file.replace(".json", "")
        user_ids = [posts["a"] for posts in doc_data.values()]
        for user_id in user_ids:
            if user_id not in user_doc_map:
                user_doc_map[user_id] = set()
            user_doc_map[user_id].add(doc_id)
    user_doc_map = {user_id: list(docs) for user_id, docs in user_doc_map.items()}
    write_json(user_doc_map, OUTPUT_USER_DOC_MAP_PATH)
    return user_doc_map


def build_user_doc_drug_map(valid_docs=None, drug_vocab=None):
    if os.path.exists(OUTPUT_USER_MENTIONED_DRUG_MAP_PATH) and os.path.exists(OUTPUT_DOC_DRUG_MAP_PATH):
        print("User mentioned drug map exists in {}".format(OUTPUT_USER_MENTIONED_DRUG_MAP_PATH))
        print("Doc drug map exists in {}".format(OUTPUT_DOC_DRUG_MAP_PATH))
        return read_json(OUTPUT_USER_MENTIONED_DRUG_MAP_PATH), read_json(OUTPUT_DOC_DRUG_MAP_PATH)
    drug_vocab = set(drug_vocab)
    author_drug_docs = read_csv(AUTHOR_DRUG_DOCS_PATH, True, "\t")
    doc_drug_map = {doc: set() for doc in valid_docs}
    author_drug_map = {}
    for row in tqdm(author_drug_docs, desc="Building user doc drug map"):
        doc_list = [t for t in row[2][1:-1].split(", ")]  # remove brackets, commas
        author_experienced_docs = [d for d in doc_list if d in valid_docs]
        author_id, drug = row[0], row[1]
        if drug in drug_vocab:
            for doc_id in doc_list:
                if doc_id not in doc_drug_map:
                    doc_drug_map[doc_id] = set()
                doc_drug_map[doc_id].add(drug)
            if len(author_experienced_docs) > 0:
                if author_id not in author_drug_map:
                    author_drug_map[author_id] = set()
                author_drug_map[author_id].add(drug)
    author_drug_map = {author: list(drugs) for author, drugs in author_drug_map.items()}
    doc_drug_map = {doc: list(drugs) for doc, drugs in doc_drug_map.items()}
    write_json(author_drug_map, OUTPUT_USER_MENTIONED_DRUG_MAP_PATH)
    write_json(doc_drug_map, OUTPUT_DOC_DRUG_MAP_PATH)
    return author_drug_map, doc_drug_map


def build_user_participated_drug_map(valid_docs=None, drugs=None):
    if os.path.exists(OUTPUT_USER_PARTICIPATED_DRUG_MAP_PATH):
        print("User participated drug map exists in {}".format(OUTPUT_USER_PARTICIPATED_DRUG_MAP_PATH))
        return read_json(OUTPUT_USER_PARTICIPATED_DRUG_MAP_PATH)
    user_doc_map = build_user_doc_map()
    _, doc_drug_map = build_user_doc_drug_map(valid_docs, drugs)
    user_drug_map = {}
    for user_id, docs in user_doc_map.items():
        participated_drugs = []
        [participated_drugs.extend(doc_drug_map[d]) for d in docs if d in doc_drug_map and d in valid_docs]
        user_drug_map[user_id] = list(set(participated_drugs))
    user_drug_map = {user: list(drugs) for user, drugs in user_drug_map.items()}
    write_json(user_drug_map, OUTPUT_USER_PARTICIPATED_DRUG_MAP_PATH)
    return user_drug_map


def analyse_map(analysing_map):
    value_corpus, key_value_counter = [], Counter()
    for user, drugs in analysing_map.items():
        value_corpus.extend(drugs)
        key_value_counter[user] = len(drugs)
    key_value_counter_cnt = [x[1] for x in key_value_counter.most_common()]     # key must have some value
    value_key_counter_cnt = [x[1] for x in Counter(value_corpus).most_common()]
    print("Num of keys: {}, Max: {}, Min: {}, Mean: {}, Std: {}, 50th percentile: {}".format(
        len(key_value_counter_cnt), max(key_value_counter_cnt),
        min(key_value_counter_cnt), np.mean(key_value_counter_cnt),
        np.std(key_value_counter_cnt), np.percentile(key_value_counter_cnt, 50)))

    print("Num of values: {}, Max: {}, Min: {}, Mean: {}, Std: {}, 50th percentile: {}".format(
        len(value_key_counter_cnt), max(value_key_counter_cnt),
        min(value_key_counter_cnt), np.mean(value_key_counter_cnt),
        np.std(value_key_counter_cnt), np.percentile(value_key_counter_cnt, 50)))


def k_means_cluster_users(user_drug_map, drug_vocab=None, name=""):
    output_dir = os.path.join(OUTPUT_CLUSTERING_DIR, name)
    if len(os.listdir(ensure_path(output_dir))) > 0:
        print("K means clustering exists in {}".format(output_dir))
        return
    user_drug_arr, user_list = [], []
    if os.path.exists(OUTPUT_USER_MTX_FORMAT_PATH.format(name)):
        user_drug_mtx, user_list = load_from_pickle(OUTPUT_USER_MTX_FORMAT_PATH.format(name))
    else:
        for user, drugs in tqdm(user_drug_map.items(), "Build one-hot drug vector"):
            one_hot_vector = np.zeros(shape=(len(drug_vocab)))  
            valid_drugs = [d for d in drugs if d in drug_vocab]
            if len(valid_drugs) == 0:
                continue
            for drug in valid_drugs:
                one_hot_vector[drug_vocab.index(drug)] = 1
            user_drug_arr.append(one_hot_vector)
            user_list.append(user)
        user_drug_mtx = np.array(user_drug_arr)
        save_to_pickle((user_drug_mtx, user_list), OUTPUT_USER_MTX_FORMAT_PATH.format(name))
    user_drug_mtx = normalize(user_drug_mtx)
    print(len(user_drug_mtx))
    range_n_clusters = [2] + list(range(3, 51, 1))
    max_silhouette_score, max_n_clusters = 0., 0
    for n_clusters in tqdm(range_n_clusters, desc="Clustering users by {} ses".format(name)):
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(user_drug_mtx)
        silhouette_avg = silhouette_score(user_drug_mtx, cluster_labels)
        output_result_path = os.path.join(output_dir, "cluster_{}_results.txt".format(n_clusters))
        output_meta_path = os.path.join(output_dir, "cluster_{}_meta.json".format(n_clusters))
        clustering_results = [str(user) + " " + str(cluster_labels[i]) for i, user in enumerate(user_list)]
        clustering_meta = {
            "silhouette_avg": silhouette_avg
        }
        save_list_as_text(clustering_results, output_result_path)
        write_json(clustering_meta, output_meta_path)
        max_silhouette_score = max(max_silhouette_score, silhouette_avg)
        max_n_clusters = n_clusters if max_silhouette_score == silhouette_avg else max_n_clusters
        print("Max max_silhouette_score {} of {} clusters".format(max_silhouette_score, max_n_clusters))


def build_doc_se_map(standard_doc_se_map, drug_vocab=None):
    if os.path.exists(OUTPUT_DOC_SE_MAP_PATH):
        print("Doc se map exists in {}".format(OUTPUT_DOC_SE_MAP_PATH))
        return read_json(OUTPUT_DOC_SE_MAP_PATH)
    drug_se_map = read_json(OUTPUT_DRUG_SE_MAP_PATH)
    _, doc_drug_map = build_user_doc_drug_map(input_valid_docs, drug_vocab)
    doc_se_map = {}
    for doc, drugs in tqdm(doc_drug_map.items(), desc="Build doc se map"):
        all_ses = []
        for d in drugs:
            drug_ses = drug_se_map[d]
            for t, ses in drug_ses.items():
                if t in SE_TYPES:
                    all_ses.extend(ses)
        all_ses = list(set(all_ses))       
        valid_ses = [standard_doc_se_map[se] for se in all_ses]
        if len(valid_ses) > 0:
            doc_se_map[doc] = list(set(valid_ses))
    write_json(doc_se_map, OUTPUT_DOC_SE_MAP_PATH)
    return doc_se_map


def build_user_se_maps(standard_doc_se_map, drug_vocab=None, valid_drugs=None):
    if os.path.exists(OUTPUT_USER_MENTIONED_SE_MAP_PATH) and os.path.exists(OUTPUT_USER_PARTICIPATED_SE_MAP_PATH):
        print("User mentioned se map exists in {}".format(OUTPUT_USER_MENTIONED_SE_MAP_PATH))
        print("User participated se map exists in {}".format(OUTPUT_USER_PARTICIPATED_SE_MAP_PATH))
        return read_json(OUTPUT_USER_MENTIONED_SE_MAP_PATH), read_json(OUTPUT_USER_PARTICIPATED_SE_MAP_PATH)
    drug_se_map = read_json(OUTPUT_DRUG_SE_MAP_PATH)
    user_mentioned_drug_map, _ = build_user_doc_drug_map(input_valid_docs, drug_vocab)
    user_participated_drug_map = build_user_participated_drug_map(valid_drugs, drug_vocab)
    user_mentioned_se_map, user_participated_se_map = {}, {}

    for doc, drugs in tqdm(user_mentioned_drug_map.items(), desc="Build user mentioned se map"):
        all_ses = []
        for d in drugs:
            drug_ses = drug_se_map[d]
            for t, ses in drug_ses.items():
                if t in SE_TYPES:
                    all_ses.extend(ses)
        all_ses = list(set(all_ses))
        valid_ses = [standard_doc_se_map[se] for se in all_ses]
        if len(valid_ses) > 0:
            user_mentioned_se_map[doc] = list(set(valid_ses))
    write_json(user_mentioned_se_map, OUTPUT_USER_MENTIONED_SE_MAP_PATH)

    for doc, drugs in tqdm(user_participated_drug_map.items(), desc="Build user participated se map"):
        all_ses = []
        for d in drugs:
            drug_ses = drug_se_map[d]
            for t, ses in drug_ses.items():
                if t in SE_TYPES:
                    all_ses.extend(ses)
        all_ses = list(set(all_ses))
        valid_ses = [standard_doc_se_map[se] for se in all_ses]
        if len(valid_ses) > 0:
            user_participated_se_map[doc] = list(set(valid_ses))
    write_json(user_participated_se_map, OUTPUT_USER_PARTICIPATED_SE_MAP_PATH)

    return user_mentioned_se_map, user_participated_se_map


if __name__ == "__main__":
    # input_valid_docs = load_text_as_list(OUTPUT_VALID_DOC_PATH)
    train_test_docs = read_json("data/pod/meta/train_test.json")
    input_valid_docs = train_test_docs["train"]
    preprocess_pod(input_valid_docs)
    # build_user_doc_map()
    drug_list = list(read_json(OUTPUT_DRUG_SE_MAP_PATH).keys())
    _, input_doc_drug_map = build_user_doc_drug_map(input_valid_docs, drug_list)
    print("Analysing doc_drug_map")
    analyse_map(input_doc_drug_map)
    print("Analysing user_participated_drug_map")
    input_user_participated_drug_map = build_user_participated_drug_map(input_valid_docs, drug_list)
    analyse_map(input_user_participated_drug_map)
    input_standard_doc_se_map = read_json(OUTPUT_STANDARD_SE_MAP_PATH)
    doc_se_map = build_doc_se_map(input_standard_doc_se_map, drug_list)
    input_standard_se_list = list(set(input_standard_doc_se_map.values()))
    input_user_mentioned_se_map, input_user_participated_se_map = build_user_se_maps(input_standard_doc_se_map,
                                                                                     drug_list, input_valid_docs)
    # k_means_cluster_users(input_user_participated_drug_map, drug_list, "experienced_filtered")
    k_means_cluster_users(input_user_participated_se_map, input_standard_se_list, "participated_filtered")
    k_means_cluster_users(input_user_mentioned_se_map, input_standard_se_list, "mentioned_filtered")
