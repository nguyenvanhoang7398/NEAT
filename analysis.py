from utils import *
from constants import *
from collections import Counter
from tqdm import tqdm
from model.metrics import precision_score
import numpy as np
from dataset import NeatLoader
from train import load_model
from infer import infer_fn
from model.utils import credibility_feature_correlation, credibility_precision_correlation
from preprocessing_data import build_docs_users
from scipy.stats import spearmanr
from nltk.tokenize import TweetTokenizer
from train import evaluate
from dataset import NeatThreadInputFeature


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def analyse_clusters(user_cluster_map_path):
    user_cluster_data = load_text_as_list(user_cluster_map_path)
    user_participated_se_map = read_json(OUTPUT_USER_PARTICIPATED_SE_MAP_PATH)
    user_participated_drug_map = read_json(OUTPUT_USER_PARTICIPATED_DRUG_MAP_PATH)
    user_mentioned_se_map = read_json(OUTPUT_USER_MENTIONED_SE_MAP_PATH)
    user_mentioned_drug_map = read_json(OUTPUT_USER_MENTIONED_DRUG_MAP_PATH)

    cluster_user_map = {}
    for row in user_cluster_data:
        tokens = row.split(" ")
        cluster, user = tokens[1], tokens[0]
        if cluster not in cluster_user_map:
            cluster_user_map[cluster] = []
        cluster_user_map[cluster].append(user)

    mapped_content_dicts = [
        ("participated side effects", user_participated_se_map),
        ("participated drugs", user_participated_drug_map),
        ("mentioned side effects", user_mentioned_se_map),
        ("mentioned drugs", user_mentioned_drug_map)
    ]
    top_k = 5
    export_data = {}
    for name, content_dict in mapped_content_dicts:
        export_data[name] = {}
        for cluster, users in cluster_user_map.items():
            mapped_data = []
            for user in users:
                if user in content_dict:
                    mapped_data.extend(content_dict[user])
            mapped_cnt = Counter(mapped_data).most_common(top_k)
            export_data[name][cluster] = [x[0] for x in mapped_cnt]
    write_json(export_data, "export_cluster.json")


def extract_user_features():
    if os.path.exists(OUTPUT_USER_FEATURE_PATH):
        print("User feature exists at {}".format(OUTPUT_USER_FEATURE_PATH))
        return read_json(OUTPUT_USER_FEATURE_PATH)
    author_detail_content = read_csv(AUTHOR_DETAIL_PATH, None, "\t")
    author_features = {}
    for row in tqdm(author_detail_content, desc="Extracting user features"):
        author_id, num_posts, num_questions, num_replies, \
        num_thanks, membership = row[0], row[-5], row[-3], row[-2], row[-1], row[-4]
        author_features[author_id] = {
            "n_posts": int(num_posts),
            "n_questions": int(num_questions) if num_questions != "null" else 0,
            "n_replies": int(num_replies) if num_replies != "null" else 0,
            "n_thanks": int(num_thanks) if num_thanks != "null" else 0,
            "membership": membership
        }

    write_json(author_features, OUTPUT_USER_FEATURE_PATH)
    return author_features


def compile_adr(doc_se_map):
    standard_se_map = read_json(OUTPUT_STANDARD_SE_MAP_PATH)
    doc_user_se_map = {}
    for batch_file in tqdm(os.listdir(OUTPUT_SE_DOC_DIR), desc="Compiling ADR results"):
        se_doc_content = read_csv(os.path.join(OUTPUT_SE_DOC_DIR, batch_file), True, "\t")
        for row in se_doc_content:
            user_id, doc_id, raw_ses = row[1], row[2], row[3][1:-1]  # eliminate brackets
            if doc_id in doc_se_map:
                tokens = [x for x in raw_ses.split(",") if len(x) > 1]
                parsed_ses = [se.replace("\'", "").strip().rstrip() for se in tokens]
                if len(parsed_ses) > 0:
                    standardized_ses = set([standard_se_map[se] for se in parsed_ses])
                    if doc_id not in doc_user_se_map:
                        doc_user_se_map[doc_id] = {}
                    if user_id not in doc_user_se_map[doc_id]:
                        doc_user_se_map[doc_id][user_id] = set()
                    doc_user_se_map[doc_id][user_id] = doc_user_se_map[doc_id][user_id].union(standardized_ses)
    return doc_user_se_map


def compile_umls(doc_se_map):
    standard_se_map = read_json(OUTPUT_STANDARD_SE_MAP_PATH)
    doc_user_se_map = {}
    for doc_file in tqdm(os.listdir(OUTPUT_UMLS_DOC_DIR), desc="Compiling UMLS results"):
        doc_id = doc_file.replace(".json", "")
        if doc_id in doc_se_map:
            umls_doc_content = read_json(os.path.join(OUTPUT_UMLS_DOC_DIR, doc_file))
            for user_id, ses in umls_doc_content.items():
                standardized_ses = [standard_se_map[se] for se in ses]
                if doc_id not in doc_user_se_map:
                    doc_user_se_map[doc_id] = {}
                if user_id not in doc_user_se_map[doc_id]:
                    doc_user_se_map[doc_id][user_id] = set()
                doc_user_se_map[doc_id][user_id] = doc_user_se_map[doc_id][user_id].union(standardized_ses)
    return doc_user_se_map


def extract_user_precision():
    if os.path.exists(OUTPUT_USER_PRECISION_PATH):
        print("User precision map in {}".format(OUTPUT_USER_PRECISION_PATH))
        return read_json(OUTPUT_USER_PRECISION_PATH)
    doc_se_map = read_json(OUTPUT_DOC_SE_MAP_PATH)
    doc_user_se_map = compile_umls(doc_se_map)
    user_precision_map = {}
    for doc_id, doc_content in tqdm(doc_user_se_map.items(), desc="Evaluating user precision"):
        for user_id, reported_ses in doc_content.items():
            true_ses = doc_se_map[doc_id]
            precision = precision_score(true_ses, reported_ses)
            if user_id not in user_precision_map:
                user_precision_map[user_id] = []
            user_precision_map[user_id].append(precision)
    user_precision_map = {user_id: np.mean(precisions) for user_id, precisions in user_precision_map.items()}
    write_json(user_precision_map, OUTPUT_USER_PRECISION_PATH)
    return user_precision_map


def credibility_correlation(user_features=None, user_precision=None, user_list=None, credibility_list=None):
    user_features = extract_user_features() if user_features is None else user_features
    user_precision = extract_user_precision() if user_precision is None else user_precision
    user_list = load_text_as_list(OUTPUT_USER_CREDIBILITY_LIST_PATH) if user_list is None else user_list
    credibility_list = load_text_as_list(OUTPUT_USER_CREDIBILITY_SCORE_PATH) \
        if credibility_list is None else credibility_list
    post_corr, question_corr, reply_corr, thank_corr = credibility_feature_correlation(user_features,
                                                                                       user_list, credibility_list)
    precision_corr = credibility_precision_correlation(user_precision, user_list, credibility_list)
    return post_corr, question_corr, reply_corr, thank_corr, precision_corr


def precision_thank_correlation():
    user_features = extract_user_features()
    user_precision_umls = read_json(os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "user_precision_umls.json"))
    user_precision_adr = read_json(os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "user_precision_adr.json"))

    all_precisions = [
        ("umls", user_precision_umls),
        ("adr", user_precision_adr)
    ]

    for name, user_precision in all_precisions:
        precision_list, thank_list, normalized_thank_list = [], [], []
        for user, precision in tqdm(user_precision.items(), "Thank correlation"):
            if user in user_features:
                if user_features[user]["n_thanks"] > 0:
                    thank_list.append(user_features[user]["n_thanks"])
                    normalized_thank_list.append(user_features[user]["n_thanks"]/
                                                 float(1+user_features[user]["n_replies"]))
                    precision_list.append(precision)
        norm_thank_corr = spearmanr(precision_list, normalized_thank_list)
        thank_corr = spearmanr(precision_list, thank_list)
        print("For {} precision".format(name))
        print("Thank correlation: {}".format(thank_corr))
        print("Norm thank correlation: {}".format(norm_thank_corr))


def search_best_precision_correlations():
    user_features = extract_user_features()
    model_dirs = [
        "exp_logs/neat_cnn_full-12_10_19-08-53-47",
        "exp_logs/neat_cnn_wpe-12_10_19-08-54-00",
        "exp_logs/neat_cnn_wpeu-12_10_19-08-53-56",
        "exp_logs/neat_wpeu-12_10_19-10-06-18",
        "exp_logs/neat_full-12_10_19-10-06-32",
        "exp_logs/neat_wpe-12_10_19-10-49-46",
        "exp_logs/neat_cnn_full-12_09_19-16-49-03",
        "exp_logs/neat_cnn_wpeu-12_08_19-13-56-26",
        "exp_logs/neat_cnn_wpe-12_08_19-13-55-55"
    ]
    precision_paths = [
        os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "user_precision_umls.json"),
        os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "user_precision_adr.json")
    ]

    all_correlations = []

    for precision_path in tqdm(precision_paths, desc="Precision type"):
        user_precision = read_json(precision_path)
        for model_dir in tqdm(model_dirs, desc="Model type"):
            for file_name in tqdm(os.listdir(model_dir), desc="File name"):
                if is_int(file_name):
                    user_list = load_text_as_list(os.path.join(model_dir, file_name,
                                                  "User credibility", "metadata.tsv"))
                    credibility_list = load_text_as_list(os.path.join(model_dir, file_name,
                                                         "User credibility", "tensors.tsv"))
                    post_corr, question_corr, reply_corr, \
                        thank_corr, precision_corr = credibility_correlation(user_features, user_precision,
                                                                             user_list, credibility_list)
                    all_correlations.append((model_dir, file_name, precision_path,
                                             post_corr, question_corr, reply_corr,
                                             thank_corr, precision_corr))
    save_to_pickle(all_correlations, os.path.join(OUTPUT_USER_CREDIBILITY_DIR, "all_correlations.pickle"))


def test_all_models():
    data_dir = OUTPUT_DOC_DIR
    meta_dir = OUTPUT_META_DIR
    cache_dir = OUTPUT_CACHE_DIR
    #w2v_path = "data/glove.6B.300d.txt"
    w2v_path = ""
    ue_path = OUTPUT_USER_EXPERTISE_META_PATH
    ue_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = NeatLoader(data_dir, meta_dir, cache_dir,
                             w2v_path, ue_path)

    model_list = [
        ("neat_cnn_full", "exp_ckpt/neat_cnn_full-12_10_19-08-53-47", 1959999),
        ("neat_cnn_wpeu", "exp_ckpt/neat_cnn_wpeu-12_10_19-08-53-56", 2179999),
        ("neat_cnn_wpe", "exp_ckpt/neat_cnn_wpe-12_10_19-08-54-00", 119999),
        ("neat_cnn", "exp_ckpt/neat_cnn-12_10_19-09-24-08", 99999)
    ]

    test_docs = read_json("data/pod/meta/train_test.json")["test"]
    for model_name, model_path, checkpoint in tqdm(model_list, desc="Models"):
        model_path_full = os.path.join(model_path, "model_ckpt_{}.tar".format(checkpoint))
        input_model = load_model(model_name, ue_size, data_loader)
        loaded_state_dict = torch.load(model_path_full)["state_dict"]
        input_model.load_state_dict(loaded_state_dict)
        input_model.to(device)
        output_dir = os.path.join(OUTPUT_BENCHMARK_DIR, model_name)
        ensure_path(os.path.join(output_dir, "a.txt"))
        for doc_file in tqdm(os.listdir(OUTPUT_DOC_DIR), desc="Test docs"):
            if doc_file.replace(".json", "") in test_docs:
                thread_content = read_json(os.path.join(OUTPUT_DOC_DIR, doc_file))
                predict_side_effects, user_creds, most_attended_word_posts = \
                    infer_fn(input_model, data_loader, thread_content, device, verbose=False)
                result = {
                    "predictions": list(predict_side_effects),
                    "user_creds": [(x[0], float(x[1])) for x in user_creds],
                    "attention": most_attended_word_posts
                }
                write_json(result, os.path.join(output_dir, doc_file))


def attention_extraction():
    data_dir = "data/pod/docs" 
    meta_dir = "data/pod/meta_100"
    cache_dir = "data/pod/cache_100"
    ue_coeff = 0.0

    attention_model_list = [
        ("neat_cnn_full", "exp_ckpt/neat_cnn_full-12_18_19-16-42-15/", 199999),
    ]

    w2v_path = ""
    ue_path = OUTPUT_USER_EXPERTISE_META_PATH
    ue_size = 100
    num_styles = 17
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_docs = read_json(os.path.join(meta_dir, "train_test.json"))["test"]
    data_loader = NeatLoader(data_dir, meta_dir, cache_dir,
                             w2v_path, ue_path)

    for model_name, model_path, checkpoint in tqdm(attention_model_list, desc="Models"):
        model_path_full = os.path.join(model_path, "model_ckpt_{}.tar".format(checkpoint))
        input_model = load_model(model_name, ue_size, num_styles, device, data_loader, ue_coeff)
        loaded_state_dict = torch.load(model_path_full)["state_dict"]
        input_model.load_state_dict(loaded_state_dict)
        input_model.to(device)
        doc_attention = {}

        for k in range(5, 30, 2):
            for doc_file in tqdm(os.listdir(OUTPUT_DOC_DIR), desc="Inferring attention of {}".format(k)):
                doc_id = doc_file.replace(".json", "")
                if doc_id in test_docs:
                    thread_content = read_json(os.path.join(OUTPUT_DOC_DIR, doc_file))
                    predict_side_effects, user_creds, most_attended_word_posts = \
                        infer_fn(input_model, data_loader, thread_content, device, verbose=False, k=k)
                    doc_attention[doc_id] = [x for post in most_attended_word_posts for x in post]
            write_json(doc_attention, os.path.join(OUTPUT_ATTENTION_DIR, "{}_{}.json".format(model_name, k)))


def extract_user_rankings():
    if os.path.exists(OUTPUT_TRUE_CREDIBILITY_RANKING):
        print("Ranking exists in {}".format(OUTPUT_TRUE_CREDIBILITY_RANKING))
        return read_json(OUTPUT_TRUE_CREDIBILITY_RANKING)
    user_features = extract_user_features()
    all_docs = read_json("data/pod/meta/train_test.json")
    input_valid_docs = all_docs["train"] + all_docs["val"] + all_docs["test"]
    all_users = build_docs_users()
    thank_map = {user: features["n_thanks"] for user, features in user_features.items() if user in all_users
                 and features["n_thanks"] > 0}
    user_ranking_map = {}
    for file_name in tqdm(os.listdir(OUTPUT_DOC_DIR), "Building user ranking map"):
        doc_id = file_name.replace(".json", "")
        if doc_id in input_valid_docs:
            doc_data = read_json(os.path.join(OUTPUT_DOC_DIR, file_name))
            post_user_rankings = set()
            for post in doc_data.values():
                user = post["a"]
                if user in thank_map:
                    post_user_rankings.add((user, thank_map[user]))
            if len(post_user_rankings) > 1:
                post_user_rankings = sorted(list(post_user_rankings), key=lambda x: x[1])[::-1]
                user_ranking_map[doc_id] = post_user_rankings

    write_json(user_ranking_map, OUTPUT_TRUE_CREDIBILITY_RANKING)


def print_data_stats():
    train_test = read_json(os.path.join("data/pod/meta_100/train_test.json"))
    all_docs = train_test["train"][:100] + train_test["val"] + train_test["test"]
    drug_ses = read_json(OUTPUT_DRUG_SE_MAP_PATH)
    drugs = set(drug_ses.keys())
    excluded_words = drugs.union(set(EXCLUDED_WORDS))
    doc_ses = read_json(OUTPUT_DOC_SE_MAP_PATH)
    tokenizer = TweetTokenizer()
    all_post_len, all_thread_len, all_se_nums = [], [], []

    for doc_id in tqdm(all_docs, desc="Getting doc stats"):
        doc_file = os.path.join(OUTPUT_DOC_DIR, "{}.json".format(doc_id))
        doc_content = read_json(doc_file)
        for post_id, post_content in doc_content.items():
            post_text = [t for t in tokenizer.tokenize(post_content["t"].lower()) if t not in excluded_words]
            all_post_len.append(len(post_text))
        all_thread_len.append(len(doc_content))
        all_se_nums.append(len(doc_ses[doc_id]))

    user_participated_ses = read_json(OUTPUT_USER_PARTICIPATED_SE_MAP_PATH)
    user_se_counts = [len(v) for k, v in user_participated_ses.items()]
    standard_se_map = read_json(OUTPUT_STANDARD_SE_MAP_PATH)
    all_ses = set()
    users = load_text_as_list(os.path.join("data/pod/meta_100/users.txt"))
    for k, v in standard_se_map.items():
        all_ses.add(v)
    user_doc_map = read_json(OUTPUT_USER_DOC_MAP_PATH)
    user_doc_counts = [len(v) for k, v in user_doc_map.items()]

    print("Number of threads: {}".format(len(all_docs)))
    print("Number of users: {}".format(len(users)))
    print("Avg #words per post: {}".format(np.mean(all_post_len)))
    print("Avg #posts per thread: {}".format(np.mean(all_thread_len)))
    print("Avg #threads per user: {}".format(np.mean(user_doc_counts)))
    print("# Side effects: {}".format(standard_se_map))
    print("# Side effects per thread: {}".format(np.mean(all_se_nums)))


def export_data_pr_curve():
    data_dir = "data/pod/docs"
    meta_dir = "data/pod/meta"
    cache_dir = "data/pod/cache"
    true_ses_map = read_json(OUTPUT_DOC_SE_MAP_PATH)

    # model = ("neat_cnn", "exp_ckpt/neat_cnn-12_19_19-17-50-38", 599999)
    # model = ("neat_cnn_wpe", "exp_ckpt/neat_cnn_wpe-12_19_19-18-06-20", 1279999)
    # model = ("neat_cnn_wpeu", "exp_ckpt/neat_cnn_wpeu-12_21_19-13-28-25", 79999)
    model = ("neat_cnn_full", "exp_ckpt/neat_cnn_full-12_21_19-13-26-47", 259999)


    w2v_path = ""
    ue_path = os.path.join(meta_dir, "user_expertise.pickle")
    ue_size = 100
    num_styles = 17

    data_loader = NeatLoader(data_dir, meta_dir, cache_dir,
                             w2v_path, ue_path)

    model_name, model_path, checkpoint = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_docs = read_json("data/pod/meta/train_test.json")["val"]
    model_path_full = os.path.join(model_path, "model_ckpt_{}.tar".format(checkpoint))
    input_model = load_model(model_name, ue_size, num_styles, device, data_loader)  
    loaded_state_dict = torch.load(model_path_full)["state_dict"]
    input_model.load_state_dict(loaded_state_dict)

    input_model.to(device)
    input_model.eval()
    truth_buffer, pred_buffer = [], []
    for doc_file in tqdm(os.listdir(OUTPUT_DOC_DIR), desc="Export predictions"):
        doc_id = doc_file.replace(".json", "")
        if doc_id in val_docs and doc_id in true_ses_map:
            thread_content = read_json(os.path.join(OUTPUT_DOC_DIR, doc_file))
            for post_id, content in thread_content.items():
                content["a"] = np.random.choice(data_loader.users)

            posts = data_loader.neat_preprocess_single_thread(thread_content)
            dummy_label = np.zeros(len(data_loader.side_effects))
            thread_input_feature = NeatThreadInputFeature(posts, dummy_label)
            all_post_word_idxs, all_user_idxs, all_user_clusters, _ = data_loader.compile_thread_feature(
                thread_input_feature)

            x_post_word_idxs = all_post_word_idxs.to(device)
            x_user_idxs = all_user_idxs.to(device)
            x_user_clusters = all_user_clusters.to(device)

            outputs = input_model(x_post_word_idxs, x_user_idxs, x_user_clusters)
            logits = outputs[0]
            predict_prob = list(torch.sigmoid(logits).cpu().data.flatten())

            true_ses = true_ses_map[doc_id]
            binary_true_ses = [0 if se not in true_ses else 1 for se in data_loader.side_effects]

            truth_buffer.append(binary_true_ses)
            pred_buffer.append(predict_prob)
    save_to_pickle((truth_buffer, pred_buffer), "{}_pr_auc.pickle".format(model_name))


def plot_pr_auc():
    pass


if __name__ == "__main__":
    # analyse_clusters("output/clustering/fold/participated_filtered/cluster_7_results.txt")
    # extract_user_features()
    # extract_user_precision()
    # output_post_corr, output_question_corr, output_reply_corr, output_thank_corr, \
    #     output_precision_corr = credibility_correlation()
    # print("Correlation between credibility and # post", output_post_corr)
    # print("Correlation between credibility and # questions", output_question_corr)
    # print("Correlation between credibility and # replies", output_reply_corr)
    # print("Correlation between credibility and # thanks", output_thank_corr)
    # print("Correlation between credibility and precision", output_precision_corr)
    # search_best_precision_correlations()
    # test_all_models()
    # precision_thank_correlation()
    # attention_extraction()
    # post_thank_corr()
    # extract_user_rankings()
    # attention_extraction()
    # print_data_stats()
    export_data_pr_curve()
