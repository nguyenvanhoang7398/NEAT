import time
from constants import *
from utils import *
from tqdm import tqdm
from dataset import NeatLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import IncrementalPCA
from train import evaluate, load_model
from infer import infer_fn
from pprint import pprint
from side_effects import parse_raw_side_effects_simple, lemmatize
from analysis import extract_user_rankings, extract_user_features
from model.utils import credibility_ranking
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier


def compile_adr_results(standard_se_map, credible_users=None, output_path=None):
    output_path = output_path if output_path is not None else OUTPUT_ADR_RESULT_PATH
    if os.path.exists(output_path):
        print("Adr results exist in {}".format(output_path))
        return read_json(output_path)
    adr_doc_se_map = {}
    for batch_file in tqdm(os.listdir(OUTPUT_SE_DOC_DIR), desc="Compiling ADR results"):
        se_doc_content = read_csv(os.path.join(OUTPUT_SE_DOC_DIR, batch_file), True, "\t")
        for row in se_doc_content:
            author_id, doc_id, raw_ses = row[1], row[2], row[3][1:-1]     # eliminate brackets
            if credible_users is None or author_id in credible_users:
                tokens = [x for x in raw_ses.split(",") if len(x) > 1]
                parsed_ses = [se.replace("\'", "").strip().rstrip() for se in tokens]
                standardized_ses = set([standard_se_map[se] for se in parsed_ses])
                if doc_id not in adr_doc_se_map:
                    adr_doc_se_map[doc_id] = set()
                adr_doc_se_map[doc_id] = adr_doc_se_map[doc_id].union(standardized_ses)
    adr_doc_se_map = {k: list(v) for k, v in adr_doc_se_map.items()}
    write_json(adr_doc_se_map, output_path)
    return adr_doc_se_map


def compile_umls_results(standard_se_map, credible_users=None, output_path=None):
    output_path = output_path if output_path is not None else OUTPUT_UMLS_RESULT_PATH
    if os.path.exists(output_path):
        print("UMLS results exist in {}".format(output_path))
        return read_json(output_path)
    umls_doc_se_map = {}
    for doc_file in tqdm(os.listdir(OUTPUT_UMLS_DOC_DIR), desc="Compiling UMLS results"):
        doc_id = doc_file.replace(".json", "")
        umls_doc_content = read_json(os.path.join(OUTPUT_UMLS_DOC_DIR, doc_file))
        doc_ses = []
        for user, ses in umls_doc_content.items():
            if credible_users is None or user in credible_users:
                doc_ses.extend(ses)
        standardized_ses = set([standard_se_map[se] for se in doc_ses])
        umls_doc_se_map[doc_id] = list(standardized_ses)
    write_json(umls_doc_se_map, output_path)
    return umls_doc_se_map


def benchmark(pred_json_file, num_labels=315, name="", valid_docs=None):
    benchmark_output_path = os.path.join(OUTPUT_BENCHMARK_DIR, "{}_results.json".format(name))
    if os.path.exists(benchmark_output_path):
        print("Benchmark result exists in {}".format(benchmark_output_path))
        return read_json(benchmark_output_path)

    standard_se_map = read_json(OUTPUT_STANDARD_SE_MAP_PATH)
    all_ses = sorted(list(set(standard_se_map.values())))
    se2idx = {se: i for i, se in enumerate(all_ses)}
    truth_json_file = OUTPUT_DOC_SE_MAP_PATH
    truth_content = read_json(truth_json_file)
    pred_content = read_json(pred_json_file)
    truth_buffer, pred_buffer = [], []
    miss, total = 0, 0
    for doc_id, truth in tqdm(truth_content.items(), desc="Benchmarking {}".format(name)):
        if valid_docs is None or doc_id in valid_docs:
            if doc_id not in pred_content:
                miss += 1
            elif len(pred_content[doc_id]) > 0:
                pred_buffer.append([se2idx[x] for x in pred_content[doc_id]] if doc_id in pred_content else [])
                truth_buffer.append([se2idx[x] for x in truth])
            total += 1
    miss_ratio = float(miss) / total
    benchmark_result = evaluate(truth_buffer, pred_buffer, num_labels, vector=False)
    benchmark_result["miss_ratio"] = miss_ratio
    write_json(benchmark_result, benchmark_output_path)
    return benchmark_result


def build_bow_dataset(mode):
    if os.path.exists(OUTPUT_BOW_FEATURE.format(mode)) \
            and os.path.exists(OUTPUT_BOW_LABEL.format(mode)) \
            and os.path.exists(OUTPUT_BOW_DOCS.format(mode)):
        print("Bow dataset exists")
        return load_from_pickle(OUTPUT_BOW_FEATURE.format(mode)), \
            load_from_pickle(OUTPUT_BOW_LABEL.format(mode)), \
            load_text_as_list(OUTPUT_BOW_DOCS.format(mode))

    data_loader = NeatLoader(OUTPUT_DOC_DIR, OUTPUT_META_DIR, OUTPUT_CACHE_DIR)
    data_loader.set_mode(mode)
    doc_names = data_loader.get_doc_names()
    data_paths = [os.path.join(data_loader.cache_dir, "{}.cache".format(doc_name)) for doc_name in doc_names]
    vocab_size = len(data_loader.vocab)

    idf_vector = np.zeros(vocab_size)
    total_posts = 0

    doc_list = []
    x_data, y_data = [], []
    for data_path in tqdm(data_paths, desc="Building BOW dataset"):
        doc_id = data_path.replace(".cache", "")
        doc_list.append(doc_id)
        thread_input_feature = torch.load(data_path)
        all_post_word_idxs_array = [p.post_word_idxs for p in thread_input_feature.posts]
        thread_multi_hot_label = thread_input_feature.label
        tf_vector = np.zeros(vocab_size)
        for post_word_idxs in all_post_word_idxs_array:
            for word_idx in set(post_word_idxs):
                idf_vector[word_idx] += 1
            for word_idx in post_word_idxs:
                if word_idx != 0:
                    tf_vector[word_idx] += 1
            total_posts += 1
        x_data.append(tf_vector)
        y_data.append(thread_multi_hot_label)

    idf_vector = np.log10(total_posts) / idf_vector
    idf_vector[np.isinf(idf_vector)] = 0
    x_data = [np.multiply(doc_vector, idf_vector) for doc_vector in x_data]

    save_to_pickle(x_data, OUTPUT_BOW_FEATURE.format(mode))
    save_to_pickle(y_data, OUTPUT_BOW_LABEL.format(mode))
    save_list_as_text(doc_list, OUTPUT_BOW_DOCS.format(mode))

    return x_data, y_data, doc_list


def preprocessing_bow():
    train_bow_feature, train_bow_label, _ = build_bow_dataset("train")
    train_bow_label = np.vstack(train_bow_label).astype(np.int)
    train_bow_feature = train_bow_feature[:60000]
    train_bow_label = train_bow_label[:60000]
    print("Fit scaler")
    scaler = MinMaxScaler()
    scaler_batch_size = 1024
    i = 0
    while i < len(train_bow_feature):
        scaler.partial_fit(train_bow_feature[i: i + scaler_batch_size])
        i += scaler_batch_size
    scaled_train_bow_feature = []
    i = 0
    while i < len(train_bow_feature):
        scaled_train_bow_feature_batch = scaler.transform(train_bow_feature[i: i + scaler_batch_size])
        scaled_train_bow_feature.extend(scaled_train_bow_feature_batch)
        i += scaler_batch_size
    scaled_train_bow_feature = np.vstack(scaled_train_bow_feature)
    print("Fit pca")
    num_components = 300
    pca = IncrementalPCA(n_components=num_components)
    pca_batch_size = 5000
    i = 0
    while i < len(scaled_train_bow_feature):
        pca_batch_size = 2 * pca_batch_size if i + 2 * pca_batch_size >= len(
            scaled_train_bow_feature) else pca_batch_size
        pca.partial_fit(scaled_train_bow_feature[i: i + pca_batch_size])
        i += pca_batch_size
    reduced_train_bow_feature = []
    i = 0
    while i < len(scaled_train_bow_feature):
        reduced_train_bow_feature_batch = pca.transform(scaled_train_bow_feature[i: i + pca_batch_size])
        reduced_train_bow_feature.extend(reduced_train_bow_feature_batch)
        i += pca_batch_size
    print("Num pca components {}".format(pca.n_components_))
    reduced_train_bow_feature = np.vstack(reduced_train_bow_feature)

    save_to_pickle(scaler, OUTPUT_BOW_SCALER)
    save_to_pickle(pca, OUTPUT_BOW_PCA)
    save_to_pickle(reduced_train_bow_feature, OUTPUT_BOW_TRAIN_X)
    save_to_pickle(train_bow_label, OUTPUT_BOW_TRAIN_Y)

    return reduced_train_bow_feature, train_bow_label


def train_svm():
    print("Fitting svm")
    reduced_train_bow_feature = load_from_pickle(OUTPUT_BOW_TRAIN_X)
    train_bow_label = load_from_pickle(OUTPUT_BOW_TRAIN_Y)
    one_versus_rest_svm = OneVsRestClassifier(SVC(probability=True))
    one_versus_rest_svm.fit(X=reduced_train_bow_feature, y=train_bow_label)
    save_to_pickle(one_versus_rest_svm, OUTPUT_BOW_SVM)


def train_rf():
    if os.path.exists(OUTPUT_BOW_RF):
        return load_from_pickle(OUTPUT_BOW_RF)
    print("Fitting rf")
    reduced_train_bow_feature = load_from_pickle(OUTPUT_BOW_TRAIN_X)
    train_bow_label = load_from_pickle(OUTPUT_BOW_TRAIN_Y)
    one_versus_rest_rf = OneVsRestClassifier(RandomForestClassifier(verbose=2))
    one_versus_rest_rf.fit(X=reduced_train_bow_feature, y=train_bow_label)
    save_to_pickle(one_versus_rest_rf, OUTPUT_BOW_RF)
    return one_versus_rest_rf


def benchmark_bow(model):
    scaler = load_from_pickle(OUTPUT_BOW_SCALER)
    pca = load_from_pickle(OUTPUT_BOW_PCA)

    val_bow_feature, val_bow_label, _ = build_bow_dataset("val")
    val_bow_feature = scaler.transform(val_bow_feature)
    val_bow_feature = pca.transform(val_bow_feature)
    val_predictions = model.predict_proba(val_bow_feature)
    val_results  = evaluate(val_bow_label, val_predictions, None)
    print("Validation result")
    write_json(val_results, "val_bow.json")

    test_bow_feature, test_bow_label, _ = build_bow_dataset("test")
    test_bow_feature = scaler.transform(test_bow_feature)
    test_bow_feature = pca.transform(test_bow_feature)
    test_predictions = model.predict_proba(test_bow_feature)
    test_results = evaluate(test_bow_label, test_predictions, None)
    print("Testing result")
    write_json(test_results, "test_bow.json")


def benchmark_trained_models():
    models = ["neat_cnn", "neat_cnn_full", "neat_cnn_wpeu", "neat_cnn_wpe"]
    doc_ses = read_json(OUTPUT_DOC_SE_MAP_PATH)
    standard_se_map = read_json(OUTPUT_STANDARD_SE_MAP_PATH)
    se_list = sorted(list(set(standard_se_map.values())))
    se2idx = {se: i for i, se in enumerate(se_list)}
    for model in tqdm(models, desc="Models"):
        model_prediction_dir = os.path.join(OUTPUT_BENCHMARK_DIR, model)
        total_f1, total_precision, total_recall, total_accuracy = [], [], [], []
        for doc_file in tqdm(os.listdir(model_prediction_dir), desc="Docs"):
            predictions = read_json(os.path.join(model_prediction_dir, doc_file))
            predicted_ses = predictions["predictions"]
            true_ses = doc_ses[doc_file.replace(".json", "")]
            predicted_ses = [se2idx[se] for se in predicted_ses]
            true_ses = [se2idx[se] for se in true_ses]
            validation_results = evaluate(true_ses, predicted_ses, 315, vector=False)
            total_f1.append(validation_results["f1"])
            total_precision.append(validation_results["precision"])
            total_recall.append(validation_results["recall"])
            total_accuracy.append(validation_results["accuracy"])
        write_json({
            "f1": np.mean(total_f1),
            "precision": np.mean(total_precision),
            "recall": np.mean(total_recall),
            "accuracy": np.mean(total_accuracy)
        }, os.path.join(OUTPUT_BENCHMARK_DIR, "{}.json".format(model)))


def benchmark_adr_extraction_with_credibility():
    user_list = load_text_as_list(OUTPUT_USER_CREDIBILITY_LIST_PATH)
    credibility_list = load_text_as_list(OUTPUT_USER_CREDIBILITY_SCORE_PATH)
    standard_se_map = read_json(OUTPUT_STANDARD_SE_MAP_PATH)
    threshold = 1

    credible_users = [user for user, credibility in zip(user_list, credibility_list) if float(credibility) > threshold]
    output_adr_path = os.path.join("output/adr/adr.json")
    output_umls_path = os.path.join("output/adr/umls.json")
    compile_adr_results(standard_se_map, credible_users, output_path=output_adr_path)
    compile_umls_results(standard_se_map, credible_users, output_path=output_umls_path)
    benchmark(output_umls_path, name="umls_cred")
    benchmark(output_adr_path, name="adr_cred")


def benchmark_attention(valid_docs=None):
    standard_se_map = read_json(OUTPUT_STANDARD_SE_MAP_PATH)

    for model_name_file in os.listdir(OUTPUT_ATTENTION_DIR):
        attn_doc_ses = {}
        model_name = model_name_file.replace(".json", "")
        if "extracted" in model_name:
            continue
        output_path = os.path.join(OUTPUT_ATTENTION_DIR, "{}_extracted.json".format(model_name))
        if os.path.exists(output_path):
            print("{}'s extracted side effects exist in {}".format(model_name, output_path))
        else:
            attn_predictions = read_json(os.path.join(OUTPUT_ATTENTION_DIR, model_name_file))
            for doc_id, ses in tqdm(attn_predictions.items(), desc="Benchmarking Attention for {}".format(model_name_file)):
                lemmatized_ses = [lemmatize(x) for x in ses]
                parsed_ses = parse_raw_side_effects_simple(lemmatized_ses, list(standard_se_map.keys()))
                standardized_ses = [standard_se_map[se] for se in parsed_ses]
                attn_doc_ses[doc_id] = list(set(standardized_ses))
            write_json(attn_doc_ses, output_path)
        benchmark(output_path, valid_docs=valid_docs, name="{}_attention".format(model_name))


def benchmark_extraction():
    train_test_docs = read_json(os.path.join("data/pod/meta_100/train_test.json"))
    input_valid_docs = train_test_docs["test"]
    input_umls_pred_json_file = OUTPUT_UMLS_RESULT_PATH
    input_adr_pred_json_file = OUTPUT_ADR_RESULT_PATH
    benchmark(input_umls_pred_json_file, valid_docs=input_valid_docs, name="umls")
    benchmark(input_adr_pred_json_file, valid_docs=input_valid_docs, name="adr")
    benchmark_attention(input_valid_docs)


def benchmark_user_ranking():
    user_ranking_content = extract_user_rankings()
    train_test = read_json("data/train_test/ibuprofen_levothyroxine.json")
    valid_docs = set(train_test["train"])
    user_rankings = []
    for doc_id, post_user_ranking in user_ranking_content.items():
        user_rankings.append([x for x in post_user_ranking])
    user_features = extract_user_features()
    user_list = load_text_as_list(OUTPUT_USER_PATH)
    user_docs = read_json(OUTPUT_USER_DOC_MAP_PATH)
    user_n_docs = {u: len([d for d in docs if d in valid_docs]) for u, docs in user_docs.items()}

    num_posts, num_questions, num_random = [], [], []
    for user in user_list:
        num_posts.append(user_n_docs[user] if user in user_n_docs else 0)
        num_questions.append((user_features[user]["n_questions"])
                             if user in user_features else 0)
        num_random.append(np.random.random())

    post_ndcg_scores, post_spearman_score = credibility_ranking(user_rankings, user_list, num_posts)
    question_ndcg_scores, question_spearman_score = credibility_ranking(user_rankings, user_list, num_questions)
    random_ndcg_scores, random_spearman_score = credibility_ranking(user_rankings, user_list, num_random)
    print("Random ndcg: {}, spearman: {}".format(random_ndcg_scores, random_spearman_score))
    print("Post ndcg: {}, spearman: {}".format(post_ndcg_scores, post_spearman_score))
    print("Question ndcg: {}, spearman: {}".format(question_ndcg_scores, question_spearman_score))


def benchmark_post_thank_corr():
    user_features = extract_user_features()
    num_thanks_p, num_posts = [], []
    num_thanks_q, num_questions = [], []
    num_thanks_rand, num_rand = [], []
    for user, features in user_features.items():
        if "n_thanks" and "n_posts" in features:
            num_thanks_p.append(features["n_thanks"])
            num_posts.append(features["n_posts"])
        if "n_thanks" and "n_questions" in features:
            num_thanks_q.append(features["n_thanks"])
            num_questions.append(features["n_questions"])
        if "n_thanks" in features:
            num_rand.append(np.random.random())
            num_thanks_rand.append(features["n_thanks"])

    post_corr = spearmanr(num_thanks_p, num_posts)
    question_corr = spearmanr(num_thanks_q, num_questions)
    random_corr = spearmanr(num_thanks_rand, num_rand)
    print("Post-thank corr {}".format(post_corr))
    print("Question-thank corr {}".format(question_corr))
    print("Random-thank corr {}".format(random_corr))


def run_bow():
    build_bow_dataset("train")
    build_bow_dataset("val")
    build_bow_dataset("test")
    rf_model = train_rf()
    print("Benchmark rf")
    # preprocessing_bow()
    benchmark_bow(rf_model)


def benchmark_random_cred():
    data_dir = "data/pod/docs"
    meta_dir = "data/pod/meta_100"
    cache_dir = "data/pod/cache_100"
    true_ses = read_json(OUTPUT_DOC_SE_MAP_PATH)

    # model = ("neat_cnn_full", "exp_ckpt/neat_cnn_full-12_18_19-17-45-47/", 3319999)
    # model = ("neat_cnn_full", "exp_ckpt/neat_cnn_full-12_18_19-16-42-15/", 199999)
    # model = ("neat_cnn_full", "exp_ckpt/neat_cnn_full-12_18_19-16-50-55/", 159999) 
    model = ("neat_cnn_full", "exp_ckpt/neat_cnn_full-12_18_19-17-48-19/", 639999)
    # model = ("neat_cnn_full", "exp_ckpt/neat_cnn_full-12_18_19-17-48-10/", 3879999)

    w2v_path = ""
    ue_path = os.path.join(meta_dir, "user_expertise.pickle")
    ue_size = 100
    num_styles = 17

    data_loader = NeatLoader(data_dir, meta_dir, cache_dir,
                             w2v_path, ue_path)

    model_name, model_path, checkpoint = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_docs = read_json("data/pod/meta_100/train_test.json")["val"]
    model_path_full = os.path.join(model_path, "model_ckpt_{}.tar".format(checkpoint))
    input_model = load_model(model_name, ue_size, num_styles, device, data_loader)
    loaded_state_dict = torch.load(model_path_full)["state_dict"]
    input_model.load_state_dict(loaded_state_dict)
    
    user_cred = input_model.user_credibility.weight.detach().numpy().flatten()
    unk_cred = user_cred[0]
    print(user_cred)
    # np.random.seed(999)
    # np.random.shuffle(user_cred)
    # user_cred = np.random.normal(size=len(data_loader.users))
    #user_cred[0] = unk_cred
    user_cred = user_cred.reshape(-1, 1)
    print(user_cred)
    input_model.user_credibility.weight.data.copy_(torch.from_numpy(user_cred))

    input_model.to(device)
    input_model.eval()
    truth_buffer, pred_buffer = [], []
    for doc_file in tqdm(os.listdir(OUTPUT_DOC_DIR), desc="Val docs"):
        doc_id = doc_file.replace(".json", "")
        if doc_id in val_docs:
            thread_content = read_json(os.path.join(OUTPUT_DOC_DIR, doc_file))
            for post_id, content in thread_content.items():
                content["a"] = np.random.choice(data_loader.users)
            predict_side_effects, user_creds, most_attended_word_posts = \
                infer_fn(input_model, data_loader, thread_content, device, verbose=False)
           
            truth_buffer.append(list(true_ses[doc_id]))
            pred_buffer.append(list(predict_side_effects))
    eval_results = evaluate(truth_buffer, pred_buffer, len(data_loader.side_effects), vector=False)
    write_json(eval_results, "drug_discovery_{}.json".format(time.time()))


if __name__ == "__main__":
    # compile_adr_results(input_standard_se_map)
    # compile_umls_results(input_standard_se_map)
    # benchmark_extraction()
    # benchmark_post_thank_corr()
    # benchmark_user_ranking()
    # run_bow()
    # benchmark_trained_models()
    # benchmark_adr_extraction_with_credibility()
    # benchmark_post_thank_corr()
    benchmark_random_cred()
