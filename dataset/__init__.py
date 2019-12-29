from sklearn.model_selection import train_test_split
import dataset.utils as dataset_utils
from utils import *
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
from constants import *
import numpy as np
import torch
from collections import defaultdict


class NeatPostInputFeature(object):
    def __init__(self, post_word_idxs, user_idx):
        self.post_word_idxs = post_word_idxs
        self.user_idx = user_idx


class NeatThreadInputFeature(object):
    def __init__(self, posts, label):
        self.posts = posts
        self.label = label


class NeatLoader(object):
    UNK_TOK_IDX = 1
    PAD_TOK_IDX = 0
    UNK_USER_IDX = 0
    UNK_TOK = "<UNK>"
    PAD_TOK = "<PAD>"
    UNK_USER = "<UNK_USER>"

    def __init__(self, data_dir, meta_dir, cache_dir, size, w2v_path="", user_expertise_path="", user_style_path="",
                 max_post_len=200):
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.cache_dir = cache_dir
        self.max_post_len = max_post_len
        self.mode = "train"
        self.size = size
        self.vocab, self.vocab2idx = self.build_vocab()
        self.users, self.user2idx = self.build_users()
        self.doc_se_map, self.side_effects, self.se2idx = self.build_ses()
        self.user2cluster, self.clusters = self.build_user_clusters()
        self.word_embeddings = self.build_word_embedding(w2v_path)
        self.user_expertise = self.build_user_expertise(user_expertise_path)
        self.user_styles = self.build_user_styles(user_style_path)
        self.user_precision_adr, self.user_precision_umls, self.user_features = self.build_thanks_precision()
        self.thank_users, self.thank_scores = self.build_thank_users()
        self.user_rankings = self.build_user_ranking()
        self.drug_list = self.build_drug_list()
        self.tokenizer = TweetTokenizer()
        self.excluded_words = self.drug_list.union(set(EXCLUDED_WORDS))
        self.train_docs, self.val_docs, self.test_docs = self.train_val_test_split()
        self.all_docs = set(self.train_docs + self.val_docs + self.test_docs)
        self.neat_preprocess()

    def build_user_ranking(self):
        user_ranking_path = os.path.join(self.meta_dir, "user_ranking.json")
        user_ranking_content = read_json(user_ranking_path)
        user_rankings = []
        for doc_id, post_user_ranking in user_ranking_content.items():
            user_rankings.append([x for x in post_user_ranking])
        return user_rankings

    def build_thanks_precision(self):
        user_precision_adr = read_json(os.path.join(self.meta_dir, "user_precision_adr.json"))
        user_precision_umls = read_json(os.path.join(self.meta_dir, "user_precision_umls.json"))
        user_features = read_json(os.path.join(self.meta_dir, "user_features.json"))
        return user_precision_adr, user_precision_umls, user_features

    def build_word_embedding(self, w2v_path):
        embed_size = 50
        cache_embedding_path = os.path.join(self.cache_dir, "cache_word_embedding_{}.pickle".format(embed_size))
        if os.path.exists(cache_embedding_path):
            return load_from_pickle(cache_embedding_path)
        word_embedding = dataset_utils.init_word_embedding(self.vocab, w2v_path, embed_size)
        save_to_pickle(word_embedding, cache_embedding_path)
        return word_embedding

    def build_user_expertise(self, user_expertise_path):
        return dataset_utils.init_user_expertise(self.users, user_expertise_path, 100)

    def build_user_styles(self, user_style_path):
        return dataset_utils.init_user_styles(self.users, user_style_path, 17)

    def build_vocab(self):
        vocab_path = os.path.join(self.meta_dir, "vocab.txt")
        vocab = [NeatLoader.PAD_TOK, NeatLoader.UNK_TOK]
        vocab.extend(load_text_as_list(vocab_path))
        vocab2idx = {word: i for i, word in enumerate(vocab)}
        return vocab, vocab2idx

    def build_users(self):
        user_path = os.path.join(self.meta_dir, "users.txt")
        users = [NeatLoader.UNK_USER]
        users.extend(load_text_as_list(user_path))
        user2idx = {user: i for i, user in enumerate(users)}
        return users, user2idx

    def build_drug_list(self):
        drug_list = sorted(list(read_json(os.path.join(self.meta_dir, "drug_ses.json")).keys()))
        return set(drug_list)

    def build_thank_users(self):
        delimiter = " "
        output_thank_user_path = os.path.join(self.meta_dir, "thank_users.txt")
        if os.path.exists(output_thank_user_path):
            thank_user_data = load_text_as_list(output_thank_user_path)
            thank_users, thank_scores = [], []
            for row in thank_user_data:
                tokens = row.split(delimiter)
                user = tokens[0]
                if user in self.users:
                    thank_users.append(self.user2idx[tokens[0]])
                    thank_scores.append(float(tokens[1]))
        else:
            semi_credibility_percentage = 0.2
            thank_user_data = [(user, float(features["n_thanks"])) for user, features in self.user_features.items()
                               if features["n_thanks"] > 0]
            # normalize thank user
            thank_user_norm = np.linalg.norm([x[1] for x in thank_user_data])
            thank_user_data = [(x[0], x[1] / thank_user_norm) for x in thank_user_data]

            # choose a subset of user to regularize correlation
            num_choice = int(len(thank_user_data) * semi_credibility_percentage)
            chosen_idx = np.random.choice(range(len(thank_user_data)), size=num_choice, replace=False)
            chosen_thank_user_data = [thank_user_data[i] for i in chosen_idx]

            thank_users, thank_scores = [], []
            for x in chosen_thank_user_data:
                if x[0] in self.users:
                    thank_users.append(self.user2idx[x[0]])
                    thank_scores.append(x[1])
            save_list_as_text([delimiter.join([str(y) for y in x]) for x in chosen_thank_user_data],
                              output_thank_user_path)
        thank_users = torch.LongTensor(thank_users)
        thank_scores = torch.FloatTensor(thank_scores)
        return thank_users, thank_scores

    def build_user_clusters(self):
        user_cluster_path = os.path.join(self.meta_dir, "user_clusters.txt")
        user_cluster_data = load_text_as_list(user_cluster_path)
        user_cluster_data = [user_cluster.split(" ") for user_cluster in user_cluster_data]
        clusters, user2cluster = {0}, {}
        for user_cluster in user_cluster_data:
            cluster = int(user_cluster[1]) + 1
            user2cluster[user_cluster[0]] = cluster
            clusters.add(cluster)
        user2cluster = defaultdict(lambda: 0, user2cluster)
        return user2cluster, sorted(list(clusters))

    def build_ses(self):
        doc_se_map = read_json(os.path.join(self.meta_dir, "doc_ses.json"))
        all_ses = []
        for doc, ses in doc_se_map.items():
            all_ses.extend(ses)
        side_effects = sorted(list(set(all_ses)))
        se2idx = {se: i for i, se in enumerate(side_effects)}
        return doc_se_map, side_effects, se2idx

    def neat_preprocess_single_post(self, post, user):
        post_tokens = [t for t in self.tokenizer.tokenize(post.lower()) if t not in self.excluded_words]
        post_idxs = [self.vocab2idx[t] if t in self.vocab else NeatLoader.UNK_TOK_IDX for t in post_tokens]
        post_idxs = post_idxs[:self.max_post_len]
        padding_size = self.max_post_len - len(post_idxs)
        if padding_size > 0:
            post_idxs.extend([NeatLoader.PAD_TOK_IDX] * padding_size)
        user_idx = self.user2idx[user] if user in self.users else NeatLoader.UNK_USER_IDX
        return post_idxs, user_idx

    def neat_preprocess_single_thread(self, doc_content):
        posts = []
        for post_num, post_content in doc_content.items():
            post_text, post_author = post_content["t"], post_content["a"]
            post_idxs, user_idx = self.neat_preprocess_single_post(post_text, post_author)
            posts.append(NeatPostInputFeature(
                post_word_idxs=post_idxs,
                user_idx=user_idx
            ))
        return posts

    def neat_preprocess(self):
        for doc_file in tqdm(os.listdir(self.data_dir)):
            doc_id = doc_file[:-5]
            if doc_id not in self.all_docs:
                continue
            file_path = os.path.join(self.data_dir, doc_file)
            cache_path = os.path.join(self.cache_dir, doc_file.replace(".json", ".cache"))
            if os.path.exists(cache_path):
                continue
            doc_content = read_json(file_path)
            posts = self.neat_preprocess_single_thread(doc_content)
            true_ses = self.doc_se_map[doc_id]
            multi_hot_label_vector = np.zeros(len(self.side_effects))
            for se in true_ses:
                multi_hot_label_vector[self.se2idx[se]] = 1
            thread_input_feature = NeatThreadInputFeature(posts, multi_hot_label_vector)
            torch.save(thread_input_feature, cache_path)

    def __len__(self):
        return len(self.get_doc_names())

    def train_val_test_split(self):
        train_test_path = os.path.join(self.meta_dir, "train_test.json")
        print("Load train test split from {}".format(train_test_path))
        train_test_docs = read_json(train_test_path)
        train_docs = [d for d in train_test_docs["train"] if d in self.doc_se_map]
        val_docs = [d for d in train_test_docs["val"] if d in self.doc_se_map]
        test_docs = [d for d in train_test_docs["test"] if d in self.doc_se_map]
        print("Num train {}".format(len(train_docs)))
        print("Num val {}".format(len(val_docs)))
        print("Num test {}".format(len(test_docs)))
        return train_docs, val_docs, test_docs

    def get_all_docs_names(self):
        return [file_name.replace(".cache", "") for file_name in os.listdir(self.cache_dir)
                if file_name.endswith(".cache")]

    def get_doc_names(self):
        if self.mode == "train":
            train_size = int(len(self.train_docs) * self.size)
            return self.train_docs[:train_size]
        elif self.mode == "val":
            val_size = int(len(self.val_docs) * self.size)
            return self.val_docs[:val_size]
        elif self.mode == "test":
            test_size = int(len(self.test_docs) * self.size)
            return self.test_docs[:test_size]
        else:
            raise ValueError("Unrecognized loader mode {}".format(self.mode))

    def set_mode(self, new_mode):
        if new_mode in ["train", "val", "test"]:
            self.mode = new_mode
            print("Mode change to {}".format(self.mode))
        else:
            raise ValueError("Unrecognized loader mode {}".format(new_mode))

    def compile_thread_feature(self, thread_input_feature):
        all_post_word_idxs_array = [p.post_word_idxs for p in thread_input_feature.posts]
        all_post_word_idxs = torch.tensor(all_post_word_idxs_array, dtype=torch.long)

        all_user_idxs_array = [p.user_idx for p in thread_input_feature.posts]
        all_user_idxs = torch.tensor(all_user_idxs_array, dtype=torch.long)

        all_user_clusters_array = [self.user2cluster[user_idx] for user_idx in all_user_idxs_array]
        all_user_clusters = torch.tensor(all_user_clusters_array, dtype=torch.long)

        thread_multi_hot_label = torch.tensor(thread_input_feature.label, dtype=torch.long)
        return all_post_word_idxs, all_user_idxs, all_user_clusters, thread_multi_hot_label

    def __iter__(self):
        doc_names = self.get_doc_names()
        data_paths = [os.path.join(self.cache_dir, "{}.cache".format(doc_name)) for doc_name in doc_names]

        for data_path in data_paths:
            thread_input_feature = torch.load(data_path)
            all_post_word_idxs, all_user_idxs, all_user_clusters, thread_multi_hot_label = self.compile_thread_feature(thread_input_feature)
            yield all_post_word_idxs, all_user_idxs, all_user_clusters, thread_multi_hot_label
