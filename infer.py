import argparse
from constants import *
import torch
from train import load_model
from dataset import NeatLoader, NeatThreadInputFeature
from utils import *
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=OUTPUT_DOC_DIR, type=str, help="Input data dir.")
    parser.add_argument("--cache_dir", default=OUTPUT_CACHE_DIR, type=str, help="Input meta data dir.")
    parser.add_argument("--meta_dir", default=OUTPUT_META_DIR, type=str, help="Input cache dir.")
    parser.add_argument("--w2v_path", default="data/glove.6B.300d.txt", type=str, help="Input word embedding path")
    parser.add_argument("--ue_path", default=OUTPUT_USER_EXPERTISE_META_PATH, type=str, help="Input user expertise")
    parser.add_argument("--ue_size", default=40, type=int, help="User expertise dim")

    parser.add_argument("--model_path", default="output/inference/model.tar", type=str, help="Model path")
    parser.add_argument("--model_name", default="neat_cnn_full", type=str, help="Model name")
    parser.add_argument("--thread_path", default="output/inference/thread.json", type=str,
                        help="Path to thread for inference")

    return parser.parse_args()


def infer_fn(model, data_loader: NeatLoader, thread_content, device, verbose=True, k=5):
    posts = data_loader.neat_preprocess_single_thread(thread_content)
    dummy_label = np.zeros(len(data_loader.side_effects))
    thread_input_feature = NeatThreadInputFeature(posts, dummy_label)
    all_post_word_idxs, all_user_idxs, all_user_clusters, _ = data_loader.compile_thread_feature(thread_input_feature)

    x_post_word_idxs = all_post_word_idxs.to(device)
    x_user_idxs = all_user_idxs.to(device)
    x_user_clusters = all_user_clusters.to(device)

    outputs = model(x_post_word_idxs, x_user_idxs, x_user_clusters)
    logits = outputs[0]
    predict_prob = torch.sigmoid(logits).cpu().data[0]
    predict_labels = np.argwhere(predict_prob > 0.5).flatten()
    predict_side_effects = sorted([data_loader.side_effects[se] for se in predict_labels])
    if verbose:
        print("Predicted side effects: {}".format(predict_side_effects))

    user_creds = []
    if len(outputs) > 1:
        user_credibility = outputs[1].detach().cpu().numpy().squeeze(-1)
        users = [data_loader.users[user_idx] for user_idx in all_user_idxs]
        user_creds = list(set(zip(users, user_credibility)))
        if verbose:
            print("User credibility scores: {}".format(user_creds))

    most_attended_word_posts = []
    if len(outputs) > 2:
        raw_post_attentions = outputs[2]
        compiled_attentions = model.compile_attentions(raw_post_attentions, data_loader.max_post_len)
        for post_id, post_attn in enumerate(compiled_attentions):
            most_attended_pos = np.argsort(post_attn)[::-1][:k]
            post_word_idxs = x_post_word_idxs[post_id]
            most_attended_word_idxs = [post_word_idxs[i] for i in most_attended_pos]
            most_attended_words = [data_loader.vocab[i] for i in most_attended_word_idxs]
            most_attended_word_posts.append(most_attended_words)
        if verbose:
            print(most_attended_word_posts)

    return predict_side_effects, user_creds, most_attended_word_posts


if __name__ == "__main__":
    program_args = parse_args()
    input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    program_args.device = input_device

    input_data_loader = NeatLoader(program_args.data_dir, program_args.meta_dir, program_args.cache_dir,
                                   program_args.w2v_path, program_args.ue_path)
    input_model = load_model(program_args.model_name, program_args.ue_size, input_data_loader)

    loaded_state_dict = torch.load(program_args.model_path)["state_dict"]
    input_model.load_state_dict(loaded_state_dict)
    input_model.to(program_args.device)
    input_model.eval()
    input_infer_thread = read_json(program_args.thread_path)
    infer_fn(input_model, input_data_loader, input_infer_thread, program_args)
