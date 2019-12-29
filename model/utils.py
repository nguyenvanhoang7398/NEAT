from scipy.stats import spearmanr
from model.metrics import ndcg_at_k
import numpy as np


def credibility_ranking(user_rankings, user_list, credibility_list):
    user_credibility_map = {user: cred for user, cred in zip(user_list, credibility_list)}
    ndcg_list = []
    spearman_list = []
    for post_users in user_rankings:
        num_post_users = len(post_users)
        post_user_ids = [x[0] for x in post_users if x[0] in user_credibility_map]
        mapped_creds = [(user, user_credibility_map[user], num_post_users-1-i) for i, user in enumerate(post_user_ids)]
        sorted_mapped_creds = sorted(mapped_creds, key=lambda x: x[1])[::-1]
        cred_ranking = [x[2] for x in sorted_mapped_creds]
        ndcg_list.append(ndcg_at_k(cred_ranking, 2, method=1))

        post_thanks = [x[1] for x in post_users]
        post_creds = [x[1] for x in mapped_creds]
        spearman_score = spearmanr(post_thanks, post_creds)[0]
        if not np.isnan(spearman_score):
            spearman_list.append(spearman_score)

    return np.mean(ndcg_list), np.mean(spearman_list)


def credibility_feature_correlation(user_features, user_list, credibility_list):
    post_creds, question_creds, reply_creds, thank_creds = [], [], [], []
    num_posts, num_questions, num_replies, num_thanks = [], [], [], []

    for user, credibility in zip(user_list, credibility_list):
        if user in user_features:
            if user_features[user]["n_posts"] > 0:
                post_creds.append(float(credibility))
                num_posts.append(user_features[user]["n_posts"])
            if user_features[user]["n_questions"] > 0:
                question_creds.append(float(credibility))
                num_questions.append(user_features[user]["n_questions"])
            if user_features[user]["n_replies"] > 0:
                reply_creds.append(float(credibility))
                num_replies.append(user_features[user]["n_replies"])
            if user_features[user]["n_thanks"] > 0:
                thank_creds.append(float(credibility))
                num_thanks.append(user_features[user]["n_thanks"])

    print("Calculate credibility corr on {} users with thanks".format(len(num_thanks)))
    post_corr = spearmanr(post_creds, num_posts)[0]
    question_corr = spearmanr(question_creds, num_questions)[0]
    reply_corr = spearmanr(reply_creds, num_replies)[0]
    thank_corr = spearmanr(thank_creds, num_thanks)[0]
    return post_corr, question_corr, reply_corr, thank_corr


def credibility_precision_correlation(user_precision, user_list, credibility_list):
    precision_creds, precisions = [], []

    for user, credibility in zip(user_list, credibility_list):
        if user in user_precision:
            if user_precision[user] > 0:
                precision_creds.append(float(credibility))
                precisions.append(user_precision[user])
    precision_corr = spearmanr(precision_creds, precisions)[0]
    return precision_corr
