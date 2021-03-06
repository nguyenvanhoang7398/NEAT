import torch
from torch import nn
import numpy as np
from model.utils import credibility_precision_correlation, credibility_feature_correlation, credibility_ranking
from dataset import NeatLoader


class NeatWPE(nn.Module):
    def __init__(self, vocab_size, user_size, num_classes, config):
        super(NeatWPE, self).__init__()
        self.config = config

        # Word embedding Layer
        self.word_embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.user_credibility = nn.Embedding(user_size, 1)
        self.lstm = nn.LSTM(input_size=self.config.embed_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.n_layers,
                            dropout=self.config.dropout,
                            batch_first=True,
                            bidirectional=self.config.bidirectional)
        # Dropout layer
        self.dropout = nn.Dropout(1-self.config.dropout)
        # tanh
        self.tanh = nn.Tanh()
        # Fully-connected layer
        self.fc = nn.Linear(
            self.config.hidden_size * self.config.n_layers * (1 + self.config.bidirectional),
            num_classes,
            bias=False
        )

    def forward(self, x_post_word_idxs, x_user_idxs, x_user_clusters):
        embedded_post_word_idxs = self.word_embeddings(x_post_word_idxs)
        user_creds = self.user_credibility(x_user_idxs)
        scaled_user_creds = torch.exp(user_creds)
        lstm_out, (h_n, c_n) = self.lstm(embedded_post_word_idxs)
        final_feature_map = self.dropout(h_n)  # shape=(num_layers * num_directions, num_posts, hidden_size)
        post_feature = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1)
        thread_feature = torch.matmul(torch.transpose(scaled_user_creds, 0, 1), post_feature)
        thread_feature = self.tanh(thread_feature)
        final_out = self.fc(thread_feature)
        return final_out[0], scaled_user_creds

    @staticmethod
    def init_module_weights(module):
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                if param.requires_grad:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        module.bias.data.fill_(0.01)

    def init_weights(self, data_loader):
        self.apply(self.init_module_weights)
        self.init_word_embeddings(data_loader.word_embeddings)
        self.init_user_credibility()

    def init_user_credibility(self):
        nn.init.constant_(self.user_credibility.weight, 0.)

    def init_word_embeddings(self, word_embeddings, freeze=True):
        word_embeddings_tensor = torch.from_numpy(word_embeddings)
        self.word_embeddings.weight.data.copy_(word_embeddings_tensor)
        self.word_embeddings.weight.requires_grad = (not freeze)

    def update_writer(self, tb_writer, data_loader: NeatLoader, global_step):
        user_credibility = np.exp(self.user_credibility.weight.detach().cpu().numpy())
        umls_precision_corr = credibility_precision_correlation(data_loader.user_precision_umls, data_loader.users,
                                                                user_credibility)
        adr_precision_corr = credibility_precision_correlation(data_loader.user_precision_adr, data_loader.users,
                                                               user_credibility)
        post_corr, question_corr, reply_corr, thank_corr = \
            credibility_feature_correlation(data_loader.user_features, data_loader.users, user_credibility)
        # ndcg_score, spearman_score = credibility_ranking(data_loader.user_rankings, data_loader.users, user_credibility)
        ndcg_score, spearman_score = 0, 0
        user_credibility_summary = {
            "max": np.max(user_credibility),
            "min": np.min(user_credibility),
            "post_corr": post_corr,
            "question_corr": question_corr,
            "reply_corr": reply_corr,
            "thank_corr": thank_corr,
            "umls_precision_corr": umls_precision_corr,
            "adr_precision_corr": adr_precision_corr,
            "ndcg": ndcg_score,
            "spearman": spearman_score
        }
        tb_writer.add_scalars("user_credibility", user_credibility_summary, global_step)
        tb_writer.add_embedding(user_credibility, metadata=data_loader.users,
                                global_step=global_step, tag="User credibility")
        return user_credibility_summary

    def regularize_loss(self):
        return torch.norm(torch.exp(self.user_credibility.weight), p=1)

