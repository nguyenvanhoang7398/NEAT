import torch
from torch import nn
import numpy as np
from model.layers import CnnBlock
from model.utils import credibility_precision_correlation, credibility_feature_correlation, credibility_ranking
from dataset import NeatLoader


class NeatCnnWpe(nn.Module):
    def __init__(self, vocab_size, user_size, num_classes, max_post_len, config):
        super(NeatCnnWpe, self).__init__()
        self.config = config
        self.user_size = user_size

        # Word embedding Layer
        self.word_embeddings = nn.Embedding(vocab_size, self.config.embed_size)

        # User credibility
        self.user_credibility = nn.Embedding(user_size, 1)

        # 3 conv
        self.conv1 = CnnBlock(self.config.embed_size, self.config.kernel_size[0], self.config.num_channels,
                              max_post_len)
        self.conv2 = CnnBlock(self.config.embed_size, self.config.kernel_size[1], self.config.num_channels,
                              max_post_len)
        self.conv3 = CnnBlock(self.config.embed_size, self.config.kernel_size[2], self.config.num_channels,
                              max_post_len)

        # Dropout layer
        self.dropout = nn.Dropout(1-self.config.dropout)

        # tanh
        self.tanh = nn.Tanh()

        # Fully-connected layer
        self.fc = nn.Linear(
            self.config.num_channels * len(self.config.kernel_size),
            num_classes,
        )

    def forward(self, x_post_word_idxs, x_user_idxs, x_user_clusters):
        embedded_post_word_idxs = self.word_embeddings(x_post_word_idxs).transpose(1, 2)

        conv_out1 = self.conv1(embedded_post_word_idxs).squeeze(2)
        conv_out2 = self.conv2(embedded_post_word_idxs).squeeze(2)
        conv_out3 = self.conv3(embedded_post_word_idxs).squeeze(2)

        post_feature = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        post_feature = self.dropout(post_feature)

        # Look up user credibility
        user_creds = self.user_credibility(x_user_idxs)
        scaled_user_creds = torch.exp(user_creds)

        # Weight each post by its user credibility
        thread_feature = torch.matmul(torch.transpose(scaled_user_creds, 0, 1), post_feature)
        thread_feature = self.tanh(thread_feature)
        final_out = self.fc(thread_feature)
        return final_out, scaled_user_creds

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
        ndcg_score, spearman_score = credibility_ranking(data_loader.user_rankings, data_loader.users, user_credibility)
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

    def weight_norm(self):
        conv1_reg_loss = self.conv1.regularize_loss()
        conv2_reg_loss = self.conv2.regularize_loss()
        conv3_reg_loss = self.conv3.regularize_loss()
        linear_reg_loss = torch.norm(self.fc.weight)
        return conv1_reg_loss + conv2_reg_loss + conv3_reg_loss + linear_reg_loss

    def regularize_loss(self):
        weight_l2_norm = self.weight_norm()
        user_cred_l1_norm = torch.div(self.user_size,
                                      torch.norm(self.user_credibility.weight, dim=0, p=1))
        return self.config.weight_reg_coeff * weight_l2_norm + \
            (1 - self.config.weight_reg_coeff) * user_cred_l1_norm
