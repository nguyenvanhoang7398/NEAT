from torch import nn
from torch.nn.modules import pooling
import torch


class CnnBlock(nn.Module):
    def __init__(self, input_size, kernel_size, num_channels, max_post_len):
        super(CnnBlock, self).__init__()
        self.conv_1d = nn.Conv1d(in_channels=input_size, out_channels=num_channels,
                                 kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.mean_pool = pooling.AvgPool1d(max_post_len - kernel_size + 1)

    def forward(self, embedded_post_word_idxs):
        conv_result = self.conv_1d(embedded_post_word_idxs)
        conv_result = self.relu(conv_result)
        conv_output = self.mean_pool(conv_result)
        return conv_output

    def regularize_loss(self):
        return torch.norm(self.conv_1d.weight) + torch.norm(self.conv_1d.bias)


class AttentionCnnBlock(nn.Module):
    def __init__(self, input_size, kernel_size, num_channels, max_post_len, cluster_size):
        super(AttentionCnnBlock, self).__init__()
        self.conv_1d = nn.Conv1d(in_channels=input_size, out_channels=num_channels,
                                 kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.mean_pool = pooling.AvgPool1d(max_post_len - kernel_size + 1)

        # User cluster attention
        self.cluster_attention = nn.Embedding(cluster_size, num_channels)

    def forward(self, embedded_post_word_idxs, x_user_clusters):
        conv_result = self.conv_1d(embedded_post_word_idxs)
        conv_result = self.relu(conv_result)

        # Lookup cluster attention
        cluster_attn = self.cluster_attention(x_user_clusters).unsqueeze(-1)

        attn_scores = torch.matmul(conv_result.transpose(1, 2), cluster_attn)
        scaled_attn_scores = torch.softmax(attn_scores, dim=1)
        post_feature = torch.matmul(conv_result, scaled_attn_scores)
        return post_feature, scaled_attn_scores

    def regularize_loss(self):
        return torch.norm(self.conv_1d.weight) + torch.norm(self.conv_1d.bias)