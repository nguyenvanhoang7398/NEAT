import torch
from torch import nn
from model.layers import CnnBlock


class NeatCnn(nn.Module):
    def __init__(self, vocab_size, num_classes, max_post_len, config):
        super(NeatCnn, self).__init__()
        self.config = config

        # Word embedding Layer
        self.word_embeddings = nn.Embedding(vocab_size, self.config.embed_size)

        # 3 conv
        self.conv1 = CnnBlock(self.config.embed_size, self.config.kernel_size[0], self.config.num_channels,
                              max_post_len)
        self.conv2 = CnnBlock(self.config.embed_size, self.config.kernel_size[1], self.config.num_channels,
                              max_post_len)
        self.conv3 = CnnBlock(self.config.embed_size, self.config.kernel_size[2], self.config.num_channels,
                              max_post_len)

        # tanh
        self.tanh = nn.Tanh()

        # Dropout layer
        self.dropout = nn.Dropout(1-self.config.dropout)

        # Fully-connected layer
        self.fc = nn.Linear(
            self.config.num_channels * len(self.config.kernel_size),
            num_classes,
            bias=False
        )

    def forward(self, x_post_word_idxs, x_user_idxs, x_user_clusters):
        embedded_post_word_idxs = self.word_embeddings(x_post_word_idxs).transpose(1, 2)

        conv_out1 = self.conv1(embedded_post_word_idxs).squeeze(2)
        conv_out2 = self.conv2(embedded_post_word_idxs).squeeze(2)
        conv_out3 = self.conv3(embedded_post_word_idxs).squeeze(2)

        post_feature = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        post_feature = self.dropout(post_feature)

        # Averaging all post feature to form thread feature
        thread_feature = torch.sum(post_feature, dim=0)
        thread_feature = self.tanh(thread_feature)
        final_out = self.fc(thread_feature)
        return [final_out]

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

    def init_word_embeddings(self, word_embeddings, freeze=True):
        word_embeddings_tensor = torch.from_numpy(word_embeddings)
        self.word_embeddings.weight.data.copy_(word_embeddings_tensor)
        self.word_embeddings.weight.requires_grad = (not freeze)

    def update_writer(self, tb_writer, data_loader, global_step):
        pass

    def regularize_loss(self):
        return 0
