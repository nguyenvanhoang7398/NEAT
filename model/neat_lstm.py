import torch
from torch import nn


class NeatLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, config):
        super(NeatLSTM, self).__init__()
        self.config = config

        # Word embedding Layer
        self.word_embeddings = nn.Embedding(vocab_size, self.config.embed_size)

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
        lstm_out, (h_n, c_n) = self.lstm(embedded_post_word_idxs)
        final_feature_map = self.dropout(h_n)  # shape=(num_layers * num_directions, num_posts, hidden_size)
        post_feature = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1)
        thread_feature = torch.mean(post_feature, dim=0)
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
