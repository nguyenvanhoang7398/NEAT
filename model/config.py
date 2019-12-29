class NeatLstmConfig(object):
    def __init__(self, hidden_size, embed_size, n_layers, dropout, bidirectional):
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

    @staticmethod
    def get_common():
        return NeatLstmConfig(
            hidden_size=32,
            embed_size=50,
            n_layers=1,
            dropout=0.3,
            bidirectional=True
        )


class NeatCnnConfig(object):
    def __init__(self, embed_size, num_channels, kernel_size, dropout,
                 cred_lr_coeff, ue_coeff, hidden_unit):
        self.embed_size = embed_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.cred_lr_coeff = cred_lr_coeff
        self.ue_coeff = ue_coeff
        self.hidden_unit = hidden_unit

    @staticmethod
    def get_common(ue_coeff=0.2):
        return NeatCnnConfig(
            embed_size=50,
            num_channels=100,
            kernel_size=[3, 4, 5],
            dropout=0.3,
            cred_lr_coeff=1,
            ue_coeff=ue_coeff,
            hidden_unit=5
        )
