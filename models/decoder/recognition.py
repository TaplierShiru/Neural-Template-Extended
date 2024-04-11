from torch import nn
import torch.nn.functional as F


class RecognitionDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.in_f = config.single_embbeding_size
        self.out_f = config.recognition_num_classes
        self.linear_1 = nn.Linear(self.in_f, self.in_f * 2, bias=True)
        self.linear_2 = nn.Linear(self.in_f * 2, self.in_f * 4, bias=True)
        self.linear_3 = nn.Linear(self.in_f * 4, self.in_f * 4, bias=True)
        self.linear_4 = nn.Linear(self.in_f * 4, self.out_f, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias, 0)

    def forward(self, embedding):
        l1 = self.linear_1(embedding)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        return l4
