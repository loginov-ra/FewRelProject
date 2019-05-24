import sys
sys.path.append('../')
import torch
import torch.nn as nn
from openai_transformer.model_pytorch import TransformerModel
from openai_transformer.model_pytorch import load_openai_pretrained_model, DEFAULT_CONFIG


class TransformerSentenceEncoder(nn.Module):
    def __init__(self, n_special, n_ctx=512, transformer_out_shape=768, out_shape=230):
        nn.Module.__init__(self)
        self.args = DEFAULT_CONFIG
        self.model = TransformerModel(self.args, vocab=40990 + n_special, n_ctx=n_ctx)
        load_openai_pretrained_model(self.model,
                                     path='../openai_transformer/model/',
                                     path_names='../openai_transformer/',
                                     n_special=n_special, n_ctx=n_ctx)
        self.model.embed.requires_grad = False
        for layer in self.model.h[:len(self.model.h) - 1]:
            for p in layer.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(transformer_out_shape, out_shape)

    def forward(self, inputs):
        h = self.model(inputs)
        h = self.fc(h[-1].squeeze(-1))
        return h
