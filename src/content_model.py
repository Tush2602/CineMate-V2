import torch
import torch.nn as nn

class ContentTower(nn.Module):
    def __init__(self, bert_dim=768, output_dim=32, dropout=0.2):
        super().__init__()

        self.bert_dim =bert_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.projector = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, content_embedding):
        return self.projector(content_embedding)
    
    

