
import torch
import numpy as np
import torch.nn as nn
from transformers import CLIPTextConfig, CLIPTextModel



class CLIPForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip_text = CLIPTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.clip_text.text_model.embeddings.token_embedding.weight

    def forward(self, input_ids, seq_posids, attention_mask=None, labels=None):
        outputs = self.clip_text(input_ids=input_ids, attention_mask=attention_mask, position_ids=seq_posids)
        logits = self.lm_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., 1:-1, :].contiguous()
            shift_labels = labels[..., 2:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits}

