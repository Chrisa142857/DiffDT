import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import CLIPTextConfig, CLIPTextModel
from event_tokenizer import tokenizer, tokenizer_encode, tokenizer_decode, vocab_size
# device = 'cuda:6'
# Hugging Face Accelerate
from accelerate import Accelerator
import copy
from event_model import CLIPForCausalLM
from transformers import get_linear_schedule_with_warmup
def main():
    # 1. Initialize Accelerator
    accelerator = Accelerator()
    EPOCHS = 500
    batch_size = 768
    lr = 1e-3 * (batch_size*accelerator.state.num_processes/256)
    
    max_patience = 10
    # test_code = "A00"
    # tokens = tokenizer_encode(test_code)
    # print(f"\n[Verification] Tokenizing '{test_code}':")
    # print(f"   Tokens: {tokens}") 
    # # EXPECTED OUTPUT: ['A00', '.', '1'] 
    # # (Because 'A00' is in vocab, but 'A00.' is not merged)
    # print("-" * 40)
    # exit()
    
    # ==========================================
    # 4. Training Loop
    # ==========================================
    config = CLIPTextConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,       
        intermediate_size=1024, 
        num_hidden_layers=16,
        num_attention_heads=8,
        max_position_embeddings=200
    )
    
    model = CLIPForCausalLM(config)#.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    val_dataset = ICDDataset('icd10_val_split.txt', tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    dataset = ICDDataset("icd10_train_split.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model, optimizer, dataloader, val_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader)
    total_training_steps = len(dataloader) * EPOCHS
    
    # Standard Rule: Warmup is usually 5% to 10% of total steps
    num_warmup_steps = int(0.1 * total_training_steps)

    # 3. Create the Scheduler
    # Note: This returns a LambdaLR scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    # Register the scheduler with Accelerate (Important for checkpointing/resuming!)
    accelerator.register_for_checkpointing(lr_scheduler)
    print("Starting Training (Split Token Strategy)...")
    model.train()
    best_model = None
    best_loss = 1e+10
    patience = 0
    for epoch in range(EPOCHS):
        if patience > max_patience: break
        total_loss = 0
        val_loss = 0
        model.train()
        for batch in dataloader:
            seq_posids = batch["seq_posids"]#.to(device)
            input_ids = batch["input_ids"]#.to(device)
            mask = batch["attention_mask"]#.to(device)
            labels = batch["labels"]#.to(device)
    
            optimizer.zero_grad()
            output = model(input_ids, seq_posids, mask, labels)
            loss = output["loss"]
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                seq_posids = batch["seq_posids"]#.to(device)
                input_ids = batch["input_ids"]#.to(device)
                mask = batch["attention_mask"]#.to(device)
                labels = batch["labels"]#.to(device)
        
                output = model(input_ids, seq_posids, mask, labels)
                loss = output["loss"]
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        if val_loss <= best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            patience = 0
        else:
            patience += 1
        print(f"Epoch {epoch+1}, LR {lr_scheduler.get_last_lr()[0]:.06f} Patience {patience:02d} | Train Loss: {total_loss/len(dataloader):.10f} Val Loss: {val_loss:.10f} Best Loss: {best_loss:.10f}")
    
    save_path = 'icd10_tokenizer'
    print(f"Saving to {save_path}...")
    # tokenizer.save_pretrained(tokenizer_path)
    torch.save(best_model.state_dict(), f'{save_path}/text_model.pth')
    print("Done! Files saved.")

class ICDDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, 'r') as f:
            self.lines = [line.strip() for line in f.readlines()]
        # self.tokenizer = tokenizer
        self.max_len = 200

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        input_ids, seq_posids = tokenizer_encode(self.lines[idx])
        input_ids = torch.LongTensor(input_ids)
        seq_posids = torch.LongTensor(seq_posids)
        attn_mask = torch.ones_like(input_ids)
        seq_posids = torch.cat([seq_posids, torch.zeros(self.max_len-len(seq_posids))]).long()
        input_ids = torch.cat([input_ids, torch.zeros(self.max_len-len(input_ids))]).long()
        attn_mask = torch.cat([attn_mask, torch.zeros(self.max_len-len(attn_mask))]).long()
        return {
            "input_ids": input_ids,
            "seq_posids": seq_posids,
            "attention_mask": attn_mask,
            "labels": input_ids.clone()
        }
        
if __name__ == "__main__":
    main()