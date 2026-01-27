import torch
import os
import shutil
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import Qwen3Config, Qwen3ForCausalLM, get_scheduler
import torch.optim as optim
from event_complete_tokenizer import tokenizer, tokenizer_encode, vocab_size
from datetime import datetime
# from tqdm import tqdm


def main():
    # Define output path
    # OUTPUT_DIR = "./icd10_tokenizer_base/qwen-icd10"
    OUTPUT_DIR = "./icd10_tokenizer_posid/qwen-icd10-complete"
    # --- 1. CONFIGURATION ---
    VOCAB_SIZE = vocab_size
    SEQ_LENGTH = 300   
    num_epochs = 500
    batch_size = 2048
    max_patience = 10
    lr = 1e-4

    ## Light 1.2B 
    # config = Qwen3Config(
    #     vocab_size=VOCAB_SIZE,
    #     hidden_size=2048,
    #     intermediate_size=5504,
    #     num_hidden_layers=24,
    #     num_attention_heads=16,
    #     num_key_value_heads=16,
    #     hidden_act="silu",
    #     max_position_embeddings=SEQ_LENGTH,
    #     initializer_range=0.02,
    #     rms_norm_eps=1e-6,
    #     use_cache=False,
    #     rope_theta=1000000.0,
    #     attention_dropout=0.0,
    #     tie_word_embeddings=False
    # )
    ## Nano 128M
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=512,
        intermediate_size=5504,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=SEQ_LENGTH,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        rope_theta=1000000.0,
        attention_dropout=0.1,
        tie_word_embeddings=False
    )
    
    # --- 2. INITIALIZE MODEL ---
    print("Initializing Qwen3 model...")
    model = Qwen3ForCausalLM(config)
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # Enable Gradient Checkpointing to save VRAM if needed
    model.gradient_checkpointing_enable() 
    
    # --- 3. DATA & ACCELERATOR ---
    # Accelerator handles mixed precision and device placement
    accelerator = Accelerator(mixed_precision="fp16") 

    val_dataset = ICDDataset('icd10_val_split.txt')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    dataset = ICDDataset("icd10-complete_train_split.txt")
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Now calculate steps correctly based on sharded dataloader
    num_training_steps = num_epochs * len(train_dataloader)
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    # Register scheduler with accelerator (handles stepping logic automatically)
    lr_scheduler = accelerator.prepare(lr_scheduler)

# --- 5. TRAINING LOOP ---
    print("Starting training...")
    best_loss = float('inf')
    patience = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # if patience > max_patience: 
        #     print("Max patience reached. Stopping.")
        #     break
            
        total_loss = 0
        model.train()
        
        for batch in train_dataloader:
            # ... (Training code is fine) ...
            seq_posids = batch["seq_posids"]
            input_ids = batch["input_ids"]
            mask = batch["attention_mask"]
            
            outputs = model(
                input_ids=input_ids, 
                labels=input_ids, 
                attention_mask=mask, 
                # position_ids=seq_posids 
            )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
        # --- VALIDATION ---
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                seq_posids = batch["seq_posids"]
                input_ids = batch["input_ids"]
                mask = batch["attention_mask"]
                
                outputs = model(
                    input_ids=input_ids, 
                    labels=input_ids, 
                    attention_mask=mask, 
                    # position_ids=seq_posids
                )
                val_loss += outputs.loss.item()
        
        # 1. Calculate Local Average
        local_avg_val_loss = val_loss / len(val_dataloader)
        
        # 2. SYNC: Gather metrics from all GPUs to calculate Global Average
        # Convert to tensor for gathering
        val_loss_tensor = torch.tensor(local_avg_val_loss, device=accelerator.device)
        
        # Gather all values and take the mean so all GPUs have the EXACT same number
        gathered_losses = accelerator.gather(val_loss_tensor)
        global_avg_val_loss = gathered_losses.mean().item()

        # 3. Use the GLOBAL average for the check
        if global_avg_val_loss <= best_loss:
            best_loss = global_avg_val_loss
            patience = 0
            
            # Now it is safe to sync because ALL GPUs are guaranteed to enter this block
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            
            if accelerator.is_main_process:
                best_model_state = {
                    k: v.cpu().clone() for k, v in unwrapped_model.state_dict().items()
                }
        else:
            patience += 1
            
        # Use accelerator.print to avoid duplicate logs
        accelerator.print(f"[{datetime.now()}] Epoch {epoch+1} | LR: {lr_scheduler.get_last_lr()[0]:.6f} | Train: {total_loss/len(train_dataloader):.5f} | Val: {global_avg_val_loss:.5f}")
        # Check for stopping
        if patience >= max_patience:
            accelerator.print("Early stopping triggered.")
            break
            
        # break
    print("Training finished.")
    # Only the main process has the 'best_model_state' variable populated
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and best_model_state is not None:
        print("Saving best model from CPU RAM to disk...")
        
        # Get a clean model shell
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Load the best weights we saved in RAM
        unwrapped_model.load_state_dict(best_model_state)
        
        # Save to disk
        save_path = os.path.join(OUTPUT_DIR, "best_model")
        unwrapped_model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
        
class ICDDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.lines = [line.strip() for line in f.readlines()]
        self.max_len = 200

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # Assuming tokenizer_encode is available globally or imported
        input_ids, seq_posids = tokenizer_encode(self.lines[idx])
        
        # Create tensors
        input_ids = torch.LongTensor(input_ids)
        seq_posids = torch.LongTensor(seq_posids)
        attn_mask = torch.ones_like(input_ids)
        
        # Padding logic
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            seq_posids = torch.cat([seq_posids, torch.zeros(pad_len, dtype=torch.long)])
            attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])
        else:
            # Safety truncation if line is longer than max_len
            input_ids = input_ids[:self.max_len]
            seq_posids = seq_posids[:self.max_len]
            attn_mask = attn_mask[:self.max_len]

        return {
            "input_ids": input_ids,
            "seq_posids": seq_posids,
            "attention_mask": attn_mask,
        }

if __name__ == "__main__":
    main()