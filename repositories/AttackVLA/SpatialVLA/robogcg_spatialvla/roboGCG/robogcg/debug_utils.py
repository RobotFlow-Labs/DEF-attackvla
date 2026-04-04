import torch
import logging

logger = logging.getLogger(__name__)

def debug_perfect_match(
    self,
    predicted_tokens,
    shift_labels,
    shift_logits,
    input_embeds_batch,
    action_dim
):
    """Debug helper for when a perfect token match is found."""
    predicted_actions = self.action_tokenizer.tokens_to_actions(predicted_tokens)
    target_actions = self.action_tokenizer.tokens_to_actions(shift_labels)
    print(f"\nPerfect token match found!")
    
    # Debug search context
    print("\n=== Search Context ===")
    # Extract the optimized tokens from the current batch
    if self.prefix_cache:
        optim_embeds = input_embeds_batch[0, :-(self.after_embeds.shape[1] + self.target_embeds.shape[1])]
    else:
        optim_embeds = input_embeds_batch[0, self.before_embeds.shape[1]:-(self.after_embeds.shape[1] + self.target_embeds.shape[1])]
    
    # Convert embeddings back to tokens
    curr_optim_tokens = torch.argmax(optim_embeds @ self.embedding_layer.weight.T, dim=-1)
    
    # Convert before/after strings to token IDs
    before_tokens = self.tokenizer(self.before_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.model.device)
    after_tokens = self.tokenizer(self.after_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.model.device)

    search_tokens = torch.cat([
        before_tokens,
        curr_optim_tokens,
        after_tokens,
        self.target_ids[0]
    ])
    # Get search embeddings for prompt portion only
    search_embeds = torch.cat([
        self.before_embeds[0],
        optim_embeds,
        self.after_embeds[0]
    ])
        
    print(f"Full search sequence length: {len(search_tokens)}")
    print(f"Full search tokens: {search_tokens.tolist()}")
    print(f"Prediction window starts at: {len(search_tokens) - len(self.target_ids[0]) - 1}")

    debug_logits("Search", shift_logits, shift_labels)
    
    print(f"\nPredicted actions: {predicted_actions[0].tolist()}")
    print(f"Target actions: {target_actions[0].tolist()}")
    print(f"Action differences: {(predicted_actions[0] - target_actions[0]).tolist()}")

    # Validate the full prompt
    print("\n=== Validating Full Prompt ===")
    optim_str = self.tokenizer.decode(curr_optim_tokens)
    full_prompt = f"{self.before_str}{optim_str}{self.after_str}"
    print(f"Full prompt: {full_prompt}")
    
    debug_generate(self, full_prompt, target_actions, action_dim)
    debug_forward_pass(self, full_prompt, shift_labels, search_embeds, action_dim, target_actions)

def debug_logits(prefix, logits, labels):
    """Debug helper for examining logits."""
    print(f"\n=== {prefix} Logits ===")
    for pos in range(logits.shape[1]):
        top_logits, top_tokens = logits[0, pos].topk(5)
        print(f"\nPosition {pos}:")
        print(f"Target token: {labels[0, pos]}")
        print(f"Top 5 predictions: {top_tokens.tolist()} with logits: {top_logits.tolist()}")

def debug_generate(self, full_prompt, target_actions, action_dim):
    """Debug helper for generation method."""
    print("\n--- Method 1: Using generate() ---")
    inputs = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
    print(f"Full validation sequence length: {len(inputs['input_ids'][0])}")
    print(f"Full validation tokens: {inputs['input_ids'][0].tolist()}")
    print(f"Prediction window should start at: {len(inputs['input_ids'][0]) - action_dim - 1}")
    
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=action_dim,
            pad_token_id=self.tokenizer.pad_token_id
        )
    
    new_tokens = outputs[0, inputs['input_ids'].shape[1]:]
    validation_actions = self.action_tokenizer.tokens_to_actions(new_tokens.unsqueeze(0))
    
    print(f"Generate() actions: {validation_actions[0].tolist()}")
    print(f"Target actions: {target_actions[0].tolist()}")
    print(f"Generate() differences: {(validation_actions[0] - target_actions[0]).tolist()}")

def debug_forward_pass(self, full_prompt, shift_labels, search_embeds, action_dim, target_actions):
    """Debug helper for forward pass method."""
    print("\n--- Method 2: Single Forward Pass ---")
    inputs = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
    
    with torch.no_grad():
        validation_embeds = self.embedding_layer(inputs['input_ids'])
        
        debug_embedding_comparison(self, search_embeds, validation_embeds[0])
        
        forward_outputs = self.model(inputs_embeds=validation_embeds)
    
    forward_logits = forward_outputs.logits[:, -action_dim-1:-1, :]
    debug_logits("Forward Pass", forward_logits, shift_labels)
    
    predicted_tokens = torch.argmax(forward_logits, dim=-1)
    forward_actions = self.action_tokenizer.tokens_to_actions(predicted_tokens)
    
    print(f"\nForward pass actions: {forward_actions[0].tolist()}")
    print(f"Target actions: {target_actions[0].tolist()}")
    print(f"Forward pass differences: {(forward_actions[0] - target_actions[0]).tolist()}")
    print("=" * 50)

def debug_embedding_comparison(self, search_embeds, validation_embeds):
    """Debug helper for comparing embeddings."""
    print("\n=== Embedding Comparison ===")
    print(f"Search embeddings shape: {search_embeds.shape}")
    print(f"Validation embeddings shape: {validation_embeds.shape}")
    
    min_length = min(search_embeds.shape[0], validation_embeds.shape[0])
    embedding_diff = (search_embeds[:min_length] - validation_embeds[:min_length]).abs()
    print(f"Max embedding difference: {embedding_diff.max().item()}")
    print(f"Mean embedding difference: {embedding_diff.mean().item()}")
    
    if embedding_diff.max().item() > 1e-5:
        print("\nPositions with large differences:")
        large_diffs = torch.where(embedding_diff.max(dim=-1)[0] > 1e-5)[0]
        for pos in large_diffs:
            search_token = torch.argmax(search_embeds[pos] @ self.embedding_layer.weight.T)
            val_token = torch.argmax(validation_embeds[pos] @ self.embedding_layer.weight.T)
            print(f"Position {pos}:")
            print(f"  Search token: {search_token.item()} ({self.tokenizer.decode([search_token.item()])})")
            print(f"  Validation token: {val_token.item()} ({self.tokenizer.decode([val_token.item()])})") 