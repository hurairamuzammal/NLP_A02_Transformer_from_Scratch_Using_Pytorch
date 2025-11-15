"""
Fix the model bundle by correcting the embedded config from h=8 to h=16
"""
import torch
from pathlib import Path

# Path to the model bundle
model_path = Path("urdu_transformer_complete_export/model_export/urdu_transformer_causal.pt")

print("Loading model bundle...")
checkpoint = torch.load(model_path, map_location='cpu')

print(f"Current config: {checkpoint.get('config', 'No config found')}")

# Fix the config if it exists
if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
    if checkpoint['config'].get('h') == 8:
        print("Fixing h from 8 to 16...")
        checkpoint['config']['h'] = 16
        
        # Save the corrected bundle
        output_path = model_path.parent / "urdu_transformer_causal_fixed.pt"
        torch.save(checkpoint, output_path)
        print(f"✓ Fixed model saved to: {output_path}")
        print(f"New config: {checkpoint['config']}")
    else:
        print(f"Config already has h={checkpoint['config'].get('h')}")
else:
    print("No config found in bundle, checking state_dict keys...")
    if 'state_dict' in checkpoint:
        # Check attention head count from state_dict
        for key in checkpoint['state_dict'].keys():
            if 'self_attn.linears.0.weight' in key:
                weight_shape = checkpoint['state_dict'][key].shape
                print(f"Found attention weight: {key} with shape {weight_shape}")
                # For d_model=512 and h heads: weight should be (512, 512)
                # The actual head count is determined by MultiHeadedAttention.__init__
                break
        
        print("\nThe model state_dict exists. Creating new bundle with correct config...")
        new_bundle = {
            "state_dict": checkpoint.get('state_dict', checkpoint),
            "config": {
                "src_vocab": 10000,
                "tgt_vocab": 10000,
                "N": 2,
                "d_model": 512,
                "d_ff": 2048,
                "h": 16,  # CORRECT VALUE
                "dropout": 0.1
            },
            "training_type": "causal_language_modeling",
            "exported_at": "2025-11-15T00:00:00"
        }
        
        output_path = model_path.parent / "urdu_transformer_causal_fixed.pt"
        torch.save(new_bundle, output_path)
        print(f"✓ New bundle with correct config saved to: {output_path}")

print("\n" + "="*60)
print("INSTRUCTIONS:")
print("="*60)
print("1. Upload 'urdu_transformer_causal_fixed.pt' to Hugging Face Hub")
print("2. Rename it to 'urdu_transformer_causal.pt' (replace the old one)")
print("3. Or update cloud_config.json to use 'urdu_transformer_causal_fixed.pt'")
print("="*60)
