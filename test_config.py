# test_none_mode.py
import sys
sys.path.append('.')
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

config_path = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
config = get_config_from_file(config_path)

# Get the model config
model_config = config.model

# Set semantic_mode to 'none'
model_config.params.shape_module_cfg.params.semantic_mode = 'none'

print(f"{'='*70}")
print("TESTING 'none' SEMANTIC MODE")
print(f"{'='*70}")

# Try to instantiate
model = instantiate_from_config(model_config)
print(f"✓ Model instantiated successfully!")

if hasattr(model, 'shape_model'):
    shape_model = model.shape_model
    print(f"\n✓ Found shape_model")
    print(f"  semantic_mode = '{shape_model.semantic_mode}'")
    
    print(f"\n  Semantic heads:")
    print(f"  Hidden head exists: {hasattr(shape_model, 'semantic_projection_hidden') and shape_model.semantic_projection_hidden is not None}")
    print(f"  Geometric head exists: {hasattr(shape_model, 'semantic_projection_geometric') and shape_model.semantic_projection_geometric is not None}")
    print(f"  Attention head exists: {hasattr(shape_model, 'semantic_attention_head') and shape_model.semantic_attention_head is not None}")
    
    # Expected output: All should be False for 'none' mode
    if (not hasattr(shape_model, 'semantic_projection_hidden') or shape_model.semantic_projection_hidden is None) and \
       (not hasattr(shape_model, 'semantic_projection_geometric') or shape_model.semantic_projection_geometric is None) and \
       (not hasattr(shape_model, 'semantic_attention_head') or shape_model.semantic_attention_head is None):
        print(f"\n✅ SUCCESS: No semantic heads initialized for 'none' mode!")

print(f"{'='*70}")