import sys
from src.utils.config_loader import load_config
from src.engine.device_manager import get_optimized_engine
from src.interface.ui import create_ui

# 1. Load Configuration
config = load_config("config.yaml")

# 2. Hardware-Aware Initialization
try:
    # The manager decides which heavy-lifting script to import
    engine = get_optimized_engine(config)
except Exception as e:
    print(f"CRITICAL HARDWARE ERROR: {e}")
    sys.exit(1)

# 3. Launch Interface
print("🚀 Engine Ready. Launching UI...")
demo = create_ui(engine, config)

if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)