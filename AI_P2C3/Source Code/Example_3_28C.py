# Example 3.28c Plotting LazyPredict Results
import importlib.util
import os
import matplotlib.pyplot as plt

# Import data models dari Example_3_28B.py
if 'models' not in globals():
    try:
        from Example_3_28B import models
    except ImportError:
        _spec = importlib.util.spec_from_file_location(
            "Example_3_28B",
            os.path.join(os.path.dirname(__file__), "Example_3_28B.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        models = _mod.models

# Plot Visualisasi Akurasi
plt.figure(figsize=(12, 6))
plt.plot(models.index, models['Accuracy'], marker='o', color='b', linewidth=2)
plt.xticks(rotation=90)
plt.title("Perbandingan Akurasi Model (LazyPredict)")
plt.xlabel("Nama Model")
plt.ylabel("Akurasi")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()