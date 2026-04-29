# EyeScan-AI-Retinal-Disease-Detection-System
Deep learning app for early detection of retinal diseases (Cataract, Glaucoma, Diabetic Retinopathy) from fundus images using EfficientNetB0 — built for KaggleHacX '26 AI for Healthcare.
# EyeScan AI 👁️

> Advanced deep learning system for early detection of retinal diseases from fundus images.
> Built for **KaggleHacX '26 — AI for Healthcare** track.

## Results
| Metric | Score |
|--------|-------|
| Overall Accuracy | 94% |
| Cataract F1 | 0.96 |
| Diabetic Retinopathy F1 | 0.99 |
| Glaucoma F1 | 0.89 |
| Normal F1 | 0.91 |

## Disease Classes
- **Cataract** — Clouding of the eye's natural lens
- **Diabetic Retinopathy** — Damage to retinal blood vessels
- **Glaucoma** — Optic nerve damage from eye pressure
- **Normal** — No disease detected

## Model Architecture
- Backbone: EfficientNetB0 (ImageNet pretrained)
- Transfer learning + fine-tuning
- Input size: 224×224
- Training images: 4,217
- Framework: TensorFlow/Keras

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn



## Project Structure
