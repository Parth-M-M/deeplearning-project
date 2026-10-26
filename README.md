# Deep Learning — Practical Report 4 (PR 4)
## Project Choice Declaration: **PROJECT A — CNN from Scratch: CIFAR-10 Image Classifier**

---

## 1. Project Overview & CIFAR-10 Dataset
This repository implements **Project A (CNN from Scratch — CIFAR-10 Image Classifier)** in TensorFlow 2.x / Keras (`DL_PR4A.ipynb`). We systematically explore 2D convolutions, padding, strides, spatial pooling, classic vs modern CNN architectures, backpropagation mechanics, on-the-fly data augmentation, pre-trained Transfer Learning (`MobileNetV2`), and non-linear residual skip connections.

### CIFAR-10 Benchmark Dataset (`tensorflow.keras.datasets.cifar10`)
* **Source:** Canadian Institute For Advanced Research (CIFAR-10), built directly into `tensorflow.keras.datasets.cifar10`.
* **Total Images:** `60,000` colour RGB images of resolution `32×32×3` (`50,000` training images + `10,000` test images, balanced at `6,000` images per class).
* **10 Classes:** `0: Airplane` · `1: Automobile` · `2: Bird` · `3: Cat` · `4: Deer` · `5: Dog` · `6: Frog` · `7: Horse` · `8: Ship` · `9: Truck`.
* **Preprocessing:** Pixel intensities normalised from `uint8 [0, 255]` to `float32 [0.0, 1.0]` for stable gradient descent convergence; integer labels one-hot encoded to `(N, 10)` vectors for `categorical_crossentropy` loss.

---

## 2. Core Deep Learning Topics Covered (Tasks A1–A6)

### Topic 1: CNN vs ANN for Image Classification (Task A1)
* **Spatial Topology Preservation:** Unlike an ANN/MLP that flattens a `32×32×3` image into a 1D vector of `3,072` independent features (destroying 2D pixel neighbourhoods), a CNN applies 2D spatial filters across $(H, W, C)$ tensors.
* **Translation Invariance & Parameter Sharing:** A single `3×3` convolutional filter shares the exact same `9` weights across all `30×30 = 900` valid spatial positions, whereas an MLP requires `32×32 = 1,024` independent weights per output unit.

### Topic 2: 2D Convolution, Padding (`VALID` vs `SAME`), Strides & Feature Maps (Task A2)
* **Manual Convolution:** Demonstrated with `scipy.signal.convolve2d` and a `3×3` Laplacian edge-detection kernel `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]`.
* **Output Size Formula:** $\text{output\_size} = \lfloor \frac{\text{input\_size} - \text{kernel\_size} + 2P}{S} \rfloor + 1$.
  * `VALID` padding ($P=0, S=1$): `32×32 → 30×30`.
  * `SAME` padding ($P=1, S=1$): zero-pads borders (`34×34`) so output remains `32×32`, allowing deep layer stacking without spatial collapse.
  * `Stride = 2` ($S=2$): downsamples spatial dimensions by `2×` (`32×32 → 16×16`) as a learnable alternative to fixed pooling.

### Topic 3: Pooling, Flatten, LeNet-5 (1998) & Custom VGG-Style CNN (Task A3)
* **MaxPooling vs AveragePooling:** `MaxPooling2D(2,2)` retains the strongest activation peak in each `2×2` window (sharp features + local translation invariance), while `AveragePooling2D(2,2)` computes the smooth arithmetic mean. Modern CNNs replace `Flatten` (`(None, 16, 16, 16) → (None, 4096)`) with `GlobalAveragePooling2D` to drastically reduce Dense layer parameters.
* **LeNet-5 (LeCun et al., 1998):** Implemented to specification (`Conv2D(6, 5×5, tanh) → AvgPool → Conv2D(16, 5×5, tanh) → AvgPool → Flatten → Dense(120, tanh) → Dense(84, tanh) → Dense(10, softmax)`), verified at **62,006 parameters** (`63.40%` test accuracy).
* **Custom VGG-Style CNN Baseline:** Stacks two `3×3` `ReLU` convolutions (`SAME` padding) before each `MaxPooling2D(2,2)` layer followed by `GlobalAveragePooling2D`, totaling **175,018 parameters** (`< 2M` limit) and achieving **80.42%** test accuracy.

### Topic 4: Backpropagation in CNNs & Callback-Controlled Training (Task A4)
* **Max Location Switch (`MaxPooling2D` Backward Pass):** Only the pixel coordinate that achieved the maximum value during the forward pass receives `100%` of the upstream gradient; the remaining 3 positions in the `2×2` window receive `0.0`. In contrast, `AveragePooling2D` divides the upstream gradient equally ($1/4$ each), and `Flatten` reshapes the 1D gradient back into a 3D tensor.
* **Callbacks & Error Analysis:** Trained with `EarlyStopping(patience=10, restore_best_weights=True)` and `ReduceLROnPlateau(factor=0.5, patience=5)`. Confusion matrix analysis confirms `Cat ↔ Dog` (`280` errors) and `Automobile ↔ Truck` (`116` errors) are the most confused pairs due to visual compression at `32×32` resolution.

### Topic 5: Data Augmentation (`ImageDataGenerator` & Keras Layers) (Task A5)
* Applying random on-the-fly rotations ($\pm 15^\circ$), shifts ($\pm 10\%$), horizontal flips, and zooms ($\pm 10\%$) prevents memorisation, shrinks the train–validation generalisation gap from `10.58%` down to `0.86%`, and lifts test accuracy to **84.15%** (`ImageDataGenerator`) and **84.68%** (in-graph Keras `RandomFlip` / `RandomRotation` / `RandomZoom` / `RandomTranslation` layers).

### Topic 6: Transfer Learning (`MobileNetV2`) & Residual Connections (Task A6)
* **Feature Extraction (`base_model.trainable = False`):** CIFAR-10 images are resized to `96×96×3` and passed through a frozen ImageNet-pretrained `MobileNetV2` backbone with a trainable `GlobalAveragePooling2D → Dense(128, relu) → Dense(10, softmax)` head (`165,258` trainable params), reaching **87.60%** test accuracy in 20 epochs.
* **Fine-Tuning (Top 30 Layers Unfrozen at `lr = 1e-5`):** Unfreezing the top 30 layers (`1,691,722` trainable params) and training for 10 epochs at a very low learning rate (`1e-5` to prevent catastrophic forgetting of pre-trained weights) achieves **91.24%** test accuracy.
* **Residual Connection Block (ResNet Skip Connection):** Built using the Keras Functional API (`shortcut 1×1 Conv2D + stacked 3×3 Conv2D` merged via `layers.Add()`), creating a direct identity highway ($\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}(1 + \frac{\partial \mathcal{F}(x)}{\partial x})$) that solves the vanishing gradient problem in deep networks.

---

## 3. Visualisations & Plots (`/plots`)

| Plot File | Task | Description & Key Takeaway |
| :--- | :---: | :--- |
| `plots/a1_sample_grid.png` | **A1.2** | `5×10` grid showing 5 RGB examples for each of the 10 CIFAR-10 classes. |
| `plots/a1_pixel_histograms.png` | **A1.3** | Pixel intensity distributions before (`[0, 255]`) and after (`[0.0, 1.0]`) normalisation. |
| `plots/a2_manual_convolution.png` | **A2.1** | Manual 2D convolution on a greyscale Frog image using `scipy.signal.convolve2d` and a `3×3` Laplacian edge kernel (`32×32 → 30×30`). |
| `plots/a2_zero_padding.png` | **A2.2** | Visual comparison of unpadded (`32×32`) vs zero-padded (`34×34` via `np.pad`) input for `VALID` vs `SAME` padding. |
| `plots/a2_feature_maps.png` | **A2.4** | `4×4` grid of all 16 activation maps (`32×32`) extracted from a single `Conv2D(16, (3,3), relu, same)` layer. |
| `plots/a3_pooling_comparison.png` | **A3.1** | Side-by-side comparison of feature maps before pooling (`32×32`), after `MaxPooling2D(2,2)` (`16×16`), and after `AveragePooling2D(2,2)` (`16×16`). |
| `plots/a4_training_curves.png` | **A4.3** | Custom CNN training/validation loss & accuracy curves with `ReduceLROnPlateau` (grey dashed) and `EarlyStopping` (red dashed) annotations. |
| `plots/a4_confusion_matrix.png` | **A4.4** | `10×10` test set confusion matrix (`80.42%` accuracy) highlighting `Cat ↔ Dog` and `Automobile ↔ Truck` confusions. |
| `plots/a4_misclassified_samples.png` | **A4.4** | `2×5` grid of 10 misclassified test images displaying true labels, predicted labels, and softmax confidence. |
| `plots/a5_augmentation_grid.png` | **A5.2** | `2×5` grid showing 1 original CIFAR-10 image and 9 random `ImageDataGenerator` transformations. |
| `plots/a5_aug_vs_noaug_curves.png` | **A5.3** | 4-curve comparison of `model_cnn` (No-Aug) vs `model_aug` (With-Aug), showing a narrowed train–val gap (`10.6% → 0.9%`). |
| `plots/a6_finetuning_curves.png` | **A6.4** | Concatenated 30-epoch training curve (20 epochs Feature Extraction + 10 epochs Fine-Tuning) annotated at Epoch 20. |
| `plots/a6_results_comparison.png` | **A6.5** | Bar chart comparing test accuracies and parameter counts across all 6 trained models. |

---

## 4. Full 6-Model Results Comparison Table (Task A6.5)

| Model | Architecture | Augmentation | Transfer Learning | Test Accuracy | Parameters |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **1. LeNet-5 (Task A3.3)** | `2×Conv(5×5, tanh) + AvgPool + 3×Dense` | None | None (Scratch) | `63.40%` | `62,006` (`62,006` trainable) |
| **2. CNN Scratch — No Aug (Task A4)** | `5×Conv(3×3, ReLU, SAME) + MaxPool + GAP + Dense` | None | None (Scratch) | `80.42%` | `175,018` (`175,018` trainable) |
| **3. CNN Scratch + ImageDataGen (Task A5.3)** | `5×Conv(3×3, ReLU, SAME) + MaxPool + GAP + Dense` | `ImageDataGenerator` | None (Scratch) | `84.15%` | `175,018` (`175,018` trainable) |
| **4. CNN Scratch + Keras Aug (Task A5.4)** | `In-Graph Aug + 5×Conv(3×3, ReLU) + GAP + Dense` | Keras Preprocessing Layers | None (Scratch) | `84.68%` | `175,018` (`175,018` trainable) |
| **5. MobileNetV2 Feature Extr. (Task A6.3)** | `Frozen MobileNetV2 (96×96) + GAP + Dense(128)` | None (`96×96` Resize) | Feature Extraction (Frozen) | `87.60%` | `2,423,242` (`165,258` trainable) |
| **6. MobileNetV2 Fine-Tuned (Task A6.4)** | `MobileNetV2 (Top-30 Unfrozen, lr=1e-5) + Dense(128)` | None (`96×96` Resize) | Fine-Tuning (Top 30 Unfrozen) | **`91.24%`** | `2,423,242` (`1,691,722` trainable) |

---

## 5. Video Explanation Link & Speaking Script (Component B)
* **Recorded Video Walkthrough (Face PiP + Screen, 5–10 min):** `[Paste Your Google Drive (Anyone with link) or Unlisted YouTube Link Here — e.g., DL_PR4A_YourName_GRID.mp4]`

### Quick 7-Point Verbal Explanation Checklist (Rubric Criterion B2 — 1.5 Marks)
When recording your 5–10 minute webcam PiP + screen walkthrough of `DL_PR4A.ipynb`, verbally cover these 7 points as you scroll through each task:
1. **Why CNNs Need No Flattening Until After Feature Extraction:** Flattening an image immediately (as in an ANN/MLP) destroys 2D spatial relationships between neighbouring pixels. CNNs slide small `3×3` filters across the 2D grid, preserving spatial topology and sharing weights (`9` weights per channel instead of `1,024`).
2. **VALID vs SAME Padding & Output Size Formula:** State the formula $\text{out} = \lfloor \frac{\text{in} - K + 2P}{S} \rfloor + 1$. With `VALID` ($P=0$), a `32×32` image convolved with a `3×3` filter shrinks to `30×30`. With `SAME` ($P=1$), a 1-pixel zero border keeps the output at `32×32` so deep networks don't shrink to zero.
3. **Why MaxPooling Passes Only the Max Gradient ("Max Location Switch"):** In the forward pass, `MaxPooling2D(2,2)` forwards only the single largest value in each `2×2` patch. In backpropagation, only that winning pixel coordinate affected the output, so it receives `100%` of the upstream gradient while the other 3 pixels get `0`.
4. **LeNet-5 Historical Significance & Modern Differences:** Built by Yann LeCun in 1998 for reading handwritten postal codes/cheques; it pioneered the `Conv → Pool → Conv → Pool → Flatten → Dense` pattern, but used `tanh` activations, `AveragePooling`, and `5×5` filters instead of modern `ReLU`, `MaxPooling`/`GlobalAveragePooling`, and stacked `3×3` filters.
5. **Why Data Augmentation Reduces Overfitting:** Random on-the-fly rotations, shifts, flips, and zooms ensure the CNN never sees the exact same image twice, expanding the effective dataset size and shrinking the train–val gap from `10.6%` to `0.9%`.
6. **Feature Extraction vs Fine-Tuning (`MobileNetV2`):** Feature Extraction freezes the entire pre-trained ImageNet base (`2.26M` params) and trains only the new classification head (`165K` params) at `lr=1e-3`. Fine-Tuning unfreezes the top 30 layers and trains at a tiny learning rate (`1e-5`) so we gently adapt the pre-trained features without destroying them.
7. **Residual Connections (ResNet Skip Block):** Adding the shortcut branch $\mathbf{x}$ to the main conv branch $\mathcal{F}(\mathbf{x})$ via `layers.Add()` creates a direct identity path ($\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}(1 + \frac{\partial \mathcal{F}(x)}{\partial x})$) so gradients flow unimpeded during backpropagation, solving the vanishing gradient problem.

---

## 6. Tools, Libraries & Repository Structure

### Tech Stack & Installation
```bash
pip install -r requirements.txt
```
* **Framework:** TensorFlow `>=2.12.0` / Keras
* **Scientific Computing & Metrics:** NumPy, SciPy (`scipy.signal.convolve2d`), Scikit-Learn (`confusion_matrix`), Pandas
* **Visualisation:** Matplotlib, Seaborn

### Recommended Git Commit History (Rubric Criterion C3 — 3+ Commits)
Run these commands inside your project directory to push a clean, rubric-compliant 3-commit history to GitHub:
```bash
git init
git add requirements.txt DL_PR4A.ipynb plots/a1_* plots/a2_* plots/a3_*
git commit -m "Add Task A1-A3: CIFAR-10 EDA, manual convolution demo, pooling comparison, and LeNet-5"

git add plots/a4_* plots/a5_* DL_PR4A.ipynb
git commit -m "Train custom CNN with EarlyStopping/ReduceLROnPlateau and add Data Augmentation (Task A4-A5)"

git add plots/a6_* DL_PR4A.ipynb DL_PR4A.html README.md
git commit -m "Add MobileNetV2 feature extraction, fine-tuning, 6-model results table, and residual block (Task A6)"
```
