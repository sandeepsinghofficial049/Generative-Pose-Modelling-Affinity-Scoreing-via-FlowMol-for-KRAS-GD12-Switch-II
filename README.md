# Generative Pose Modeling & Affinity Scoring via Flow Matching

An advanced, production-grade computational pipeline utilizing **multi-modal flow matching architectures** and **deep geometric representations** for de novo ligand pose generation and native binding affinity scoring, bypassing traditional physics-based scoring limitations.

---

## 🚀 Overview

Traditional structure-based drug discovery (SBDD) relies heavily on rigid docking algorithms and resource-heavy scoring functions that often struggle with receptor flexibility and accurate binding free-energy estimation.

**Generative Pose Modeling & Affinity Scoring via Flow Matching** bridges this gap by framing ligand pose generation as a continuous normalizing flow problem. By learning vector fields that push a prior noise distribution directly onto the true bound atomic coordinate distribution, this framework achieves state-of-the-art accuracy in predicting ligand conformations and binding affinities.

---

## 🛠️ Core Architecture & Methodology

```
[Protein PDB & Ligand SMILES] 
       │
       ▼
[Geometric Feature Extraction] (SE(3)-Equivariant Graph Layers)
       │
       ▼
[Conditional Flow Matching Network] (Neural Vector Field Estimation)
       │
       ▼
[Generative Pose Sampling] (ODE Integration from Prior Noise)
       │
       ▼
[Native Affinity Scoring Head] (Continuous pKd / ΔG Prediction)

```

1. **Geometric Representation:** Encodes protein pockets and ligand atom graphs using invariant and equivariant features.
2. **Flow Matching Backbone:** Learns a time-dependent vector field $v_t$ that steers straight trajectories from a standard Gaussian noise prior to the target atomic coordinates.
3. **Affinity Prediction Head:** Jointly evaluates the generated 3D poses against the receptor pocket microenvironment to output robust binding affinity metrics ($\Delta G$ and $pK_d$).

---

## ⚙️ Key Features

* **Continuous Flow Matching:** Avoids the sampling bottlenecks of traditional diffusion models by utilizing optimal-transport straight-line trajectories.
* **SE(3)-Equivariance:** Ensures that rotations and translations of the input protein-ligand complex transform the output coordinates correctly.
* **End-to-End Inference:** Takes raw PDB structures and ligand SMILES strings, outputs fully optimized 3D binding poses, and predicts binding affinity scores.
* **Cloud-Optimized Execution:** Engineered to run efficiently on standard cloud GPUs (such as Google Colab Tesla T4/A100 instances), bypassing heavy local VRAM restrictions.

---

## 📂 Repository Structure

```text
├── data/                  # Sample datasets and preprocessed protein-ligand complexes
├── models/                # Flow matching network and SE(3)-equivariant architecture definitions
├── scripts/               # Training, inference, and evaluation utility scripts
├── notebooks/             # Interactive Google Colab / Kaggle execution notebooks
├── requirements.txt       # Python dependencies (PyTorch, PyTorch Geometric, RDKit)
└── README.md              # Project documentation

```

---

## 📦 Installation & Setup

Clone the repository and install the required dependencies within your Python/conda environment:

```bash
git clone https://github.com/sandeepsinghofficial049/Generative-Pose-Flow-Matching.git
cd Generative-Pose-Flow-Matching

pip install -r requirements.txt

```

---

## 💻 Quickstart Inference

To run generative pose prediction and affinity scoring on a sample protein-ligand complex:

```python
from models.flow_matcher import FlowMatchingPoseGenerator
from utils.data_loader import load_protein_ligand_pair

# Load receptor pocket and ligand features
protein_pocket, ligand = load_protein_ligand_pair("data/sample_pocket.pdb", "CCO")

# Initialize the flow matching model
model = FlowMatchingPoseGenerator.from_pretrained("weights/flow_matching_pose_v1.pt")
model.eval()

# Generate 3D atomic coordinates via vector field integration
generated_poses, predicted_affinity = model.sample_poses_and_score(
    protein_pocket, 
    ligand, 
    num_steps=50
)

print(f"Generated Pose Coordinates Shape: {generated_poses.shape}")
print(f"Predicted Binding Affinity (pK_d): {predicted_affinity.item():.2f}")

```

---

## 📊 Evaluation Metrics

* **Root Mean Square Deviation (RMSD):** Measures spatial deviation between generated ligand heavy-atom poses and experimental crystal structures ($\text{Å}$).
* **Pearson/Spearman Correlation ($R$):** Evaluates ranking capability of predicted binding affinities against experimental benchmark datasets.

---

## 👤 Author

**Sandeep Singh**

*M.Sc. Bioinformatics, Central University of South Bihar*

[GitHub Profile](https://github.com/sandeepsinghofficial049) | [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/sandeep-singh-9b36a9407)
