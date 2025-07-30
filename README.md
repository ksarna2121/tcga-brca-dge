# 🧬 Gene Expression Clustering in Breast Cancer Tissue Using Custom ISUmap

This project explores unsupervised analysis of gene expression data from TCGA BRCA (breast cancer) samples using PCA, UMAP, and a custom variant of ISUmap. The goal is to uncover intrinsic tumor vs. normal structure through clustering, while testing improvements to UMAP via a custom neighborhood distance metric.

---

## 📁 Dataset

- **Source**: [TCGA BRCA (GDC)](https://gdc.cancer.gov/about-data/publications/tcga)
- **Samples**: ~1,200 (tumor and normal)
- **Genes**: ~60,000
- **Data**: RNA-seq raw counts + phenotype metadata

---

## 🔬 Methods So Far

### 1. 🧼 Preprocessing
- Filtered out low-expression genes (kept genes with count >10 in ≥10 samples)
- Dropped columns with >50% missing data
- Log-transformed count data
- Added tumor/normal binary labels

### 2. 📊 PCA
- Applied PCA for dimensionality reduction
- First 2 PCs capture strong tumor vs. normal separation
- Achieved **89% classification accuracy** using just PCs

### 3. 📈 Clustering and Visualization
- Visualized PCA and UMAP embeddings
- Integrated a custom ISUmap implementation using a √2-based intra-cluster distance metric
- Custom ISUmap preserves neighborhood relationships and improves local clustering in high dimensions

---

## 🧠 Custom ISUmap

This repo includes a modified version of ISUmap (Improved Symmetric UMAP), which adjusts intra-neighborhood distances using a √2-based metric inspired by high-dimensional geometry. This reflects the near-orthogonality of neighbors in high-dimensional space and offers better preservation of cluster structure.

📄 See included PDF:  
**"Improving ISUmap: Modifying the Intra-Neighborhood Distance Metric for High-Dimensional Data"**  
*by Kavi Sarna (2024)*

---

## 📌 Project Structure
📁 data/ # Raw and processed gene expression data
📁 figures/ # PCA and UMAP plots
📁 isumap/ # Custom ISUmap implementation
📄 analysis.ipynb # Notebook for cleaning, PCA, and clustering
📄 README.md # You're here!


---

## 🚀 Next Steps

- Perform differential gene expression analysis (e.g., DESeq2-like approach)
- Run pathway enrichment using KEGG or MSigDB
- Quantify cluster coherence and separation metrics
- Expand analysis to ATAC-seq or methylation datasets

---

## 🧪 How to Run

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
pip install -r requirements.txt
jupyter notebook
