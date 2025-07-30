#!/usr/bin/env python3
"""
KJZUmap analysis for TCGA BRCA gene expression data
This script demonstrates how to use KJZUmap on gene expression data
"""

from time import time
from isumap import isumap
from data_and_plots import plot_data, load_TCGA_BRCA_GeneExpression, printtime, plot_gene_expression_data
from multiprocessing import cpu_count
import numpy as np

# KJZUmap parameters
k = 15  # Number of nearest neighbors
d = 2   # Embedding dimension
N = 1000  # Number of samples to use
sqrt = False
normalize = True
metricMDS = True
distBeyondNN = True
tconorm = "canonical"

if __name__ == '__main__':
    dataset_name = "TCGA_BRCA_GeneExpression"
    
    print("=== KJZUmap Analysis for TCGA BRCA Gene Expression Data ===")
    print(f"Parameters: k={k}, d={d}, N={N}")
    print(f"normalize={normalize}, metricMDS={metricMDS}, distBeyondNN={distBeyondNN}")
    
    # Load the gene expression data
    data, labels = load_TCGA_BRCA_GeneExpression(N)
    
    print(f"\nDataset loaded successfully!")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Plot the initial data (if 2D or 3D)
    if data.shape[1] <= 3:
        plot_data(data, labels, title=f"Initial_dataset_{dataset_name}", display=True)
    
    # Run KJZUmap
    print("\nRunning KJZUmap...")
    t0 = time()
    
    finalEmbedding, clusterLabels, sqrt_coef = isumap(
        data, k, d, sqrt=True,
        normalize=normalize, 
        distBeyondNN=distBeyondNN, 
        verbose=True, 
        dataIsDistMatrix=False, 
        dataIsGeodesicDistMatrix=False, 
        saveDistMatrix=False, 
        labels=labels, 
        initialization="cMDS", 
        metricMDS=metricMDS, 
        sgd_n_epochs=1500, 
        sgd_lr=1e-2, 
        sgd_batch_size=None,
        sgd_max_epochs_no_improvement=75, 
        sgd_loss='MSE', 
        sgd_saveplots_of_initializations=True, 
        sgd_saveloss=True, 
        tconorm=tconorm,
    )
    
    t1 = time()
    
    # Create title for the results
    title = f"{dataset_name}_N{N}_k{k}_beyondNN{distBeyondNN}_normalize{normalize}_metricMDS{metricMDS}_tconorm{tconorm}_sqrt{sqrt}"
    
    # Plot the results with labeled colors and sqrt coefficient from isumap
    plot_gene_expression_data(finalEmbedding, labels, title=title, display=True, dijkstra_info=sqrt_coef)
    print(f"\nResult saved in './Results/{title}.png'")
    print(f"Sqrt Coefficient from isumap: {sqrt_coef:.4f}")
    
    # Print timing information
    printtime("KJZUmap total time", t1-t0)
    
    # Show sample type distribution
    import pandas as pd
    label_names = {0: 'Normal', 1: 'Tumor', 2: 'Metastatic'}
    label_counts = pd.Series(labels).value_counts()
    print(f"\nSample type distribution:")
    for label, count in label_counts.items():
        print(f"  {label_names.get(label, f'Type {label}')}: {count}")
    
    # Print some statistics
    print(f"\n=== Results Summary ===")
    print(f"Original data shape: {data.shape}")
    print(f"Embedding shape: {finalEmbedding.shape}")
    print(f"Number of clusters found: {len(np.unique(clusterLabels))}")
    
    # Show sample type distribution
    import pandas as pd
    label_names = {0: 'Normal', 1: 'Tumor', 2: 'Metastatic'}
    label_counts = pd.Series(labels).value_counts()
    print(f"\nSample type distribution:")
    for label, count in label_counts.items():
        print(f"  {label_names.get(label, f'Type {label}')}: {count}")
    
    print(f"\nAnalysis complete! Check the Results folder for saved plots.") 