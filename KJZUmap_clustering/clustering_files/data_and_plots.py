import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
from sklearn import datasets
from sklearn.utils import check_random_state
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import pickle
import os
import pickle

def createFourGaussians(distance,numberOfPoints):
    def generate_points(means,covs,ns):
        assert len(means)==len(covs)==len(ns)
        
        X= [np.random.multivariate_normal(means[i],covs[i],size=(ns[i],)) for i in range(len(ns))]
        X= np.vstack(X)
        
        y = [np.ones(ns[i])*i for i in range(len(ns))]
        y = np.hstack(y)
        
        return X,y

    pointsPerGaussian = round(numberOfPoints/4)
    means = [np.array([-4*distance,0]),np.array([1.2*distance,0]),np.array([0,1*distance]),np.array([0,-2*distance])]
    covs = [2*np.eye(2),2*np.eye(2),2*np.eye(2),2*np.eye(2)]
    ns = [pointsPerGaussian,pointsPerGaussian,pointsPerGaussian,pointsPerGaussian]

    dataset,colors = generate_points(means,covs,ns)
    return dataset,colors

def createMoons(numberOfPoints,noise=0.1,seed=42):
    ms = datasets.make_moons(n_samples=numberOfPoints,noise=noise,random_state=np.random.RandomState(seed))
    dataset = ms[0]
    colors = ms[1]
    return dataset,colors

def createNonUniformHemisphere(N,seed0=0,k=30):
    rng = np.random.default_rng(seed=seed0)
    theta = rng.uniform(0, 2 * np.pi, size=N)
    phi = rng.uniform(0, 0.5 * np.pi, size=N)
    t = rng.uniform(0,1,size = N)
    P=  np.arccos(1 - 2 * t)

    if ((P < (np.pi - (np.pi / 4))) & (P > ((np.pi / 4)))).all():
        indices = (P < (np.pi - (np.pi / 4))) & (P > ((np.pi /4)))
        x,y,z = (np.sin(P[indices]) * np.cos(theta[indices]),
                np.sin(P[indices]) * np.sin(theta[indices]),
                np.cos(P[indices]))            
    else :
        x,y,z =(np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi))
    
    data = np.array([x,y,z]).T

    # for coloring the hemisphere:
    M_train, M_test, N_train, N_test = train_test_split(data, data, test_size=int(N/2), random_state=seed0)
    AC = AgglomerativeClustering(n_clusters=7, linkage='ward')
    AC.fit(M_train)
    labels = AC.labels_
    KN = KNeighborsClassifier(n_neighbors=k)
    KNN = KN.fit(M_train,labels)
    colors = KN.predict(data)
    
    return data, colors

def createTorus(N,seed=0):
    random_state = check_random_state(seed)
    theta = random_state.rand(N) * (2 * np.pi)
    phi = random_state.rand(N) * (2 * np.pi)
    colors = phi
    dataset = np.array([(2+np.cos(theta))*np.cos(phi),
            (2+np.cos(theta))*np.sin(phi),
            np.sin(theta)]).T
    return dataset,colors

def createSwissRole(N,hole=True,seed=0):
    return datasets.make_swiss_roll(n_samples=N, hole = hole, noise=0.0 , random_state=seed)

def createBreastCancerDataset():
    data = pd.read_csv('Dataset_files/BreastCancerDataset.csv')
    data = data.drop('id',axis=1)
    data=data.drop('Unnamed: 32',axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    X = data.values
    colors=data['diagnosis']
    return X,colors

def createMammoth(N,k=30,seed=42):
    mammoth = pd.read_csv('Dataset_files/mammoth.csv')
    if N <= len(mammoth):
        mammoth_sample = mammoth.sample(N)
    else:
        mammoth_sample = mammoth
    
    # for coloring the mammoth:
    M_train, M_test, N_train, N_test = train_test_split(mammoth_sample, mammoth_sample, test_size=int(N/2), random_state=seed)
    AC = AgglomerativeClustering(n_clusters=11, linkage='ward')
    AC.fit(M_train)
    labels = AC.labels_
    KN = KNeighborsClassifier(n_neighbors=k)
    KNN = KN.fit(M_train,labels)
    colors = KN.predict(mammoth_sample)

    return np.array(mammoth_sample), colors

def load_and_store_data_file(N,filename,filetype='.pkl'):
    # Check if the file already exists
    if not os.path.exists('Dataset_files/'+filename+filetype):
        print("\nDownloading '"+filename+"' data.")
        ddata = datasets.fetch_openml(filename)
        # Create the output directory if it doesn't exist
        os.makedirs('Dataset_files', exist_ok=True)

        data = np.array(ddata['data'])
        labels = np.array(ddata['target'])
        # Save the dataset as a .pkl file
        with open('Dataset_files/'+filename+filetype, 'wb') as f:
            pickle.dump((data, labels), f)

        print("Download successful. The files are stored in 'Dataset_files/"+filename+filetype+"' and are directly loaded from there in case you run this script a second time.")
    print("\nLoading '"+filename+"' data from file")
    with open('Dataset_files/'+filename+filetype, 'rb') as f:
        data, labels = pickle.load(f)
    print("Selecting subset of N = ",N)
    indices = random.sample(range(len(data)), N)
    data = np.array(data[indices],dtype=np.float32)
    labels = np.array(labels[indices],dtype=np.int64)
    return data,labels

def load_MNIST(N):
    return load_and_store_data_file(N,'mnist_784')

def load_FashionMNIST(N):
    return load_and_store_data_file(N,'fashion-mnist')

def load_TCGA_BRCA_GeneExpression(N):
    """
    Load TCGA BRCA gene expression dataset
    Returns processed gene expression data and sample type labels
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    print(f"Loading TCGA BRCA gene expression data (N={N})...")
    
    # Load the data
    phenotype_data = pd.read_csv('../../data/TCGA-BRCA.clinical.tsv', sep='\t')
    gene_counts_data = pd.read_csv('../../data/TCGA-BRCA.star_counts.tsv', sep='\t')
    
    print(f"Original data shapes:")
    print(f"Phenotype data: {phenotype_data.shape}")
    print(f"Gene counts data: {gene_counts_data.shape}")
    
    # Separate the data into tumor, normal, and metastatic
    tumor_ids = phenotype_data[phenotype_data['sample_type.samples'] == 'Primary Tumor']['sample']
    normal_ids = phenotype_data[phenotype_data['sample_type.samples'] == 'Solid Tissue Normal']['sample']
    metastatic_ids = phenotype_data[phenotype_data['sample_type.samples'] == 'Metastatic']['sample']
    
    # Get the gene data for the tumor, normal, and metastatic samples
    tumor_ids_existing = [col for col in tumor_ids if col in gene_counts_data.columns]
    normal_ids_existing = [col for col in normal_ids if col in gene_counts_data.columns]
    metastatic_ids_existing = [col for col in metastatic_ids if col in gene_counts_data.columns]
    
    tumor_gene_seq = gene_counts_data[tumor_ids_existing].T
    normal_gene_seq = gene_counts_data[normal_ids_existing].T
    metastatic_gene_seq = gene_counts_data[metastatic_ids_existing].T
    
    # Add identifiers
    tumor_gene_seq['sample_type'] = 1  # Tumor
    normal_gene_seq['sample_type'] = 0  # Normal
    metastatic_gene_seq['sample_type'] = 2  # Metastatic
    
    # Combine the data
    full_data = pd.concat([tumor_gene_seq, normal_gene_seq, metastatic_gene_seq], axis=0)
    
    print(f"Combined data shape: {full_data.shape}")
    print(f"Sample type counts:")
    print(full_data['sample_type'].value_counts())
    
    # Filter low expression genes (10+ counts in 1000+ samples)
    keep_genes = (full_data.drop(columns=['sample_type']) > 10).sum(axis=1) >= 1000
    filtered_data = full_data.loc[keep_genes]
    
    print(f"Filtered data shape: {filtered_data.shape}")
    
    # Log transform the data
    log_counts = np.log2(filtered_data.drop(columns=['sample_type']) + 1)
    sample_labels = filtered_data['sample_type']
    
    print(f"Final data shape: {log_counts.shape}")
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(log_counts)
    
    # If N is smaller than available samples, randomly sample
    if N < len(data_scaled):
        indices = np.random.choice(len(data_scaled), N, replace=False)
        data_scaled = data_scaled[indices]
        sample_labels = sample_labels.iloc[indices]
    
    print(f"Final dataset shape: {data_scaled.shape}")
    print(f"Sample type distribution:")
    print(pd.Series(sample_labels).value_counts())
    
    return data_scaled, sample_labels.values



### Plotting data

def plot_MNIST_samples(images, labels, num_samples=10):
    num_rows = num_samples // 8 + (num_samples % 8 > 0)  # Calculate the number of rows
    plt.figure(figsize=(12, 12))
    jet = cm.get_cmap('jet', 10)
    for i in range(num_samples):
        plt.subplot(num_rows, 8, i+1)
        image = images.iloc[i].values.reshape(28, 28)  # reshape the data into a 28x28 array
        label = labels.iloc[i]  # get the label of the image
        color = jet(label / 9.)
        cmap = mcolors.ListedColormap(['white', color])  # create a custom color map
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
    plt.tight_layout()  # Adjust the padding between and around the subplots
    plt.show()

def plot_data(data,labels,title='Data',save=True,display=False):
    if data.shape[0]==labels.shape[0]:
        fig = plt.figure(figsize=(12, 12))
        plt.title(title)
        dim = data.shape[1]
        if dim==2:
            plt.scatter(data[:,0],data[:,1],s=3,c=labels, cmap="jet")
            # plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='datalim')
            plt.axis('off')
        elif dim==3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(data[:,0],data[:,1],data[:,2],s=3,c=labels, cmap="jet")
            ax.view_init(20, -20)
        else:
            return 
            raise("Invalid dimension for plot")
        if save:
            # Create the output directory if it doesn't exist
            os.makedirs('Results', exist_ok=True)
            plt.savefig('./Results/'+title+'.png')
        if display:
            plt.show()

def plot_gene_expression_data(data, labels, title='Gene_Expression_Data', save=True, display=False, dijkstra_info=None):
    """
    Custom plotting function for gene expression data with labeled colors and coefficient information
    """
    if data.shape[0] != labels.shape[0]:
        print("Error: Data and labels have different lengths")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors and labels for gene expression data
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    color_labels = {0: 'Normal', 1: 'Tumor', 2: 'Metastatic'}
    
    # Plot each sample type with different colors and labels
    for label, color in colors.items():
        mask = labels == label
        if np.any(mask):
            ax.scatter(
                data[mask, 0], 
                data[mask, 1], 
                c=color, 
                label=color_labels[label], 
                alpha=0.7, 
                s=50
            )
    
    # Add labels and title
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(f'{title} - TCGA BRCA Gene Expression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add coefficient information if provided
    if dijkstra_info is not None:
        info_text = f"Sqrt Coefficient: {dijkstra_info:.4f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save and display
    if save:
        # Save to Results folder (original location)
        os.makedirs('Results', exist_ok=True)
        plt.savefig(f'./Results/{title}.png', dpi=300, bbox_inches='tight')
        
        # Also save to figures folder for TCGA project
        os.makedirs('../../figures', exist_ok=True)
        plt.savefig(f'../../figures/{title}.png', dpi=300, bbox_inches='tight')
        print(f"Plot also saved in '../../figures/{title}.png'")
    
    if display:
        plt.show()
    
    return fig

def saveTotalLossPlots(total_losses,N,title='Loss per epoch'):
    # Plot total loss
    plt.figure()
    plt.plot((total_losses))
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    # plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    loss_graph_path = './Loss_graphs/'
    if not os.path.exists(loss_graph_path):
        os.makedirs(loss_graph_path)
    plt.savefig(loss_graph_path+str(N)+title+'.png')


### other helper functions

def printtime(name,delta_t):
    if delta_t > 120:
        print("\n"+name+": %.2f min" % (delta_t / 60))
    else:
        print("\n"+name+": %.2f sec" % delta_t)