import torch
import networkx as nx

from torch_geometric.data import Data

def load_data(file_path: str):
    """
    An toy example used as part of the DHPC example.
    This function loads the data from an given .txt file.torch_geometric
    
    Reads
    Args:
        data_dir: The directory where the data is stored.
    Returns:
        A list of tuples of strings.
    """
    # peform some checks
    assert os.path.isfile(file_path), "The file path is not a file."
    assert file_path.endswith('.txt'), "The file is not a .txt file."

    # Load the data
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = line.strip()
            entry = (entry.split(', '))
            data.append(entry)
    return data

def make_graphs(data: list):
    """
    An toy example used as part of the DHPC example.
    This function makes the graphs from the data and
    encodes the gender as a 0 or 1.
    
    Reads
    Args:
        data: A list of tuples of strings.
    Returns:
        A list of tuples of (grap,int)
    """
    # Make a vocabalary of the alphabet
    alpa = 'abcdefghijklmnopqrstuvwxyz'
    vocab = {char: i for i, char in enumerate(alpa)}


    # Convert the data to tensors
    data_graphs = []
    for name,gender in data:
        # Gender to interger, male = 0, felame = 1
        gender_idx = 1 if gender == "Female" else 0
        
        #Convert the letters to integers
        name = [vocab[char] for char in name.lower()]

        # Make the graph
        G = nx.DiGraph()

        # Add the nodes
        for i in range(len(name)):
            G.add_node(i, feature=name[i])

        # Add the edges
        for i in range(len(name)-1):
            G.add_edge(i, i+1)

        data_graphs.append((G, gender_idx))

    return data_graphs

def make_PgG_data(data_graphs: list):
    """
    An toy example used as part of the DHPC example.

    This function takes the graphs and converts them to
    Pytorch Geometric Data objects.

    args:
        data_graphs: A list of tuples of (graph, int)

    returns:
        A list of Pytorch Geometric Data objects
    """

    PyG_data = []

    for graph, label in data_graphs:
        #Convert the label to a tensor
        y = torch.tensor([label], dtype=torch.long)

        # Get the node features
        node_features = torch.tensor([graph.nodes[i]['feature'] for i in graph.nodes],dtype=torch.float).view(-1,1)
        # Get the edge indices
        edge_indices = torch.tensor([[i,j] for i,j in graph.edges],dtype=torch.long).transpose(0,1)
        
        data_object = Data(x=node_features, edge_index=edge_indices, y=y)

        PyG_data.append(data_object)
    
    return PyG_data

def preprocess_data(read_path: str, save_path: str):
    """
    An toy example used as part of the DHPC example.

    This function is used to preprocess the data.
    It loads the data, makes the graphs and converts
    them to Pytorch Geometric Data objects. It then
    saves the Pytorch Geometric Data objects to disk.

    args:
        read_path: The path to the data
        save_path: The path to save the data to

    """

    # Get the file names that have .txt in them
    files = [file for file in os.listdir(read_path) if file.endswith('.txt')]

    #Check that the save path exists, if not make it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Loop over the files
    for file in files:
        # Load the data
        data = load_data(os.path.join(read_path, file))
        # Make the graphs
        data_graphs = make_graphs(data)
        # Make the Pytorch Geometric Data objects
        data_PyG = make_PgG_data(data_graphs)

        # Save the Pytorch Geometric Data objects
        torch.save(data_PyG, os.path.join(save_path, file.replace('.txt', '.pt')))
        
        print(f"Processed {file} and saved it to {save_path}")

   

if __name__ == "__main__":
    # The path to the data
    read_path = './raw_data'
    # The path to save the data to
    net_id = 'kdegens'
    save_path = f'/scratch/{net_id}/.local $HOME/.local/processed_data'

    preprocess_data(read_path, save_path)

