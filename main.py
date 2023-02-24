import os
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.loader import DataLoader

class Toy_model(nn.Module):
    """
    This class defines the model used in the DHPC example.
    It is a simple GCN model, with three GCN layers and
    two output layers.

    """
    def __init__(self,):
        super(Toy_model, self).__init__()
        
        # Define three GCN layers
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        #Define the output layer
        self.out_1 = nn.Linear(32, 16)
        self.out_2 = nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Pass the data through the GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # Use global max pooling
        x = global_max_pool(x, data.batch)

        # Pass the data through the output layer
        x = F.relu(self.out_1(x))
        x = self.out_2(x)

        return  F.log_softmax(x, dim=1)


def Epoch(model, optimizer, train_data, Eval_data, device):
    """
    An toy example used as part of the DHPC example.

    This function while perform one epoch on both the
    training and evaluation data. It will return the
    loss and accuracy for both sets.

    args:
        model: The model to train
        optimizer: The optimizer to use
        train_data: The training data
        Eval_data: The evaluation data
        device: The device to use

    returns:
        train_loss: The loss on the training data
        train_acc: The accuracy on the training data
        eval_loss: The loss on the evaluation data
        eval_acc: The accuracy on the evaluation data

    """

    # Set the model to training mode
    model.train()

    # Set the loss and accuracy to 0
    train_loss = 0
    train_acc = []

    # Loop over the training data
    for data in train_data:

        # Send the data to the device
        data = data.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Get the predictions
        out = model(data)

        # Calculate the loss
        loss = F.nll_loss(out, data.y)

        # Calculate the accuracy
        pred = out.max(dim=1)[1]
        acc = out.max(dim=1)[1].eq(data.y).tolist()

        # Backpropagate the loss
        loss.backward()
        optimizer.step()

        # Update the loss and accuracy
        train_loss += loss.item()
        train_acc += acc

    # Set the model to evaluation mode
    model.eval()

    # Set the loss and accuracy to 0
    eval_loss = 0
    eval_acc = []

    # Loop over the evaluation data
    for data in Eval_data:

        # Send the data to the device
        data = data.to(device)

        # Get the predictions
        out = model(data)

        # Calculate the loss
        loss = F.nll_loss(out, data.y)

        # Calculate the accuracy
        pred = out.max(dim=1)[1]
        acc = out.max(dim=1)[1].eq(data.y).tolist()

        # Update the loss and accuracy
        eval_loss += loss.item()
        eval_acc += acc

    # Calculate the average loss and accuracy
    train_loss /= len(train_data)
    train_acc = sum(train_acc) / len(train_acc)
    eval_loss /= len(Eval_data)
    eval_acc = sum(eval_acc) / len(eval_acc)

    return train_loss, train_acc, eval_loss, eval_acc


def main(Data_path: str):
    """"
    An toy example used as part of the DHPC example.

    This function will train a simple GCN model on the
    toy dataset.

    """

    # Define the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    train_data = DataLoader(torch.load(os.path.join(Data_path, 'train.pt')), batch_size=4, shuffle=True)
    Eval_data = DataLoader(torch.load(os.path.join(Data_path, 'eval.pt')), batch_size=2, shuffle=False)

    # Define the model
    model = Toy_model().to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Loop over the epochs
    for epoch in range(1,101):

        # Perform one epoch
        train_loss, train_acc, eval_loss, eval_acc = Epoch(model, optimizer, train_data, Eval_data, device)

        # Log the results
        wandb.log({'Train Loss': train_loss, 'Train Acc': train_acc, 'Eval Loss': eval_loss, 'Eval Acc': eval_acc, 'Epoch': epoch})

        # Print the results
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}')

        # Sleep for 2 seconds, to simulate a long running job
        time.sleep(2)

if __name__ == '__main__':
    # Get the path to the data
    net_id = 'kdegens'
    Data_path = f'/scratch/{net_id}/.local $HOME/.local/processed_data'

    # Initialize wandb
    wandb.init(project='DHPC_example')
    
    #Use wandb to send a slack message
    wandb.alert(title="Job started",
                text="Your job at the DHPC has started, fingers crossed it doesn't crash!",
                level="info")

    # Run the main function
    main(Data_path)