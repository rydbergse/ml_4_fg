import torch
import timeit
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timeit
import scipy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

class DNN: 

    def __init__(self, dims, requires_grad = False):
        """Initialize Dense Neural Network

        Parameters
        ----------
        dims : array-like
            Array specifying number of neurons in (i.e. dimension of) each layer. 
            dims[0] is the input dimension, last element dims[-1] is the output dimension. The 
            number of hidden layers is len(dims)-1. 
        device 
            The torch device to use. 
        requires_grad : bool
            Set to true to check calculations against pytorch's own autograd 
            (or to False to save 
        """
        self.weights = {}
        self.intercepts = {}
        self.num_hidden_layers = len(dims) - 2
        self.dloss_dweights = {}
        self.dloss_dintercept = {}
        
        for layer_idx in range(len(dims)-1): 
            
            self.weights[layer_idx] = torch.randn(dims[layer_idx:layer_idx+2]) * np.sqrt(2. / dims[layer_idx])
            if requires_grad:
                self.weights[layer_idx].requires_grad = True
                
            self.intercepts[layer_idx] = torch.zeros(dims[layer_idx+1], requires_grad = requires_grad)

    def loss_and_gradient(self, X, y, obs_weights, backprop = True):
        
        hidden = {} 
        hidden[-1] = X 
        
        # forward
        for layer_idx in range(self.num_hidden_layers + 1): 
            g = hidden[layer_idx - 1] @ self.weights[layer_idx] + self.intercepts[layer_idx]
            hidden[layer_idx] = torch.relu(g) if (layer_idx < self.num_hidden_layers) else g
        output = hidden[self.num_hidden_layers]
        err = y - output
        loss = .5 * torch.sum(obs_weights * err * err) / obs_weights.sum()

        # backpropagation
        if backprop: 
            dloss_dhidden = {} 
            dloss_dhidden[self.num_hidden_layers] = - obs_weights * err / obs_weights.sum() 
            for layer_idx in range(self.num_hidden_layers, -1, -1):
                if layer_idx < self.num_hidden_layers: 
                    dloss_dg = dloss_dhidden[layer_idx] * (hidden[layer_idx]> 0).float()
                else: 
                    dloss_dg = dloss_dhidden[layer_idx] 
                self.dloss_dweights[layer_idx] = hidden[layer_idx-1].transpose(0,1) @ dloss_dg 
                self.dloss_dintercept[layer_idx] =  torch.sum(dloss_dg,0) 
                dloss_dhidden[layer_idx - 1] = dloss_dg @ self.weights[layer_idx].transpose(0,1)
        
        return(loss,output)
    
    def grad_step(self, X, y, obs_weights, learning_rate = 0.001):
        loss,_ = self.loss_and_gradient(X, y, obs_weights)
        for layer_idx in range(self.num_hidden_layers + 1): 
            self.weights[layer_idx] -= learning_rate * self.dloss_dweights[layer_idx]
            self.intercepts[layer_idx] -= learning_rate * self.dloss_dintercept[layer_idx]
        return(loss)

    def cache_parameters(self):
        self.cached_weights = { k:v.detach().clone() for k,v in self.weights.items() }
        self.cached_intercepts = { k:v.detach().clone() for k,v in self.intercepts.items() }

    def recover_cache(self):
        self.weights = self.cached_weights
        self.intercepts = self.cached_intercepts

    def gradient_descent(self, X, y, obs_weights, X_val, y_val, obs_weights_val, iterations, learning_rate): 
        losses = []
        val_losses = []
        patience = 10
        patience_counter = patience
        best_val_loss = np.inf
        for it in range(iterations):
            loss = self.grad_step(X, y, obs_weights, learning_rate = learning_rate)
            losses.append(loss.item())
            val_loss,_ = self.loss_and_gradient(X_val, y_val, obs_weights_val, backprop = False)
            val_losses.append(val_loss.item())
            if val_loss < best_val_loss: 
                self.cache_parameters()
                best_val_loss = val_loss
                patience_counter = patience
            else: 
                patience_counter -= 1
                if patience_counter <= 0: 
                    print("Early stopping at iteration %i" % it)
                    self.recover_cache() # recover the best model so far
                    break
        return(losses,val_losses)
    
    

class ExpressionEvoRateDataset(torch.utils.data.Dataset):
    
    def __init__(self, expression, rate):
        self.expression = expression
        self.rate = rate
        
    def __len__(self):
        return len(self.expression)
    
    def __getitem__(self, index):
        X = self.expression[index,:]
        y = self.rate[index]
        
        return X, y
    
    
class net_rnn(nn.Module):
        
    def __init__(self, n_layers, seq_length, hidden_layers, dropout):
        super(net_rnn, self).__init__()
        
        # Define parameters
        self.seq_length = seq_length
        
        # Define recurrent layers
        self.rnn = nn.RNN(input_size=1, hidden_size=1, num_layers=n_layers, batch_first=True)
        
        # Define fully connected layers
        linear_layers = []
        for i in range(len(hidden_layers)):
            if i == (len(hidden_layers) - 1):
                linear_layers += [nn.Linear(hidden_layers[i-1], hidden_layers[i])]
            elif i == 0:
                linear_layers += [nn.Linear(seq_length, hidden_layers[i])]
            else:
                linear_layers += [nn.Linear(hidden_layers[i-1], hidden_layers[i]),
                                  nn.Dropout(dropout),
                                  nn.ELU(inplace=True)]
        self.dense_net = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), 1)
        out, hidden = self.rnn(x)
        out = out.reshape(out.size(0), out.size(1))
        net = self.dense_net(out)
        return net

class Net2(nn.Module):
    
    def __init__(self, hidden_layers, dropout):
        super(Net2, self).__init__()
        linear_layers = []
        for i in range(len(hidden_layers)-1):
            if i == (len(hidden_layers) - 2):
                linear_layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1])]
            elif i == 0:
                linear_layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1])]
            else:
                linear_layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                                  nn.Dropout(dropout),
                                  nn.ELU(inplace=True)]
        self.dense_net = nn.Sequential(*linear_layers)

    def forward(self, x):
        net = self.dense_net(x)
        return net
    
class Net(nn.Module):
    
    def __init__(self, hidden_layers, dropout):
        super(Net, self).__init__()
        linear_layers = []
        for i in range(len(hidden_layers)-1):
            if i == (len(hidden_layers) - 2):
                linear_layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1])]
            else:
                linear_layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                                  nn.Dropout(dropout),
                                  nn.ELU(inplace=True)]
        self.dense_net = nn.Sequential(*linear_layers)

    def forward(self, x):
        net = self.dense_net(x)
        return net

def run_one_epoch(train_flag, dataloader, dnn, optimizer):
    
    torch.set_grad_enabled(train_flag)
    dnn.train() if train_flag else dnn.eval()
    
    losses = []
    
    for (x,y) in dataloader:
        output = dnn(x)
        output = output.squeeze()
        loss = nn.MSELoss()(output, y)

        if train_flag:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        losses.append(loss.detach().numpy())

                          
    return( np.mean(losses), output)


def train_neural_network(dnn, train_dataloader, validation_dataloader, optimizer, n_epochs, patience, print_t=True):
    # Train
    train_mse = []
    val_mse = []
    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'dnn_checkpoint.pt'
    for epoch in range(n_epochs):
        start_time = timeit.default_timer()
        train_loss, _ = run_one_epoch(True, train_dataloader, dnn, optimizer)
        val_loss, _ = run_one_epoch(False, validation_dataloader, dnn, optimizer)
        train_mse.append(train_loss)
        val_mse.append(val_loss)

        if val_loss < best_val_loss:
            torch.save(dnn.state_dict(), check_point_filename)
            best_val_loss = val_loss
            patience_counter = patience
        else:
            patience_counter -= 1
            
        if patience_counter <= 0:
            dnn.load_state_dict(torch.load(check_point_filename))
            break
        
        elapsed = float(timeit.default_timer() - start_time)
        if print_t == True:
            print("Epoch %i took %.2fs. Train loss: %.4f. Val loss: %.4f. Patience left: %i" %
                  (epoch+1, elapsed, train_loss, val_loss, patience_counter))
    
    return dnn, train_mse, val_mse


def plot_model_predictions(y_hat, y, xlims):
    
    # Calculate correlation
    R_p,_ = scipy.stats.pearsonr(y_hat, y)
    R_sp,_ = scipy.stats.spearmanr(y_hat, y)
    RMSE = np.sqrt(mean_squared_error(y_hat, y))
    
    # Plot
    fig = plt.figure(figsize=(8,6))
    xy = np.vstack([y_hat, y])
    z = gaussian_kde(xy)(xy)
    plt.scatter(y_hat, y, c=z, s=15, cmap = 'jet', edgecolors = 'black', linewidths = 0.1)
    cb = plt.colorbar(shrink = 0.5)
    plt.plot([-4, 2], [-4, 2], 'r')
    plt.xlim((xlims[0], xlims[1]))
    plt.xlabel('Predicted evolutionary rate')
    plt.ylabel('Evolutionary rate')
    plt.title("R² pearson=%.3f, R² spearman=%.3f, RMSE=%.3f" % (R_p**2, R_sp**2, RMSE))
    
