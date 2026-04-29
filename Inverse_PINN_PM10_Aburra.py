import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from time import time
import matplotlib.pyplot as plt
import os

seed = int(time())
np.random.seed(seed)
tf.random.set_seed(seed)

#%%
# USER INPUTS
#############################

## general parameters
tol = 1e-8 # tolerance for stopping training
save_fig = True # save figures or not

## Data and test case
testcase = "2d_pm10"
data_perturbation = 0.0 # perturbation for the data

## Parameters to train
train_parameters = True
# Initial guesses for the coordinates we want to find
initial_x_s = 0.5 
initial_y_s = 0.5

learning_rate_param = 1e-2 # learning rate of the parameters
train_parameters_epoch = 1000 # epoch after which train the parameters

## Loss function weights (will be normalised afterwards)
pde_weight = 1.0      # penalty for the PDE
data_weight = 1.0     # penalty for the data fitting
ic_weight = 10.0      # penalty for the initial condition
bc_weight = 10.0      # penalty for the boundary condition

# NN training parameters
epochs = 3000          # number of epochs
epoch_print = 100      # print the loss every epoch_print epochs

learning_rate = 5e-3   # learning rate for the network weights
learning_rate_decay_factor = 0.98 # decay factor for the learning rate
learning_rate_step = 100

learning_rate_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    [learning_rate_step*(i+1) for i in range(int(epochs/learning_rate_step))],
    [learning_rate*learning_rate_decay_factor**i for i in range(int(epochs/learning_rate_step)+1)])

## NN architecture
num_hidden_layers = 8
num_neurons = 30
activation = 'tanh'

#%%
# Load data
#############################

datafolder = "data_2d"
resultfolder = 'results_2d_pm10'
os.makedirs(resultfolder, exist_ok=True)

# Load data as pandas dataframes
try:
    x_grid = pd.read_csv(f'{datafolder}/x.csv', header=None, dtype=np.float32)
    y_grid = pd.read_csv(f'{datafolder}/y.csv', header=None, dtype=np.float32)
    t_grid = pd.read_csv(f'{datafolder}/t.csv', header=None, dtype=np.float32)
    c_data = pd.read_csv(f'{datafolder}/c.csv', header=None, dtype=np.float32)
    p_true = pd.read_csv(f'{datafolder}/p_true.csv', header=None, dtype=np.float32)
except FileNotFoundError:
    print(f"Data not found in {datafolder}. Please run generate_synthetic_data_2d.py first.")
    exit()

nx = len(x_grid)
ny = len(y_grid)
nt = len(t_grid)

# perturb data randomly
c_data = c_data * (1 + data_perturbation * np.random.randn(c_data.size).reshape(c_data.shape)) 

x_grid = np.array(x_grid).squeeze()
y_grid = np.array(y_grid).squeeze()
t_grid = np.array(t_grid).squeeze()

# Create meshgrids
Xgrid, Ygrid, Tgrid = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')

x_data = Xgrid.flatten().astype(np.float32)
y_data = Ygrid.flatten().astype(np.float32)
t_data = Tgrid.flatten().astype(np.float32)
c_data_flat = c_data.values.flatten().astype(np.float32)

# Convert data to tensor
x_tf = tf.expand_dims(tf.convert_to_tensor(x_data), -1)
y_tf = tf.expand_dims(tf.convert_to_tensor(y_data), -1)
t_tf = tf.expand_dims(tf.convert_to_tensor(t_data), -1)
c_tf = tf.expand_dims(tf.convert_to_tensor(c_data_flat), -1)

# Physics parameters (from true parameters or constants)
# d, u, v, sigma, x_s, y_s, Q, sigma_s
p_values = p_true.values.squeeze()
d_val = p_values[0]
u_val = p_values[1]
v_val = p_values[2]
sigma_val = p_values[3]
Q_val = p_values[6]
sigma_s_val = p_values[7]

true_x_s = p_values[4]
true_y_s = p_values[5]

print(f"True parameters: x_s = {true_x_s:.2f}, y_s = {true_y_s:.2f}")

# Trainable parameters (coordinates of the source)
x_s = keras.Variable([initial_x_s], trainable=train_parameters, dtype=tf.float32)
y_s = keras.Variable([initial_y_s], trainable=train_parameters, dtype=tf.float32)
params = [x_s, y_s]
nparam = len(params)

#%%
# Define the PINN model and loss functions
#############################

def pinn_model():
    t_input = keras.Input(shape=(1,))
    x_input = keras.Input(shape=(1,))
    y_input = keras.Input(shape=(1,))

    output_c = layers.concatenate([t_input, x_input, y_input]) 
    
    for _ in range(num_hidden_layers):
        output_c = layers.Dense(num_neurons, activation=activation, kernel_initializer='glorot_normal')(output_c)
    
    output_c = layers.Dense(1)(output_c)

    return keras.Model(inputs=[t_input, x_input, y_input], outputs=output_c)

@tf.function(reduce_retracing=True)
def custom_loss(inputs, model):
    tt, xx, yy, cc = inputs

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xx)
        tape2.watch(yy)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(tt)
            tape1.watch(xx)
            tape1.watch(yy)
            
            c_model = model([tt, xx, yy])
            
        c_t = tape1.gradient(c_model, tt)
        c_x = tape1.gradient(c_model, xx)
        c_y = tape1.gradient(c_model, yy)
        
    c_xx = tape2.gradient(c_x, xx)
    c_yy = tape2.gradient(c_y, yy)
    del tape1
    del tape2

    # Source term S(x,y)
    S = Q_val * tf.exp(-((xx - x_s)**2 + (yy - y_s)**2) / (2 * sigma_s_val**2))

    # PDE residual
    pde_residual = c_t + u_val * c_x + v_val * c_y - d_val * (c_xx + c_yy) + sigma_val * c_model - S

    pde_loss = tf.reduce_mean(pde_residual ** 2)
    data_fitting_loss = tf.reduce_mean((c_model - cc) ** 2)
    
    # We simplify BC and IC since they are included in data_fitting_loss if using the full grid
    # For a real scenario, we'd extract the indices for boundaries and t=0. 
    # Here we just use subsets of data.
    return [pde_loss, data_fitting_loss]

model = pinn_model()
trainable = model.trainable_variables
if train_parameters:
    for p in params:
        trainable.append(p)

#%%
# Train the NN
#############################

optimizer = keras.optimizers.Adam(learning_rate=learning_rate_schedule, amsgrad=True)

losses = np.zeros((epochs, 3))
param_values = np.zeros((epochs, nparam))

t0 = time()
for epoch in range(epochs):
    
    if train_parameters:
        # tanh transition from 0 to 1 for curriculum learning
        param_data_factor = (np.tanh(10*(epoch-epochs/2-train_parameters_epoch)/epochs)+1)/2
        param_data_factor *= (epoch > train_parameters_epoch)
    else:
        param_data_factor = 1.0

    weights = [pde_weight, data_weight * param_data_factor]
    weights = [w/sum(weights) for w in weights] # normalise
    
    with tf.GradientTape(persistent=True) as tape:
        loss0 = custom_loss([t_tf, x_tf, y_tf, c_tf], model)
        loss = [l * w for l, w in zip(loss0, weights)]
        total_loss = sum(loss)
        
    gradients = tape.gradient(total_loss, trainable)
    del tape
    
    if train_parameters:
        # scale parameter gradients
        for i in range(-nparam, 0):
            if gradients[i] is not None:
                gradients[i] *= learning_rate_param * param_data_factor
    
    optimizer.apply_gradients(zip(gradients, trainable))    
    
    losses[epoch, 0] = loss0[0].numpy()
    losses[epoch, 1] = loss0[1].numpy()
    losses[epoch, 2] = total_loss.numpy()
    
    param_values[epoch, 0] = x_s.numpy()[0]
    param_values[epoch, 1] = y_s.numpy()[0]
    
    if epoch % epoch_print == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}/{epochs}, Total Loss: {losses[epoch,2]:.4e}, PDE: {losses[epoch,0]:.4e}, Data: {losses[epoch,1]:.4e}")
        print(f"factor={param_data_factor:.2e}, x_s={param_values[epoch,0]:.4f} (True: {true_x_s}), y_s={param_values[epoch,1]:.4f} (True: {true_y_s})")

print(f'\nTotal training CPU time: {time() - t0:.2f} seconds')

#%%
# Plottings
#############################

plt.figure()
plt.semilogy(losses[:, 0], label='PDE Loss (unweighted)')
plt.semilogy(losses[:, 1], label='Data Loss (unweighted)')
plt.semilogy(losses[:, 2], label='Total Weighted Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
if save_fig:
    plt.savefig(f'{resultfolder}/loss_history.png')
plt.close()

plt.figure()
plt.plot(param_values[:, 0], label='Predicted x_s')
plt.plot(param_values[:, 1], label='Predicted y_s')
plt.axhline(true_x_s, color='blue', linestyle='--', label='True x_s')
plt.axhline(true_y_s, color='orange', linestyle='--', label='True y_s')
plt.xlabel('Epoch')
plt.ylabel('Coordinate')
plt.legend()
plt.grid()
if save_fig:
    plt.savefig(f'{resultfolder}/parameters_history.png')
plt.close()

print(f"Results saved in {resultfolder}/")
