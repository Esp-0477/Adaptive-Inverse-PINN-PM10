import numpy as np
import pandas as pd
import os

def generate_data():
    print("Generating 2D synthetic data for PM10 Advection-Diffusion-Reaction...")
    
    # Grid parameters
    nx, ny, nt = 30, 30, 50
    Lx, Ly, T = 1.0, 1.0, 1.0
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    t = np.linspace(0, T, nt)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Physics parameters
    D = 0.05
    u = 0.1
    v = 0.1
    sigma = 0.0  # reaction constant
    
    # Source parameters (true values we want to find)
    x_s_true = 0.7
    y_s_true = 0.3
    Q = 5.0
    sigma_s = 0.05
    
    # Source function
    def source(X, Y):
        return Q * np.exp(-((X - x_s_true)**2 + (Y - y_s_true)**2) / (2 * sigma_s**2))
    
    S = source(X, Y)
    
    # Finite difference solution
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = T / (nt - 1)
    
    # CFL check for explicit Euler
    assert D * dt / dx**2 + D * dt / dy**2 < 0.5, "Explicit Euler might be unstable, reduce dt or increase dx/dy"
    
    c = np.zeros((nx, ny, nt))
    
    # Time stepping
    for n in range(0, nt - 1):
        c_current = c[:, :, n]
        
        # Central differences for second derivatives
        c_xx = np.zeros_like(c_current)
        c_yy = np.zeros_like(c_current)
        
        c_xx[1:-1, :] = (c_current[2:, :] - 2*c_current[1:-1, :] + c_current[:-2, :]) / dx**2
        c_yy[:, 1:-1] = (c_current[:, 2:] - 2*c_current[:, 1:-1] + c_current[:, :-2]) / dy**2
        
        # Upwind differences for first derivatives (assuming u, v > 0)
        c_x = np.zeros_like(c_current)
        c_y = np.zeros_like(c_current)
        
        c_x[1:, :] = (c_current[1:, :] - c_current[:-1, :]) / dx
        c_y[:, 1:] = (c_current[:, 1:] - c_current[:, :-1]) / dy
        
        # Update
        c_next = c_current + dt * (D * (c_xx + c_yy) - u * c_x - v * c_y - sigma * c_current + S)
        
        # Boundary conditions (Dirichlet zero)
        c_next[0, :] = 0; c_next[-1, :] = 0
        c_next[:, 0] = 0; c_next[:, -1] = 0
        
        c[:, :, n + 1] = c_next

    # Save to data directory
    data_dir = 'data_2d'
    os.makedirs(data_dir, exist_ok=True)
    
    # Save coordinates and parameters
    pd.DataFrame(x).to_csv(f'{data_dir}/x.csv', index=False, header=False)
    pd.DataFrame(y).to_csv(f'{data_dir}/y.csv', index=False, header=False)
    pd.DataFrame(t).to_csv(f'{data_dir}/t.csv', index=False, header=False)
    
    # Save data: Flatten space but keep time? No, let's flatten the whole grid or save time in 3rd dim.
    # We will reshape C into a flat array, same as PINN expects.
    # Actually, saving c as a flattened 1D array across all dimensions.
    # The original script reads c.csv as a flat array and reshapes to (nt, nx).
    # For 2D, we will reshape to (nt, nx, ny).
    # We must flatten in the order: t (outer), x, y (inner).
    # Numpy meshgrid indexing is 'ij', so C shape is (nx, ny, nt).
    # Let's transpose to (nt, nx, ny) before flattening.
    c_transposed = np.transpose(c, (2, 0, 1))
    pd.DataFrame(c_transposed.flatten()).to_csv(f'{data_dir}/c.csv', index=False, header=False)
    
    # Save true parameters
    # d, u, v, sigma, x_s, y_s, Q, sigma_s
    params = [D, u, v, sigma, x_s_true, y_s_true, Q, sigma_s]
    pd.DataFrame(params).to_csv(f'{data_dir}/p_true.csv', index=False, header=False)
    
    print(f"Data successfully generated in {data_dir}/")
    print(f"True coordinates of PM10 source: x_s={x_s_true}, y_s={y_s_true}")

if __name__ == '__main__':
    generate_data()
