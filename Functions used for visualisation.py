import matplotlib.pyplot as plt
import numpy as np


import xarray as xr
import netCDF4 as nc
import dfm_tools as dfmt

from matplotlib.widgets import Slider
import IPython
# import ipympl



# %matplotlib widget


def Calculate_Q(trih, boundary=2, spinup=0):
    ds_his = xr.open_mfdataset(trih, preprocess=dfmt.preprocess_hisnc, decode_timedelta=True)
    sed_vol = ds_his.SBTRC[:,0, boundary].values[-1]    # get tot sed volume at the last time step
    a = (ds_his.time[-1] - ds_his.time[0]) / 1e9        # Total duration of the simulation converted to seconds
    b = (a - (spinup * 60)).astype(int)                 # don't take spin up into consideration
    return (sed_vol / b).values



def PlotCrossXDIR(trim, N_coords, elevation_var= "DPS"):
    ''' Plots the horzontal cross-sections in one graph'''
    # Opendataset and extract relevant variables
    dataset = nc.Dataset(trim, mode='r')
    x = dataset.variables["XCOR"][:]
    y = dataset.variables["YCOR"][:]
    
    # Exclude outer ring
    xc_inner = x[1:-1, 1:-1]
    yc_inner = y[1:-1, 1:-1]
    
    # Last time step, also excluding outer ring
    elevation_last = dataset.variables[elevation_var][-1, 1:-1, 1:-1]
    
    # colors for plotting
    colors = plt.cm.viridis(np.linspace(0, 1, len(N_coords)))
    
    # Create figure
    plt.figure(figsize=(10, 3))
    
    # Loop over cross-sectionss, and plot them     
    for i, N in enumerate(N_coords):
        z_cross = elevation_last[:, N]
        x_cross = xc_inner[:, N]
        plt.plot(x_cross, z_cross, color=colors[i], label=f'Crosssection at M= {N}')
        
    plt.xlabel('X Cross-section')
    plt.ylabel('Elevation')
    plt.title(f'Dataset: {trim}')
    plt.grid()
    plt.legend()
    
    dataset.close()
    return plt.show()

def PlotCrossYDIR(trim, M_coords, elevation_var= "DPS"):
    ''' Plots the vertical cross-sections in one graph'''
    # Opendataset and extract relevant variables
    dataset = nc.Dataset(trim, mode='r')
    x = dataset.variables["XCOR"][:]
    y = dataset.variables["YCOR"][:]
    
    # Exclude outer ring
    xc_inner = x[1:-1, 1:-1]
    yc_inner = y[1:-1, 1:-1]
    
    # Last time step, also excluding outer ring
    elevation_last = dataset.variables[elevation_var][-1, 1:-1, 1:-1]
    
    # colors for plotting
    colors = plt.cm.viridis(np.linspace(0, 1, len(M_coords)))
    
    # Create figure
    plt.figure(figsize=(10, 3))
    
    # Loop over cross-sectionss, and plot them     
    for i, M in enumerate(M_coords):
        z_cross = elevation_last[M,:]
        y_cross = yc_inner[M, :]
        plt.plot(y_cross, z_cross,color=colors[i], label=f'Crosssection at M= {M}')
        
    plt.xlabel('Y Cross-section')
    plt.ylabel('Elevation')
    plt.title(f'Dataset: {trim}')
    plt.grid()
    plt.legend()
    
    dataset.close()
    return plt.show()

def PlotHorizontalCrossSection(trim2, N=40, elevation_var = "DPS"):
    '''
    Plots a cross-section of elevation data and the full elevation map with a slider to adjust the cross-section.
    
    Parameters:
    - trim2: str, the path to the NetCDF dataset
    - N: int, initial cross-section index (default is 40)
    '''
    # Load dataset
    dataset = nc.Dataset(trim2, mode='r')
    x = dataset.variables["XCOR"][:]
    y = dataset.variables["YCOR"][:]

    # Exclude outer ring
    xc_inner = x[1:-1, 1:-1]
    yc_inner = y[1:-1, 1:-1]

    # Last time step, also excluding outer ring
    elevation_last = dataset.variables[elevation_var][-1, 1:-1, 1:-1]

    # Initial cross-section data
    z_cross = elevation_last[:, N]
    x_cross = xc_inner[:, N]
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(bottom=0.25, hspace=0.5)

    # Plot the initial cross-section on ax1
    l, = ax1.plot(x_cross, z_cross, lw=2)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylim(-6, 11)
    ax1.invert_yaxis()
    ax1.set_ylabel('Elevation')
    ax1.set_title(f'Elevation Cross-Section at N={N}')
    ax1.grid()

    # Plot the full elevation map on ax2
    c = ax2.pcolormesh(xc_inner, yc_inner, elevation_last, shading='auto', cmap='viridis', vmin=-10, vmax=5)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Full Elevation Map (Last Timestep)')
    fig.colorbar(c, ax=ax2, label='Elevation')

    # Add a vertical line on the elevation map
    hline = ax2.axhline(y=N*50, color='r', linestyle='--', label='Cross-Section')

    # Slider axis for cross-section plot
    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, label='Cross-section Index', valmin=0, valmax=len(xc_inner)-1, valinit=N, valstep=1)

    # Update function for the slider
    def update(val):
        cross = int(slider.val)  # Get the cross-section index from the slider (ensure it's an integer)
        z_cross = elevation_last[:, cross]  # Update the elevation data at the new M index
        l.set_ydata(z_cross)  # Update y-data of the plot with the new cross-section
        ax1.set_title(f'Elevation Cross-Section at N={N}')  # Update title with the new M index

        hline.set_ydata(cross*50)  # Update the x position of the vertical line

        fig.canvas.draw_idle()  # Refresh the plot

    # Link the update function to the slider
    slider.on_changed(update)
    return plt.show()


def PlotVerticalCrossSection(trim2, M=40, elevation_var = "DPS"):
    '''
    Plots a horizontal cross-section of elevation data and the full elevation map with a slider to adjust the cross-section.
    
    Parameters:
    - trim2: str, the path to the NetCDF dataset
    - M: int, initial cross-section index (default is 40)
    '''
    # Load dataset
    dataset = nc.Dataset(trim2, mode='r')
    x = dataset.variables["XCOR"][:]
    y = dataset.variables["YCOR"][:]

    # Exclude outer ring
    xc_inner = x[1:-1, 1:-1]
    yc_inner = y[1:-1, 1:-1]

    # Last time step, also excluding outer ring
    elevation_last = dataset.variables[elevation_var][-1, 1:-1, 1:-1]

    # Initial cross-section data
    z_cross = elevation_last[M, :]
    y_cross = yc_inner[M, :]

    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(bottom=0.25, hspace=0.5)

    # Plot the initial cross-section on ax1
    l, = ax1.plot(y_cross, z_cross, lw=2)
    ax1.set_xlabel('Y Coordinate')
    ax1.set_ylim(-6, 11)
    ax1.invert_yaxis()
    ax1.set_ylabel('Elevation')
    ax1.set_title(f'Elevation Cross-Section at M={M}')
    ax1.grid(True)

    # Plot the full elevation map on ax2
    c = ax2.pcolormesh(xc_inner, yc_inner, elevation_last, shading='auto', cmap='viridis', vmin=-10, vmax=5)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Full Elevation Map (Last Timestep)')
    fig.colorbar(c, ax=ax2, label='Elevation')

    # Add a vertical line on the elevation map
    vline = ax2.axvline(x=M*50, color='r', linestyle='--', label='Cross-Section')

    # Slider axis for cross-section plot
    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, label='Cross-section Index', valmin=0, valmax=len(yc_inner)-1, valinit=M, valstep=1)

    # Update function for the slider
    def update(val):
        cross = int(slider.val)  # Get the cross-section index from the slider (ensure it's an integer)
        z_cross = elevation_last[cross, :]  # Update the elevation data at the new M index
        l.set_ydata(z_cross)  # Update y-data of the plot with the new cross-section
        ax1.set_title(f'Elevation Cross-Section at M={cross}')  # Update title with the new M index

        vline.set_xdata(cross* 50)  # Update the x position of the vertical line
        fig.canvas.draw_idle()  # Refresh the plot

    # Link the update function to the slider
    slider.on_changed(update)
    
    return plt.show()


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def plot_bathymetry_with_colormap(nc_file, time_index=-1, vmin=-15, vmax=5, vcenter=0):
    """
    Plot bathymetry from a NetCDF file with a terrain-style colormap that emphasizes sea level.
    
    Parameters:
        nc_file (str): Path to the NetCDF file.
        time_index (int): Time index to plot. Default is -1 (last).
        vmin (float): Minimum value for color scale.
        vmax (float): Maximum value for color scale.
        vcenter (float): Value where color scale is centered (typically sea level = 0).
    """
    
    # Load the NetCDF file
    dataset = xr.open_dataset(nc_file)

    # Extract coordinates and depth data
    xc = dataset["XCOR"].values
    yc = dataset["YCOR"].values
    depth = dataset["DPS"]

    # Trim the edges
    xc_inner = xc[1:-1, 1:-1]
    yc_inner = yc[1:-1, 1:-1]
    depth_inner = depth[:, 1:-1, 1:-1] * -1  # Flip sign for elevation

    # Time values
    time_values = dataset["time"].values
    depth_at_t = depth_inner[time_index, :, :]

    # Define sharp sea-level-transition colormap
    colors = [
        (0.00, "#000066"),   # deep water
        (0.10, "#0000ff"),   # blue
        (0.30, "#00ffff"),   # cyan
        (0.40, "#00ffff"),  # water edge
        (0.50, "#ffffcc"),  # land edge
        (0.60, "#ffcc00"),   # orange
        (0.75, "#cc6600"),   # brown
        (0.90, "#228B22"),   # green
        (1.00, "#006400"),   # dark green
    ]
    terrain_like = LinearSegmentedColormap.from_list("custom_terrain", colors)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(xc_inner, yc_inner, depth_at_t, shading='auto', cmap=terrain_like, norm=norm)

    ax.set_title(f'Depth at Time = {time_values[time_index]}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    cb = fig.colorbar(c, ax=ax, orientation='vertical', label='Elevation relative to MSL [m]')
    cb.set_ticks([vmin, -10, -5, -2, 0, 2, vmax])
    cb.ax.set_yticklabels([str(vmin), '-10', '-5', '-2', '0', '2', str(vmax)])

    plt.tight_layout()
    plt.show()

    dataset.close()







