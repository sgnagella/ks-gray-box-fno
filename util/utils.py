import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import pickle

FONTSIZE = 16 

plt.rcParams.update({
    "pdf.fonttype":42, 
    "ps.fonttype":42,
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset":"custom",
    # "mathtext.rm": "sans", 
    "font.size": FONTSIZE,
    "axes.linewidth": 1.25, 
    "xtick.labelsize": FONTSIZE -2, 
    "ytick.labelsize": FONTSIZE -2,
    "legend.fontsize": FONTSIZE -3})

ASPECT_RATIO = 1.25 # width/height
MARKERSIZE = 8
CAPSIZE = 4
LINEWIDTH = 1.5
FIG_WIDTH = 5 # inches
FIG_HEIGHT = FIG_WIDTH/ASPECT_RATIO
MARKEREDGEWIDTH = 1

def segment_data(*, data, nLengthTraj): 
    """
        Segment the data into trajectories of length nLengthTraj.
        Output is list of segmented trajectories and their corresponding 
        scales (min and max)
    """

    # Sizes
    N = data.shape[0]
    nx = data.shape[1]

    # Number of trajectories
    nTraj = N//nLengthTraj

    # List to store the segmented data
    data_list = []
    u_mins = []
    u_maxs = []

    # Loop through all the trajectories.
    for i in range(nTraj):
        # Get the current trajectory.
        traj = data[i*nLengthTraj:(i+1)*nLengthTraj, :]
        umin, umax = np.min(traj), np.max(traj)
        u_mins.append(umin)
        u_maxs.append(umax)

        # Append the trajectory to the list.
        data_list += [traj]

    # Return the list of segmented data.
    return data_list , {"umin": np.array(u_mins), "umax": np.array(u_maxs)}

# def get_uscales(traj_list): 
#     """
#         Get the scales for the data trajectories.
#         Can use this function in case of performing scaling in different space.
#     """

#     # Sizes.
#     nTraj = len(traj_list)

#     # Lists to store the min and max values.
#     u_mins = []
#     u_maxs = []

#     # Loop through all the data trajectories.
#     for ii in range(nTraj):
#         # Get the current trajectory.
#         traj = traj_list[ii]
#         umin, umax = np.min(traj), np.max(traj)
#         u_mins.append(umin)
#         u_maxs.append(umax)

#     # Return the scales.
#     return {"umin": np.array(u_mins), "umax": np.array(u_maxs)}

# def 

def get_train_val_data(*, data_list, uscales,
                       nTrainTraj, nTrainValTraj):
    """ Scale all the data trajectories using the provided
        scaling dictionary for training and validation of the
        black-box and hybrid models.
    """

    # Extract the min and max scalings
    umin, umax = uscales["umin"], uscales["umax"]

    # Sizes.
    nTraj = len(data_list)

    # Lists to store data.
    # The xseq is collected mainly to check predictions of the unmeasured
    # grey-box states during the training.
    useq = []
    useq0 = []

    # Loop through all the data trajectories in the data list.
    for ii, data in enumerate(data_list):
        # Scale data.
        # u = (data - umin[ii])/(umax[ii] - umin[ii])
        u = 2*(data - umin[ii])/(umax[ii] - umin[ii]) - 1 # Scale to [-1, 1]
        useq += [u]
        useq0 += [u[0][None,:]]

    useq = np.asarray(useq)
    useq0 = np.asarray(useq0)

    # print(f"useq shape: {useq.shape}, useq0 shape: {useq0.shape}")

    # Data dictionary for the training trajectory.
    train_data = dict(useq=useq[:nTrainTraj],
                      useq0=useq0[:nTrainTraj], 
                      uscales={"umin": umin[:nTrainTraj], "umax": umax[:nTrainTraj]})

    # Data dictionary for the validation trajectories.
    useq_val = useq[nTrainTraj:nTrainTraj + nTrainValTraj]
    useq0_val = useq0[nTrainTraj:nTrainTraj + nTrainValTraj]
    val_data = dict(useq=useq_val,
                    useq0=useq0_val, 
                    uscales={"umin": umin[nTrainTraj:nTrainTraj + nTrainValTraj], 
                             "umax": umax[nTrainTraj:nTrainTraj + nTrainValTraj]})

    # Data dictionary for the testing trajectory.
    useq_test = useq[nTrainTraj + nTrainValTraj:]
    useq0_test = useq0[nTrainTraj + nTrainValTraj:]
    test_data = dict(useq=useq_test,
                     useq0=useq0_test,
                     uscales={"umin": umin[nTrainTraj + nTrainValTraj:], 
                              "umax": umax[nTrainTraj + nTrainValTraj:]})

    # Return.
    return train_data, val_data, test_data


def generate_info_dict(*, train_ratio, val_ratio, traj_list, uscales):
    """ Generate the dictionary containing the training data, validation data, 
        and the number of training trajectories.
    """
    nTraj = len(traj_list)
    num_train = int(train_ratio * nTraj)
    num_val = int(val_ratio * nTraj)
    num_test = nTraj - num_train - num_val

    # Load the KSDataset class and create the training, validation, and test datasets
    info = {
        "training_data": {"train": traj_list}, 
        "uscales": {"train": uscales}, 
        "numTrainTraj": {"train": (num_train, num_val)}
        }

    return info

def animate_prediction_vs_truth(*, x, predictions, truth, save=False, filename=None):
    """ Animate the results of the predictions.
    """

    # Create a figure and axis for the animation
    plt.close('all')
    fig, ax = plt.subplots(num=1, clear=True)
    l1 = ax.plot(x, predictions[0], lw=2, color='blue', label='Prediction')
    l2 = ax.plot(x, truth[0], lw=2, color='red', label='Truth')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-10,10)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.legend(loc='upper right')
    ax.grid(True)

    # Create a text object for the timer
    timer_text = ax.text(0.15, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12, color='black')

    # Animation update function
    def update(frame):
        l1[0].set_ydata(predictions[frame])
        l2[0].set_ydata(truth[frame])

        # Update the timer
        timer_text.set_text(r'$\tau = {}$'.format(frame))
        
        return ax,

    # Create and display the animation
    ani = FuncAnimation(fig, update, frames=predictions.shape[0], blit=False, interval=250)
    if save:
        os.makedirs('figures', exist_ok=True)
        if '.gif' in filename.split('.'):
            filename = filename.split('.')[0]
        ani.save(os.path.join('figures', filename + '.gif'), dpi=200, writer='pillow')
    
    plt.show()
    return None

def export_dict(info_dict, filename):
    exp_dict = info_dict.copy()
    with open(filename, 'wb') as handle:
        pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None