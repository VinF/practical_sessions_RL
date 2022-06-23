import torch
import numpy as np

import skvideo.io
from skimage.transform import resize
from skimage import img_as_ubyte

from matplotlib import pyplot as plt

def preproc_state(state):
    """
    Takes numpy array from env.reset or env.step
    and converts to PyTorch Tensor, adds batch dimension and normalizes
    
    Output:
    - `p_state`: PyTorch tensor of dimensions 1x128
    """
    p_state = torch.from_numpy(state).unsqueeze(dim=0).float()
    p_state = torch.nn.functional.normalize(p_state,dim=1)
    #
    return p_state


def get_dist_plot(env,dist,support,shape=(105,80,3)):
    """
    This function renders a side-by-side RGB image (returned as numpy array)
    of the current environment state (left) next to the current predicted probability distribution
    over rewards.
    
    `env` OpenAI Gym instantiated environment object
    `dist`: A x 51 tensor , where `A` is action space dimension
    `support` vector of supports
    `shape` (RGB: width,height,channels) desired output image size
    
    Output:
    - numpy array of RGB image (w,ht,ch)
    """
    cs = ['cyan','yellow','red','green','magenta']
    fig,ax=plt.subplots(1,1)
    fig.set_size_inches(5,5)
#     for i in range(dist.shape[0]): #loop through actions
#         _ = ax.bar(support.data.numpy(),dist[i,:].data.numpy(),\
#                 label='{}'.format(env.env.get_action_meanings()[i]),alpha=0.9,color=cs[i])
#     for i in range(dist.shape[0]): #loop through actions
#         _ = ax.bar(support,np.array(dist[i,:]),\
#                 label='{}'.format(env.env.get_action_meanings()[i]),alpha=0.9,color=cs[i])
    for i in range(dist.shape[0]): #loop through actions
        _ = ax.bar(np.asarray(support.data),np.asarray(dist[i,:].data),\
                label='{}'.format(env.env.get_action_meanings()[i]),alpha=0.9,color=cs[i])
    ax.get_yaxis().set_visible(False)
    support_ = np.linspace(support.min(),support.max(),5)
    ax.set_xticks(support_)
    ax.set_xticklabels(support_)
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='x', colors='white')
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.legend(loc='upper left',prop={'size': 20})
    plt.close(fig)
    width, height = fig.get_size_inches() * fig.get_dpi()
    width,height=int(width),int(height)
    fig.canvas.draw()
    image1 = np.fromstring(fig.canvas.tostring_rgb(), sep='', dtype='uint8').reshape(height,width,3)
    image2 = resize(image1,shape)
    image2 = img_as_ubyte(image2)
    
    state_render = img_as_ubyte(resize(env.render(mode='rgb_array'),shape))
    image3 = np.hstack((state_render,image2))
    return image3