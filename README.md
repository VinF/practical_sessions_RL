# Practical sessions RL

This repository contains several jupyter notebooks for an hands-on introduction to RL.

### Option 1: Setup using virtualenv

To get everything ready for the practicals using virtualenv, follow these commands:

Clone the repository onto your laptop:
```
$ git clone https://github.com/VinF/practical_sessions_RL.git
$ cd practical_sessions_RL
```

Create a virtual environment called 'venv' using the package virtualenv and activate it:
```
$ virtualenv venv
$ source venv/bin/activate
```

Install all packages in your virtual environment 'venv' using pip (install pip if needed), then install jupyter notebook, then add your virtual environment to the jupyter kernel using ipython (install ipython on your computer if needed).
```
$ pip install -r requirements.txt
$ pip install jupyter
$ ipython kernel install --user --name=venv
```

Now you can start the tutorials!
```
$ jupyter notebook
```

### Option 2: Setup using Anaconda

To  setup using Anaconda, you can use this tutorial:

https://towardsdatascience.com/how-to-set-up-anaconda-and-jupyter-notebook-the-right-way-de3b7623ea4a

Once you have created and activated a conda environment, you can go to the folder that contains this repository that you have cloned/downloaded and you can use the following command to install the required packages for the practical sessions:
```
$ pip install -r requirements.txt
```

You can then launch the jupyter notebook
```
$ jupyter notebook
```
and navigate to the folder of interest for each of the practical sessions.


## Colab

Alternatively, you can upload the notebooks on colab (https://colab.research.google.com) and run them online (you will need a Google account)
You can also just copy the following colab notebooks to your Google Drive:
* Policy gradient: https://colab.research.google.com/drive/1b6o6aLCIt2hdCmxehawmvCxkE0rXy5Ot?usp=sharing
