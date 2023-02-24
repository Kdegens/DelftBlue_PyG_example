# Using Delft Blue
Delft High Performance Computing Centre (DHPC), is the supercomputer of the Delft University of Technology. The system is managed by the DHPC team and is available for all researchers at TU Delft. The documentation of the system can be found here: https://www.tudelft.nl/en/dhpc/documentation/

This repository contains an toy example of how to use the DHPC system in combination with Pytorch Geometric.

## Getting started
To get started with DHPC, you need to register for an account. You can do this by following the instructions on the DHPC website: https://www.tudelft.nl/en/dhpc/getting-started/

Or directly go to the registration using this [link](https://www.tudelft.nl/en/dhpc/getting-started/register/) and fill in the form.

## Logging in
To log in to DHPC, you need to use the following command:
```
ssh <netid>@login.delftblue.tudelft.nl
```

An alternative is to use visual studio code, to remotely connect to DHPC. To do so install the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) and [Remote Explorer](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extensions. Then, in the Remote Explorer, click on the + icon and select "Connect to Host". Enter the following command given above and press enter. You will be prompted for your password. After that, you will be connected to DHPC.

## Setting up python on the DHPC
Once you are connected to DHPC, you can use the system to run your code. 
On the DHPC you acces programs using the module system. To see which modules are available, see the [DHPC website](https://doc.dhpc.tudelft.nl/delftblue/DHPC-modules/).

For this example, we will need the `python`, `cuda` and `pip`. To load these modules, use the following commands:
```
module load 2022r2
module load python/3.8.12
module load cuda/11.6
module load py-pip
```

To install the required packages, we will use `pip`. But, fist we need to link the storage to our home directory. To do so, use the following command:
```
mkdir -p /scratch/${USER}/.local
ln -s /scratch/${USER}/.local $HOME/.local
```

Now, we can use git to clone this repository and install the required packages. To do so, use the following command:
```
git clone https://github.com/Kdegens/DelftBlue_PyG_example.git
```
To enter the directory, use the following command:
```
cd DelftBlue_PyG_example
```
To install the required packages, use the following command:
```
pip install -r requirements.txt
```
As you can see, this will installs the modules specified in the `requirements.txt` file. But, it wont install any of the Pytorch related packages. This because, when using pip, it will install the packages for the CPU. To install the Pytorch packages for the GPU, we need to manually install them. To do so, use the following command:
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

And for Pytorch Geometric, use the following command:
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
```

This example uses wandb to log the results. To use wandb, you need to create an account. You can do so by following the instructions on the [wandb website](https://wandb.ai/site). After you have created an account, you need to login to wandb. To do so, use the following command:
```
wandb login
```
in the terminal, you will be prompted to enter your wandb API key. You can find this key on your [wandb profile page](https://wandb.ai/authorize).

## Things to change
Both the `process_data.py` and `main.py` files contain a variable called `net_id`. This variable should be changed to your netid. This is needed to make sure that the data is stored in the correct directory. If you don't change this variable, the data will be stored in the directory of the person who created the repository. This will cause problems when running the code on DHPC. 


## Running code on DHPC
Running code on a supercomputer is a bit different than running code on your local machine. To run code on DHPC, you need to submit a job. To do so, you need to create a job file. This file contains the instructions for the job. For this example, we will use the `job.sh` file.To submit the job, use the following command:
```
sbatch job.sh
```

After you have submitted the job, you can check the queue status using the following command:
```
squeue 
```

Since the queue can be quite long, and DHPC is an system focussed on research, you can use the following command to see the queue status of the `gpu` partition:
```
squeue -p gpu
```

If only interested in the status of your job, you can use the following command (${USER} automatically gets replaced by your username):
```
squeue -u ${USER}
```

To see your history of jobs, you can use the following command:
```
sacct
```



### making a job file
The `job.sh` file contains the following instructions:
```sh
#!/bin/sh
#
#SBATCH --job-name="try"
#SBATCH --partition=gpu
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-as-msc-ce

module load 2022r2
module load python/3.8.12
module load cuda/11.6
module load py-pip

srun python -m main
```

in the `job.sh` file, you can see that we are using the `gpu` partition. This means that the job will be run on a GPU node. If you want to use a CPU node, you can use the `cpu` partition. The `--gpus-per-task` specifies the number of GPUs that you want to use. In this case, we are using 2 GPUs. The `--cpus-per-task` specifies the number of CPUs that you want to use. In this case, we are using 24 CPUs. The `--mem-per-cpu` specifies the amount of memory that you want to use per CPU. In this case, we are using 1GB per CPU. The `--account` specifies the account that you want to use. In this case, we are using the `education-as-msc-ce` account. For more information about the different partitions, see the [DHPC website](https://doc.dhpc.tudelft.nl/delftblue/Slurm-scheduler/).


