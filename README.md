`cle-mnist`
=====
_A clean MNIST tutorial built with Pytorch_

Created with newcomers from Cleveland, Ohio in mind.

##### Installation instructions for OS X or Linux
1. Set up your package manager.  I suggest using conda or virtualenv.  There are many tutorials you can find online for doing so.  I'll assume you're using conda, but the installation process won't be very different using virtualenv.

2. Open up a Terminal, and note your current working directory by executing `echo $PWD`.  Let's assume this printed out a filepath `/<root>/<path>`.
Run `git clone https://github.com/jvmancuso/ClevelandMNIST.git`.
Run `cd ClevelandMNIST`.  Your current working directory is now `/<root>/<path>/ClevelandMNIST`.

3. Create your conda environment.  I'll call mine "cle-mnist", so I'll execute the command `conda create -y -n cle-mnist python=3.6`.

4. Next, I'll need to activate the environment with `conda activate cle-mnist`.  If you're using an older version of conda, you may need to run `source activate cle-mnist` instead.

5. There is no one-size-fits-all method for installing PyTorch, but luckily the dev team has streamlined the process considerably:
- Navigate to pytorch.org [(https)](https://pytorch.org/).
- Select your OS, then select your package manager.  Again, I'm using conda.
- Select your Python version.  I suggest using 3.5+, but it depends on which version of Python is installed on your system.  If you've followed the instructions above, you'll have Python 3.6.
- If you have an NVIDIA graphics card with CUDA installed, select your CUDA version.  Otherwise, choose "None".
- Copy the command(s) that show up below the interface, then execute in the same terminal process that you began above.  In particular, it's important that you've activated the `cle-mnist` environment already.

6. Next, install the rest of the dependencies with the command `pip install -r requirements.txt`.

You're installed!

##### Training/evaluating the model
TODO

##### Generating predictions and Kaggle submission
TODO
