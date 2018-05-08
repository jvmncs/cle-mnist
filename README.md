`cle-mnist`
=====
_A clean MNIST tutorial built with Pytorch_

Created with newcomers from Cleveland, Ohio in mind.  This is an extension of the 60 Minute Blitz ([https](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)) from the official PyTorch tutorials including a few PyTorch extras that are important for beginners to know about.

##### Installation instructions for OS X or Linux
1. Set up your package manager.  I suggest using conda or virtualenv.  There are many tutorials you can find online for doing so.  I'll assume you're using conda, but the installation process won't be very different using virtualenv.

2. Open up a Terminal, and take note of your current working directory by executing `echo $PWD`.  Let's assume this printed out a filepath `/<root>/<path>`.
Execute `git clone https://github.com/jvmancuso/cle-mnist.git` in the terminal, then execute `cd cle-mnist`.  Your current working directory is now `/<root>/<path>/cle-mnist`.

3. Create your conda environment.  I'll call mine "cle-mnist", so I'll execute the command `conda create -y -n cle-mnist python=3.6`.

4. Next, I'll need to activate the environment with `conda activate cle-mnist`.  If you're using an older version of conda, you may need to run `source activate cle-mnist` instead.

5. There is no one-size-fits-all method for installing PyTorch, but luckily the dev team has streamlined the process considerably:
- Navigate to pytorch.org ([https](https://pytorch.org/)).
- Select your OS, then select your package manager.  Again, I'm using conda.
- Select your Python version.  I suggest using 3.5+, but it depends on which version of Python is installed on your system.  If you've followed the instructions above, you'll have Python 3.6.
- If your computer has an NVIDIA graphics card with CUDA installed, select your CUDA version.  Otherwise, choose "None".
- Copy the command(s) that show up below the interface, then execute in the same terminal process that you began above.  In particular, it's important that you've activated the `cle-mnist` environment already.

You're installed!

##### Demo commands
- Softmax classifier: `python main.py --model linear`
- Two-layer neural network: `python main.py --model neuralnet`
- Convolutional network: `python main.py --model convnet`

##### Training for performance
To see all customizable arguments, execute `python main.py -h`.<br>
Suggested minimum training epochs:
- linear: 30
- neuralnet: 60
- convnet: 60

Suggested learning rate:
I've gone with .001, but feel free to tune it more!  Just be sure not to overfit.<br>
I've also neglected to tune the convolutional network architecture, mainly because I'm planning on using it to reproduce a recent paper ([https](https://blog.openai.com/debate/)).  If you're interested in tuning it, I suggest doing more to prevent overfitting, perhaps by adding dropout or batch normalization to the convolutional layers.

##### Generating Kaggle submission
Assuming I have a trained model, and it's located at `./path/to/checkpoint/from/root`...
1. Choose a filename for your submission file.  I'll go with `kaggle_submission.csv`.
2. Make sure you're in the root directory of this repo (cle-mnist, located wherever you cloned it).
3. Run `python make_submission.py
