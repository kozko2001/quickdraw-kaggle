
## Quick Draw

My efforts to try to get a bronzwe medal in QuickDraw Kaggle Competition

Also this is kind of a project to get the structure of a good pytorch project.

It has some tools inside `src/tools` to create the images and the validation set

To execute the code:

```
python src/main.py config/some_config.json
```

You can also create a dry run test, so it's kind of faster... and also uses only 1% of the data


### Installation

1. Install conda
2. Create a new conda environment

```
conda create --name quickdraw --file requirements.conda
```

3. activate the environment

```
source activate quickdraw
```

4. Install requirments using pip

```
pip install -r requirements.txt
```

### Enviromnent variables


* DRY_RUN:

Will not write anything into disk (only logs), and will set the % of the data to 1%, the idea is to be able
to test that config works properly doing small experiments

* USE_VALID

Will use valid dataset as train, this is just a hack, because it takes like 3 min to list all the training
files in quickdraw dataset, and for testing purposes it's useful to make this time go down

### Tensor board

It's nice to have the some metrics... so you can use the tensorboard

```
tensorboard --logdir=./experiments/ &
```

### Remote development with emacs

1. Use tramp C-x C-f /ssh:kozko@....
2. use anaconda-mode
3. M-x pythonic-activate and select the $HOME/anaconda3/env/ environment you want

with that emacs should work perfectly :)


### Things to improve

1. Multiple valid during epoch:

Since the dataset is SOOO HUGE, if training with the full dataset takes a lot of time to get a valid and metric, like 12 hours per epoch

One thing could be do one valid each 10% or each 25% of epoch


2. Test script

A script that uses a model and a checkpoint and writes a csv for kaggle submission

3. half precission

maybe this could make the algorithm go fast

4. Loss functions

Test other loos functions like:

  - Center Loss
  - AM Softmax
  - SoftMarginLoss
  - l-softmax


5. Stochastic Weight average

Try it? maybe... I don't know how much gain could come from here

6. Other schedulers

I'm using LRonPlateau but the problem is that the epoch is so inmense it takes forever to detect the change...
