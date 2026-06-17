## Imported Claude Cowork project instructions

This is the generalised infrastructure for the project of predicting output for one learned case of the model, including checkpoints, classes defined for training, model, analysis and checkpoints, including notebooks where we will mainly be running experiments from. 

We will be implementing mainly notebooks to test and experiment hypothesis, occasionally implementing new algorithms or training models and storing checkpoints.

The key point about the project being in a folder is recycling and saving space/time. When we load/train experiments on notebooks, we need to first check through checkpoints to see if we previously ran the experiment partially and then instead of retraining those checkpoints, just load them in directly
