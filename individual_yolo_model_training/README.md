**MODEL TRAINING**

This folder contains the bash and python scripts necessary for individually training and evaluating a model. In previous semesters, it was used to take one of the pareto optimal individuals when the evolutionary loop finished and then take it to a full 50 to 100 epochs. In addition, the base model from which the LLM evolved from should also be ran as a comparison. Honestly you only need to do it once and then use it to compare from there.

It's highly likely that the training will not finish within the allotted time as scheduled on your compute cluster (especially if you're using PACE-ICE, our university's HPC), even if you max out the time on your slurm job submission.

Note: There hasn't been any edits made yet to the paths, so before anything is ran or submitted, please edit the paths for your purposes.