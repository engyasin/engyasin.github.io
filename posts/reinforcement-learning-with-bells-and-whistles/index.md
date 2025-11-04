<!--
.. title: Software Choices in Reinforcement Learning: A Comparative Evaluation for a Fast and Accurate Learning
.. slug: reinforcement-learning-with-bells-and-whistles
.. date: 2025-08-17 19:34:35 UTC+02:00
.. tags: reinforcement learning, hyperparameter optimization, jax and flax, logging, gym
.. category: 
.. link: 
.. description: Robust training of RL agents with Gymnasium, Optuna, MLflow, Jax and Flax  
.. type: text
.. status: draft
.. has_math: true
-->

<!--In this hands-on tutorial we will go over some of the very useful practice when training RL agents. While not always necessary for small projects, these techniques are very handy and crucial if the project is to be maintained long-term and to be used in production (possibly with continuous learning over lifetime). The example showcases here specific popular python libraries; like Gymnasium, MLflow, Optuna, and Flax; not separately but rather in one integrated demonstration. : A Comparative Evaluation for Streamlined Development

*If you have an interactive problem at hand, that requires sequence of meaningful actions to optimize for a given objective, like controlling a robotic task or suggesting the best price that can maximize profit, Reinforcement Learning (RL) seems like very practical candidate framework to address your task. however, as soon as you start choosing your tech-stack and the exact algorithmic implementation to use, you will be faced with variety of choices, including popular libraries like Tensorflow or Pytorch, and whether to use Gymnasium for your environment or not; but then there's faster and more optimized packages like Jax and Flax.*
*Finally, if you have found a suitable frameworks and methodologies, you start facing a well-known problem for any machine learning tasks: setting the hyperparameters values, but with many more values to set for RL. What you can do here, is to utilize hyperparameters optimization methods, but again according to what implementation and package, as you don't want to reinvent the wheel each time.*
*In this post, we will go over a grid-world example, illustrating these multiple packages advantages and disadvantages, along with an introduction to each, and finally, how they all fit in one streamlined workflow that can be followed for any RL program need to be written*-->


*Reinforcement Learning (RL) offers a powerful framework for solving sequential decision-making problems in dynamic environments across diverse domains, like control of robots or optimization of profit. However, practical implementation requires navigating a variety of software packages, encompassing deep learning libraries (e.g., TensorFlow, PyTorch, JAX/Flax), environment frameworks (e.g., Gymnasium, Numpy), and hyperparameter optimization techniques and libraries. This post critically evaluates the common PyTorch, Gymnasium, and NumPy RL stack by comparing it to faster alternatives of JAX/Flax for both of model training and simulation of environment. A Gridworld example evaluating both training speed and accuracy is provided to test these packages. Additionally, we complement this example by a comprehensive tracking and monitoring of the training process using MLflow along with a thorough hyperparameters optimization via Optuna. The post concludes with a discussion of the results and final recommendations for optimal use cases of these packages.* 

<!--END_TEASER -->

<!-- Alternative Title: -->


<center>
<br>
<img width="100%"src='/images/rlwithbells/rlwithbells.png'>
<br>
Figure 1: The popular eco-system for modular and scalable training of RL agents. 
</center>

----------

**Table of Content**

----------

[TOC]

----------

<!-- Gif of learning across time for all envs -->

# Introduction and Prerequisites

In the following we will go over all the different libraries, highlighting their practical advantages or disadvantages and their main key difference from their alternatives. Starting with Gymnasium, MLflow, Optuna, and lastly Jax and Flax. We also will show the results of utilizing these libraries on our test program (explained below in Figure 4), focusing on performance (ability to converge optimally) and training time, on our hardware **NVIDIA GeForce RTX 5060 Ti**.

The installation in python (utilizing pip) can be done simply as follows:

```bash
pip install gymnasium
pip install mlflow
pip install optuna

#replace with your cuda version 
pip install "jax[cuda12]"
pip install flax
```

## Standardize your environment with Gymnasium

[Gymnasium](https://gymnasium.farama.org/v0.29.0/) is a new update of the popular Gym package developed originally by OpenAI [@openaigym]. It contains a set of standard simulated environments with unified interfaces, which undergo regular updates. This standardization is helpful for benchmarking different RL algorithms as well as for readability and collaboration. Among the other advantages motivating the usage of Gymnasium are:

- The ability to run **vectorized environments** (`VecEnv`) where multiple instances of the same environment are created and their input and output (states and actions) will then be processed in batches. This vectorization speeds up stepping through these environments and consequently training the RL agent as well. In Gymnasium there's two methods to deploy vectorized environment: either as *Syncretized* or *Asyncretized* environments.  A comparison of the two is displayed in Table 1 below.

<center>
Table 1:  Comparison between Gym vectorization methods `SyncVectorEnv` and `AsyncVectorEnv`
<br>
<br>
<table style="border: 1px solid black" >

<tr >
<th style="border: 1px solid black">
 `gymnasium.Vector.SyncVectorEnv` 
</th>
<th style="border: 1px solid black">
 `gymnasium.Vector.AsyncVectorEnv` 
</th>
</tr>
<tr>
<td style="border: 1px solid black"> create all environments serially and batch the output (state,reward,done flags) </td>
<td style="border: 1px solid black"> each environment is created with its own subprocess </td>
</tr>
<tr>
<td style="border: 1px solid black"> best used when environment is simple, where maintaining independent subprocesses costs more than the environment processes. </td>
<td style="border: 1px solid black"> best used when the environment processes are computationally expensive and there's enough memory for subprocesses. </td>
</tr>
<tr>
<td colspan='2' style="text-align: center;border: 1px solid black">
Input to both functions should be a list of environment creation function (for ex. with `lambda`).
</td>
</tr>
<tr >
<td colspan='2' style="text-align: center;border: 1px solid black">
If you set the optional key input (`shared_memory`) to True, then the output observation data will be referenced directly without copying, which can speed up the stepping if its size if restricting.
</td>
</tr>
</table>

</center>

- **Spaces objects**: used to define the state and action spaces and distributions (imported from `gymnasium.spaces`). Namely, they can represent sets of specific constrains. Example of all the possible sets are shown in Figure 2. 

<center>
<br>
<img width="100%"src='/images/rlwithbells/gymspaces.png'>
<br>
Figure 2: Gymnasium spaces and compound spaces.
</center>

- **Registry**: Custom environments can be registered within the installation so that they can be instanced as a standard Gym package (with `gym.make`). 

- `gymnasium.wrappers` is useful classes to *modify* a specific environment behavior (imported from `gymnasium.wrappers`). Example of these wrappers include: 

    - `ObeservationWrapper`: Modify Observation space
    - `ActionWrapper`: Modify Action space
    - `RewardWrapper`: Modify Reward space
    - `TimeLimit`: Important for truncating of an episode after a specific number of steps.
    - `Automaticreset`: When the environment reach a terminal state or get truncated, this wrapper reset in the next call to `.step()`. With that the last observed state will be directly accessible. 
    - `RecordEpisodeStatistics`: Important to collect episodic_rewards, which indicates the success or failure of a policy during training.

- If your environment is a subclass of `gymnasium.Env`, then you get the advantage of utilizing automatic testing with the function: `gymnasium.utils.env_checker.check_env`, which performs common tests on the gym environment methods and its spaces.


Newly introduced changes in Gymnasium over Gym include the following also:

- Replacing `done` flag when stepping in the environment with `termination` and `truncation`. The difference is simple: *Termination is a natural ending point when the goal of the episode is achieved (for example goal reached). Whilst truncation occurs always after a specific number of steps to avoid running the episode indefinitely.* Figure 3 depict these differences.

<center>
<br>
<img width="80%"src='/images/rlwithbells/termination.png'>
<br>
Figure 3: difference between terminating (goal achieved) and truncating (time limit reached) an episode.
</center>

- Introducing an experimental environment creation method: `gymnasium.experimental.functional.FuncEnv()` of purely functional structure (where the environment class is stateless) to reflect the formulation of POMDP (Partial Observable Markov Decision Process) more closely. Additionally, this structure should enable the integration with JAX.  *Note: we find in this post that utilizing Jax directly is more efficient currently than this new function.*


### Example: Creating custom Gym environment and training it with DQN


We domenstarte the application of Gym (and the rest of the libraries) in this post utilizing a Grid world environment called *Doors* where an agent occupying on cell in that grid is assigned with the task of moving towards a goal cell passing through one of three gaps (doors) in a wall splitting the grid in two, as shown in figure 4, which shows also the state-action configurations.

<center>
<br>
<img width="80%"src='/images/ilpost/Doors_.png'>
<br>
<br>
<img width="30%"src='/images/ilpost/gifs/expert.gif'>
<br>
Figure 4: Doors environment introduced in [previous post](https://www.rlbyexample.net/posts/hands-on-imitation-learning/). The lower image shows animation of optimal policy to solve it.
</center>



> Note: The full repository of the code here is [available here](https://github.com/engyasin/ilsurvey), where the final script with all the libraries [is here](https://github.com/engyasin/ilsurvey/blob/main/dqn_hopt_flax.py). 


We show below parts of the environment creation in Gymnasium:

```python
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import Autoreset, RecordEpisodeStatistics

#creating the environment
class DoorsGym(gym.Env):

    def __init__(self,gridSize=[15,15],nDoors=3,render_frames=True):
        super().__init__()

        self.gridSize = gridSize
        self.nDoors = nDoors
        self.render_frames = render_frames

        self.action_space = gym.spaces.Discrete(5)

        # representing the four states of cells for the entire size of the grid (flattened)
        self.observation_space = gym.spaces.MultiDiscrete([4 for _ in range(np.prod(self.gridSize))])

    def reset(self,seed=None,options=None):

        np.random.seed(seed=seed)
        super().reset(seed=seed)

        pass

    def step(self,action=None):

        pass

```

Then we can use this environment in another script as follows: 

<center>
<br>
<img width="100%"src='/images/rlwithbells/gymcode.png'>
<br>
Figure 5: Environment registering and creation
</center>


As the action space for this environment is discrete, we chose to use Deep Q-Network (DQN) training algorithm [ref] based on CleanRL [ref] implementation to learn the policy. In the following we will show how this task is done while searching for optimal set of hyperparameters utilizing Optuna, namely with Bayesian Optimization and Hyperband algorithm.
<!-- ### Results -->


## Tracking RL experiments with MLflow

- [**MLflow**](https://mlflow.org/) is popular python library for tracking, versioning, collaborating and deployment of machine learning models. Its main functionality is to show the training metrics either in local server (by running `mlflow ui` in new terminal, with the default port as 5000) or in an online cloud server such as *Databricks*.

- The way **MLflow** organizes training is by creating an *experiment* for each machine learning task (for example cat/dogs images classification). Within each experiment we can have many *runs*, which represent training trails for that task (for example different ML approaches for that task). Furthermore, smaller run can be *nested* inside major runs (which we will do for hyperparameter trails below).

- With that structure **MLflow** allows comprehensive saving of all the testing parameters and metrics, and it provides a unified interface to track them. Additionally, **MLflow** is integrated seamlessly with pytorch, tensorflow, and Keras, and has many other functionalities and features that fall out of our scope here, but can be viewed at [their website](https://mlflow.org/).

- We can start a new experiment in MLflow by running `mlflow.create_experiment('experiment_name')` representing a new task for training ML model, or continuing working on an old experiment, that is already created, with the code:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"runs/{experiment_name}")

```
- Note that you need to track the uri where the server will publish the results (in this case `http://localhost:5000`), while running the local server in another terminal with the command `mlflow ui`.

- After that you can start a specific `run` within the experiment, or *multiple children runs* nested within a parent run (utilizing the `nested` keyword argument), which is suitable, for example, if you are doing hyperparameters optimization where each trail can be tracked independently. The following code shows that.

```python

with mlflow.start_run(run_name='main_run',nested=False) as run:

    # log main parameters here
    mlflow.log_params(MainConfigs)

    with mlflow.start_run(nested=True):
        # train here

        mlflow.log_params(argsDict)
        mlflow.log_metric('metric_name',metric_value,step=global_step)
        mlflow.set_tag('label')

        mlflow.log_figure() # matplotlib figure object
        mlflow.log_image() # numpy array and PIL image

        mlflow.pytorch.save_model(model) # saving pytorch model on the server

        mlflow.log_artifacts() # saving other data types

    # save final model
    model_uri = 'copy from dashboard usually starting with models:/'
    model_info = mlflow.pytorch.log_model(pytroch_model,model_uri)

    # load the model
    model = mlflow.pytorch.load_model(model_uri)

```


Then in a new browser tab, you can go to the url: `http://localhost:5000` and view all the experiments. If you chose the active experiment, you can track the different runs with it, either as list showing all the runs, or in chart-view showing all tracked metrics as shown in the figure below:

<center>
<br>
<img width="100%"src='/images/rlwithbells/mlflowcharts.png'>
<br>
Figure 6: The MLflow interface (chart-view) of all the tracked parameters for the active run.
</center>


## RL hyperparameters optimization with Optuna

Training Reinforcement Learning is known to require plenty of hyperparameters to tune, more than its supervised learning counter-parts. This makes it very beneficial to apply efficient hyperparameters optimization methods like Bayesian optimization or Hyperband. In the following sections, we start by reviewing the most prominent methods of hyperparameters optimization, with focus on their implementation utilizing the `Optuna` package.


These hyperparameters in the case of RL, include parameters like: learning rate, episode length, gamma (in Bellman equation), as well as the agent model depth and structure.

## Types of Hyperparameters Optimization Methods:

Generally speaking, there's four main branches of hyperparameters optimization methodologies, varying in their complexity and approach, as the following figure for an example.


<center>
<br>
<img width="100%"src='/images/rlwithbells/hyperopt.png'>
<br>
Figure 7: The main search methodologies for machine learning models hyperparameters.
</center>


### Uninformed Methods

These methods are the simplest, as they test directly different samples from the search space. Depending on their sampling strategy, they can be:

- Manual: Samples are chosen manually.
- Uniform: Samples are chosen uniformly.
- Random: Samples are chosen randomly.

### Bayesian Optimization methods

This category of methods utilize a surrogate model as an approximation of the **objective function** (*a function estimating the learning objective like accuracy or negative loss given the training hyperparameters*). The training data for that model are the values from the past training attempts. While updating the objective approximation model continuously after each training trail, the new set of hyperparameters to be tested will be proposed by another model called: **acquisition function**.

Based on the nature of that surrogate model, Bayesian Optimization (BO) methods can be categorized into:

- Sequential Model-based Algorithmic Configuration (SMAC): utilizing random forest to approximate the objective function, which makes it suitable for categorical and discrete parameters search.

- Sequential Model-based Bayesian Optimization (SMBO): utilizing Gaussian Process model, suitable for continuous hyperparameters

- Tree-structured Parzen Estimators (TPE): utilizing random forest, suitable for large search space for both continuous and discrete search, with fast run-time. In `Optuna`, its implementation allow learning interactive relations between different parameters. Its Optuna function  is: `optuna.samplers.TPESampler`.

- Matis: Gaussian Process-based, utilizing also a Gaussian Mixture Model as its acquisition function.


### Heuristic Search


This branch of methods samples the hyperparameters of its next training iteration in the neighborhood of the best set of hyperparameters found so far. Clearly the definition of this neighborhood has big impact on the search performance, where we have multiple variants:

- Simulated Annealing (SA): it searches for its next sample around the best or the next-to-best set of values so far, to avoid local minima.

- Genetic Algorithm: It applies evaluations-inspired methods to select its next set of values. Namely, it is based on pairing the best samples found so far of different parameters, or mutating them.

- Particle Swarm Optimization: This method focuses especially on the case of continuous hyperparameters.

- Population-based Training: This method specializes in neural networks optimization, as it searches for both hyperparameters and normal training parameters as well. For example, it adds gradually new layers to the model under training after each training iteration, where the old trained layers are kept. However, it cannot recover the exact best hyperparameters for the best model, as it only finds the final trained model parameters.


### Multi-Fidelity Optimization (MFO)

This branch of methods adds another dimension to the problem of hyperparameters optimization, as it allows faster training by early stopping of not-so-promising samples, either by training on parts of the data, or for lower number of epochs (as the case in Optuna). This makes more sense than full training for all samples, as we don't need to invest computational resources in testing many samples, where they may have low probability of being optimal, while focusing more on areas where the performance seems more promising. The methods here try to shape this idea as recourses management algorithm.  It is also worth noting, that MFO methods can be combined directly with the previous sampling methods, as they work on different aspect of the problem. In Optuna, MFO methods are called **Pruners**,  and sampling methods are called **Samplers**.

The most popular MFO methods include:

- Coarse to Fine Pruner: as the name suggest, this method starts by light training of many samples candidates, focusing increasingly on more promising subset of samples.

- Successive Halving (SH): This method distributes the computational resources wisely on the different training trails.

- Hyper Band (HB): This method defines pairs of candidates numbers with their allocated resources, called *brackets* and starts full training of some of these brackets to avoid dropping promising candidates mistakenly, as it can happen in SH due to shallow training. Its Optuna function is: `optuna.pruners.HyperbandPruner`.

- Bayesian Optimization Hyper Band (BOHB): It was noted that better results are obtained when using BO sampler with Hyperband pruner, as the work in [x] details.  In Optuna this can be done by utilizing TPE sampler with HB pruner.


## Steps for doing Hyperparameters Optimization in Optuna:

It can seem difficult to perform hyperparameters search while training the RL agent, and possibly track all experiments and compare all trail. However, Optuna has simple implementation order, with a built-in support for most of the methods mentioned here and it can be directly integrated with libraries like MLflow, Pytorch and Jax. Specifically, we start by:

1. Defining the objective function, which returns (in the case of RL) the average episodic return for all training episodes.
2. Inside that objective function, we define the hyperparameters ranges and types to be optimized using `optuna.trail.suggest_` group of functions.
3. Initializing the optimization object (called the *study*) with `create_study()` and within it, we define our `sampler` and `pruner` methods, in addition to the *direction* (defaults to minimizing).
4. Optionally, saving the current training session by passing the `storage` argument (to `create_study()`) representing a database url to save the study object in. Additionally to resume training from saved session of trails, you can pass `load_if_exists=True` to the same function. 
4. Starting the training with `.optimize()` method of the previous study object, passing the objective function (as callable) and the number of trails.
5. When the optimization ends, show the best set of parameters (in `study.best_params`) and save the model.


It is also worth noting that `Optuna` also has a visualization module `optuna.visualization` whose functions take the optimized study object as input and present many useful plots, like plotting the most influential hyperparameters on the results. This last module requires the installation of `plotly` package.


In the following we show some illustrative code snippet to implement the above steps.

### Sample Training Code with Optuna


```python

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from functional import partial


def objective(trail,argsParams={}):

    argsParams.update({"num_steps":trial.suggest_int("num_steps", 10, 17, step=1)})
    argsParams.update({"learning_rate":trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)}) 
    # log argument makes it more probable to sample lower values.
    argsParams.update({"buffer_size":trial.suggest_int("buffer_size",16 , 48, step=1, log=True)})
    argsParams.update({"batch_size":trial.suggest_int("batch_size", 16, 128, step=16)})
    argsParams.update({"train_frequency":trial.suggest_int("train_frequency", 2, 24, step=1, log=True)})
    argsParams.update({"optimizer_name": trial.suggest_categorical("optimizer_name", ["Adam", "SGD"])})

    # define network

    # define optimizers

    # training loop

    with mlflow.start_run(nested=True) as run:

        mlflow.log_params(argsParams)

        # training logic
        for epoch in range(NumberOfEpochs):
            # training logic


            # send the final metrics
            mlflow.log_metrics({"charts/episodic_return": infos["episode"]["r"][finished].mean(),
                                "charts/episodic_length": infos["episode"]["l"][finished].mean(),
                                "charts/epsilon": f"{epsilon:2f}"},
                                step=global_step)

            # break training whenever sample seems not optimal (early stopping)
            if trial.should_prune():
                raise optuna.TrialPruned()


    # return average episodic reward (objective)

with mlflow.start_run(run_name=run_name) as run:

    study = optuna.create_study(sampler=TPESampler(seed=seed, multivariate=False), 
                                # if multivariate is true the sampler can learn the mutual interactions of variables
                                pruner=HyperbandPruner(min_resource=240, max_resource=max_epochs, reduction_factor=3), #resource represents epochs
                                direction="maximize")

    # objective function should be passed as callable without arguments to optimize method
    objective_func = partial(
        objective, argsParams=vars(argsParams).copy(), device=device
    )

    study.optimize(objective_func, n_trials=12) # hom many trails to test

    print(study.best_parameters) # the results

    mlflow.log_params(study.best_params) # log it with mlflow

    # visualizations require plotly installed

    plotly_fig = optuna.visualization.plot_param_importances(study,evaluator=None) 
    plotly_fig.show()
    # evaluator is optuna.importance.FanovaImportanceEvaluator by default or optuna.importance.MeanDecreaseImpurityImportanceEvaluator

    plotly_fig = optuna.visualization.plot_contour(study)
    plotly_fig.show()


    plotly_fig = optuna.visualization.plot_optimization_history(study)
    plotly_fig.show()
    # these images can be viewed in new widows or sent to MLflow server to view them alongside other parameters

    mlflow.log_figure(plotly_fig,artifact_file=f"opt_history.html") 

```

In our accompanying [code repository](https://github.com/engyasin/ilsurvey), we set 40 trails searching for optimal hyperparameters, and visualized the results at the end, regarding:

- Parameters importance with `optuna.visualization.plot_param_importances`

<center>
<br>
<img width="100%"src='/images/rlwithbells/params_importance.png'>
<br>
Figure 7: Hyperparameters estimated relative importance on the model training performance. The largest two important parameters are the episode length and the learning rate.
</center>

- 2D heatmaps of interactive hyperparameters importance with `optuna.visualization.plot_contour`

<center>
<br>
<img width="100%"src='/images/rlwithbells/contours.png'>
<br>
Figure 7: 2D heatmaps of interactive pair-wise importance on the performance. We see here clearly that the darker regions are the best performing are regions for that parameter.
</center>

- Performance of trails over time with `optuna.visualization.plot_optimization_history`

<center>
<br>
<img width="100%"src='/images/rlwithbells/optimization_history.png'>
<br>
Figure 7: Improvement of trails performance over order of training. We see clearly here that over time, the hyperparameter optimization was beneficial in learning better set of values to results in better performance. With further search we can expect that curve to continue his ascent.
</center>


> Lastly, we note that looking at these figures can help us estimate and understand the effective ranges or combination of ranges that result in the best performance. Possibly leading to more manual enhancement of the other program parts which is not under optimization.

<!-- ### Optimizing the hyper parameters of the trained agent -->



<!-- ### Result -->

## Speed up environment rollout and model training with JAX & Flax

The common option when training RL model utilizing simulated environment is to use Pytorch or Tensorflow for training the RL agent and Numpy for simulating your environment. However, an increasingly popular alternative to consider, replacing these libraries is a package developed by Google called  **JAX** [*(Just After Execution)*](https://docs.jax.dev/en/latest/index.html). JAX is faster way to run matrix computation efficiently (instead of `numpy`) and to train neural network models (instead of `pytorch`), due to its targeted exploitation of the hardware computational devices like GPUs and TPUs. While JAX can be utilized directly to update the neural network parameters; another JAX-based targeted package, like FLAX, can make life easier, when structuring your model and training algorithm. 

In the following, we will mention some of the key features of JAX, focusing on its Numpy-alternative functionalities, which we will demonstrate later by rewriting the same Doors Gym environment in Jax and comparing its run-time with the original. 

- JAX works by compiling the code with **XLA** *(Accelerated Linear Algebra)* compiler to statically typed expression language called **Jaxpr**. This compiled code run faster on CPUs, GPUs, and TPUs. Practically, after writing your JAX functions, you can compile them by passing them to `jax.jit()` function or by placing the decorator `@jax.jit` right above their definitions.

- JAX replaces most of Numpy functions utilizing similar names so that modifying your numpy code is minimized. Mostly you should only replace `import jax.numpy as np` with `import numpy as np`. However, some other considerations should be noted as we will:


> *Note*: JAX arrays, unlike numpy, are immutable. So they cannot by changed inplace. Instead we have to change them with the following code:

```python
import jax
arr = jax.numpy.arange(10)
arr = arr.at[1].add(2) # equivalent to arr[1] += 2 in numpy
```

> *Note*: JAX arrays don't throw an error (`OutofIndex`) if the index is out of its range, but default to giving the last item in the array.

> *Note*: JAX default precision is `float32` unlike Numpy's `float64`

> *Note*: JAX offers alternative functions of Scipy functions with `jax.scipy` 

> The following code shows an example of JAX compatible function compiled with jit, measuring its runtime


```python
import jax
import time

arr = jax.numpy.arange(35).reshape(7,5) # 7x5 array

print(f'JAX running on : {arr.device}')

@jax.jit
def ATA(x):
    
    return x.dot(x.T)

# run in IPython :
%timeit -n 100 ATA(arr).block_until_ready()


```

- JAX can autovectorise any function with its `jax.vmap()` function (alteratively with `@jax.vmap` decorator). This is needed if you want to run a function or sequence of inputs: instead of looping through each input alone, you can pass these inputs as *batch* and get huge speed up over python. Practically the input and output will be stacked and concatenated adding another dimension to their matrices (you can chose its place). We show below that this is also faster than Gymnasium way of environment vectorization.


- In Jax we can also vectorize functions across computational recourse, which allow parallel processing. This has the same implementation as `vmap` but by wrapping any function with `jax.pmap()` or the decorator `@jax.pmap`.


> *Note*: JAX execution is asyncronized by default, this means that the code return directly before calculating the output of a function. To force it to wait, we should append any function call with `.block_until_ready()`.


- In addition to compiling with XLA, JAX can calculate gradients effectively by doing **automatic differentiation** *autodiff* of the calculations of all variables. This is very useful in speeding up training of neural networks.


- Control statements (*for, while, if, switch*) are known as performance bottleneck in Python. In JAX, they can be replaced with the following: 

```python
from jax import lax

lax.cond # if
lax.switch # switch, case
lax.while_loop # while
lax.fori_loop # for

# example for fori_loop
@jax.jit
def main():

    def for_loop_body(i,accumulator):

        accumulator += accumulator

        return accumulator

    accumulator = 10
    init_val = accumulator
    start_i = 0
    end_i = 100

    final_value = lax.fori_loop(start_i, end_i, for_loop_body, init_val)

```

> *Note*: For the code to be correctly compiled or vectorized in JAX, it should be *functional* only. Object oriented code (like stateful classes) cannot be compiled in JAX . However; stateless classes objects can be used, where they don't save any internal variables (or use them as static variables only). If these variables need to be changed, then they are, by definition, part of the state.

> *Note*: This last restriction of functional code shouldn't be seen as a drawback. In fact, functional code is commonly considered more readable and a better structure of the code. 

- The following code snippet shows an example of our Doors environment converted to *stateless* class, while still compilable with gymnasium. Specific new functions are explained in comments.

```python

import gymnasium as gym
import cv2

from functools import partial

import jax
from jax import jit,random
import jax.numpy as np
from jax import lax,vmap, pmap


class DoorsEnvJax(gym.Env):

    def __init__(self,gridSize=[15,15],nDoors=3):
        super().__init__()

        # Static variables - not to be changed: otherwise an error is thrown.
        EnvConfig = {}
        self.gridSize = gridSize
        self.nDoors = nDoors

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.MultiDiscrete([4 for _ in range(self.gridSize[0]*self.gridSize[1])])

        self.actions_vocal = np.array([[0,0],[0,1],[1,0],[0,-1],[-1,0]]).astype(int)


    @partial(jit,static_argnums=(0,)) # ignore the first (self) input
    @partial(vmap,in_axes=(None,0,0,0)) # vectorize along the first dimension (order 0) of all inputs except the first (None)
    def step(self, action, env_state, info):

        key = env_state[1]
        state = env_state[0]
        agent_location = info['agent_location']
        goal_location = info['goal_location']
        episodic_reward = info['episode']['r']
        timestep = info['episode']['l']
        max_steps = info["num_steps"]


        movement = self.actions_vocal[action]
        new_location = np.clip(agent_location+movement,0,np.array(self.gridSize)-1)

        terminated = False
        truncated = np.array(max_steps<=timestep,dtype=np.bool_) 
        past_position = agent_location.copy()

        # check if wall (2)

        cell_state = state.at[*tuple(new_location)].get() # array elements are returned by .get()

        possible_moves = np.logical_or(cell_state == 0, cell_state == 3) # conditions should be performed by jax functions

        # boolean indexing can be done utilizing jax.np.where
        state = np.where(possible_moves, # boolean mask array
                state.at[tuple(agent_location)].set(0).at[tuple(new_location)].set(1), # value if True
                state # value if False
                 )

        agent_location = new_location.copy()

        terminated = (cell_state == 3) 

        reward = self._get_reward(past_position,agent_location,goal_location)
        info.update(self._get_info(agent_location,goal_location))

        # automatic reset
        new_state = np.where(np.logical_or(terminated,truncated),
                 self.reset(key[None,:])[0][0][0,...], # to remove vector dimension
                (state).copy())

        info.update({"new_state":new_state,
                     "episode":{'r':episodic_reward+reward,'l':timestep+1},
                     "agent_location":np.hstack(np.where(new_state==1,size=1)),
                     "goal_location":np.hstack(np.where(new_state==3,size=1))})

        # Random keys should be used only once. Therefore we generate a new one each step.
        new_key = random.split(key)[0,:]

        return (new_state,new_key), reward, terminated, truncated, info

```

As you can see from the example above, the environment class is vectorized by definition, where we can pass the matrices of all actions stacked along the first dimension to step through multiple environment simultaneously. Namely, this starts from the `.reset()` function, by passing a corresponding number of random keys:

```python

    key = random.PRNGKey(0)
    NUM_ENVS = 24 # vmap
    keys = random.split(key,NUM_ENVS) # generate new keys from existing ones.

```
This vectorization has shown to be extremely advantageous in our tests. To emphasis that we tested the runtime for a range of DOORS environment numbers doing the same operations, in JAX, Gym Syncretized, Gym Asyncretized, and JAX accelerated looping between steps (which is usually slow in Python) vectorized format. The following plot draws the relationship of runtime as a function to  number of environments for these three methods.

<center>
<br>
<img width="100%"src='/images/rlwithbells/runtimeEnvs.png'>
<br>
Figure 7: Comparing runtime of different vectorization methods. JAX seems insensitive to number of environments running up to 500. Speeding up the for loop led to super fast performance of 0.07s.
</center>

**JAX-based environments don't seem to slow down with big environments numbers.** This is very interesting note, because we can increase our environment counts and speed up the rollout phase in a lot of RL training methods. The test code is available in the display.py script in the repo and anyone can test it. 
Additionally, we note that Syncretized was faster than the Asyncretized version, as the DOORS environment is somewhat simple compared to the overhead of spawning many subprocess.


<!-- compare run time with Gym: Plot (Number of envs, Time): Gym (list), Asyn, Sync, JAX  and how that impact the training -->

### FLAX

FLAX [ref] is a Jax-based specialized library for building and training neural networks, which is regarded as faster and more readable library than Pytorch or Tensorflow, due to its dependence on JAX.

Additionally, for creating  composable gradient transformation in JAX, we can utilize another Jax-based library called (`optax`) [ref], beside FLAX. 

The definition of neural networks classes in FLAX is inherited from `flax.linen.Module`, where the forward inference of that network is expressed in its `__call__()` function with the annotation `@flax.linen.compact`. This means that the network creation interface in FLAX is object-oriented (unlike JAX which allows only functional programming), while still interpretable with JIT.

The following code is an example of defining a neural network in Flax and passing a random input to it, which is necessary step to initialize its parameters. Note also that these parameters are required input for the model inference (with the `.apply()`) as it is stateless class.


```python
from jax import random
from flax import linen as nn


class MLP(nn.Module):
    @nn.compact
    def __call__(self,x):

        x = nn.Dense(features=512)(x)
        x = nn.activation.swich(x)
        x = nn.Dense(features=10)(x)
        return x

    
model = MLP()
main_key = random.PRNGKey(0)
key1, key2 = random.split(main_key)

random_data = random.normal(key1,(28,28,1))
params = model.init(key2, random_data)

out = model.apply(params, random_data)
print(model.tabulate(key2,random_data))

```

Another benefit in FLAX here is the automatic vectorization of the network functions, without the need to wrap it with `jax.vmap`, where the first dimension is always the batch dimension. 

After defining the network above, we can define the optimizer utilizing `optax` and the training code will be as follows:

```python

from flax import train_state
import optax

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.sgd(learning_rate=1.0,momentum=0.9)
)

@jax.jit
def update(train_state,x,y):

    def loss(params, inputData, target):

        logits = train_state.apply_fn(params, inputData)
        log_preds = logits - jax.np.logsumexp(logits)

        return -jnp.mean(target*log_preds)

    loss, grads = jax.value_and_grad(loss)(train_state.params,x,y)

    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss_value

```

With the previous code we can update the parameters of the model, based on the loss. To save the final trained model with Flax we write:


<!-- TODO: check if we can log flax model to mlflow and load them? -->

```python
with open(model_path, "wb") as f:
    f.write(flax.serialization.to_bytes(model.params))

# This code save the model parameters in a data object, to load the parameters again use:

with open(model_path, "r") as f:
    q_state.params = flax.serialization.from_bytes(q_state.params, f.read())

```

> *Note*: `orbax` library is another higher level Jax-based package to save flax model automatically and efficiently. 


<!--### Making the environment functional for JAX vectorization -->


<!-- ### Training DQN with Flax -->


<!-- ### Result : Speed comparison with Pytorch-->


## Final Take-away

<!-- comparing run time and performance -->

The following table shows the performance (measured by the final mean of rolling rewards in the training curve for the last 2000 episode (out of 5e5 episode in total)) and the run-time of the training phase for three variants of scripts:

- Pytorch with Gym (Synchronization environments) [available here](https://github.com/engyasin/ilsurvey/blob/main/dqn_hopt_flax.py)
- FLax with Gym (Synchronization environments) [available here](https://github.com/engyasin/ilsurvey/blob/main/dqn_hopt_flax.py)
- FLax with Gym environment and JAX automatic vectorization on GPU [available here](https://github.com/engyasin/ilsurvey/blob/main/dqn_hopt_flax.py)

Note that these estimation is on tested **NVIDIA GeForce RTX 5060 Ti** as GPU and **AMD Ryzen 5 7600X 6-Core Processor** as CPU.

<center>
<br>
Table 2: Performance and Runtime of training DQN agent to solve DOORS environment utilizing three different combinations of packages (JAX, FLAX, and Pytorch)

<br>
<table style="border: 1px solid black" >

<tr >
<th style="border: 1px solid black">
</th>
<th style="border: 1px solid black">
 Pytorch for DQN
</th>
<th style="border: 1px solid black">
 FLAX for DQN
</th>
<th style="border: 1px solid black">
 FLAX-DQN and JAX for Env
</th>
</tr>
<tr>
<td style="border: 1px solid black"> Rolling Reward</td>
<td style="border: 1px solid black"> <strong>0.73 </strong></td>
<td style="border: 1px solid black"> 0.64</td>
<td style="border: 1px solid black"> 0.58 </td>
</tr>
<tr>
<td style="border: 1px solid black"> Training Time</td>
<td style="border: 1px solid black"> 22.5 min </td>
<td style="border: 1px solid black">  22.8 min </td>
<td style="border: 1px solid black"> <strong>2.5 min</strong> </td>
</tr>
<tr>
<td style="border: 1px solid black"> Training Cruves </td>
<td style="border: 1px solid black"> <img width="100%"src='/images/rlwithbells/PytorchNumpy.png'> </td>
<td style="border: 1px solid black">   <img width="100%"src='/images/rlwithbells/FLAXNumpy.png'> </td>
<td style="border: 1px solid black"> <strong><img width="100%"src='/images/rlwithbells/jax_flax.png'></strong> </td>
</tr>

</table>
<br>
</center>



We note from the results in Table 2, that hyperparameter optimization was helpful in finding a good model that reached good performance (0.73 after 40 trails) with pytorch, the other programs with JAX and FLAX were close but a bit worse which can be attributed to random initialization or potential for deeper search for hyperparameters.

The major improvement was in the training time when we replace normal Numpy operations in the DOORS environment with JAX accelerated and functional code. Beside that, as we noted previously that increasing the number of environments will not affect the speed of JAX functional stateless classes, **we took advantage of that and increased the number of number of environments 16 times, which contributed to this huge speed-up of around 10 times.** We expect the possibility of larger speed-up if this number is increased further. The rest of settings and hyperparameters ranges was the same for all three programs.


With that we form our final recommendation here of when to use each of these packages:

1. Gymnasium: If you want to create new environment, and you care about sharing it and collaborating it with others. Then making it compatible with Gymnasium is a good decision toward that goal. 
2. MLflow: If you want comprehensive tracking of all of your training metrics and parameters, full display of the hyperparameters in your programs, and deployment over life-time, then utilizing MLflow is a a great and easy way to achieve that.
3. Optuna: If your model is complex and contain a lot of hyperparameters making it hard to tune manually (as usually the case for Reinforcement Learning programs), Optuna can provide implementations of advanced algorithms to search for optimal values with direct compatibility with MLflow.

4. JAX: If your environment simulating is not fast enough and require lengthy operations (forming the bottleneck of your runtime), then vectorizing the environment with JAX eps. on GPU or TPU devices can give a great boost for the training as bigger batches can be sampled faster.

4. FLAX: If you model training is not fast enough and you have access to TPUs or GPUs, then vectorizing the environment with FLAX can speed up the training, although not confirmed in our post, but that depends heavily on your training algorithm, data size and hardware. 


## More Libraries in JAX 

### Brax 

[Brax](https://github.com/google/brax) is the JAX-based version of MujoCo, developed by Google, [(check our post here for an introduction of MujoCo)](https://www.rlbyexample.net/posts/immerse-yourself-in-reinforcement-learning-and-robotics-with-mujoco/). It shows great speed-up over standard MujoCo, and comes with implementations of SAC and PPO RL algorithms

### Dopamine

[Dopamine](https://github.com/google/dopamine) is also a package developed by Google, that provide a JAX implementation of different RL algorithms, and allow fast training and testing on different environments. 




## About the author

I'm on the lookout for RL or Robotics roles in the EU or remote! If you happen to know of any companies that are hiring for these areas – a referral or even a quick shout-out would be very helpful.  My resume is [available here](https://drive.google.com/file/d/1l5tKlT3XOJMuk5GRvXAnAl5BDLvgkT5d/view?usp=sharing) 


Lastly, I'd also welcome any contributions to the code, discussions about the topics, or questions you'd like to ask.  

## References


<!-- TODO: Turn Code into images -->
<!-- TODO: Make other images -->
<!-- TODO: Add experiments results -->
<!-- TODO: Mention the hardware -->



<!-- TODO: Add references -->
<!-- TODO: Count Figures -->
<!-- TODO: Edit by yourself then LLM then yourself -->
<!-- TODO: More on Tree-structured Parzen Estimators -->
<!-- TODO: Change or remove the cover image -->
 