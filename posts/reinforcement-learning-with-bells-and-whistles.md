<!--
.. title: Speeding up Training of Model-Free Reinforcement Learning : A Comparative Evaluation for a Fast and Accurate Learning
.. slug: reinforcement-learning-with-bells-and-whistles
.. date: 2025-08-17 19:34:35 UTC+02:00
.. tags: reinforcement learning, hyperparameter optimization, jax and flax, logging, gym
.. category: 
.. link: 
.. description: Robust training of RL agents with Gymnasium, Optuna, MLflow, Jax and Flax  
.. type: text
.. has_math: true
-->

<!--In this hands-on tutorial we will go over some of the very useful practice when training RL agents. While not always necessary for small projects, these techniques are very handy and crucial if the project is to be maintained long-term and to be used in production (possibly with continuous learning over lifetime). The example showcases here specific popular python libraries; like Gymnasium, MLflow, Optuna, and Flax; not separately but rather in one integrated demonstration. : A Comparative Evaluation for Streamlined Development

*If you have an interactive problem at hand, that requires sequence of meaningful actions to optimize for a given objective, like controlling a robotic task or suggesting the best price that can maximize profit, Reinforcement Learning (RL) seems like very practical candidate framework to address your task. however, as soon as you start choosing your tech-stack and the exact algorithmic implementation to use, you will be faced with variety of choices, including popular libraries like Tensorflow or Pytorch, and whether to use Gymnasium for your environment or not; but then there's faster and more optimized packages like Jax and Flax.*
*Finally, if you have found a suitable frameworks and methodologies, you start facing a well-known problem for any machine learning tasks: setting the hyperparameters values, but with many more values to set for RL. What you can do here, is to utilize hyperparameters optimization methods, but again according to what implementation and package, as you don't want to reinvent the wheel each time.*
*In this post, we will go over a grid-world example, illustrating these multiple packages advantages and disadvantages, along with an introduction to each, and finally, how they all fit in one streamlined workflow that can be followed for any RL program need to be written*-->


*Reinforcement Learning (RL) represents a powerful framework for solving sequential decision-making problems in dynamic environments across diverse domains, such as control of robots or optimization of profit. However, its practical implementation requires navigating a variety of software packages, encompassing deep learning libraries (e.g., TensorFlow, PyTorch, JAX/Flax), environment frameworks (e.g., Gymnasium, Numpy), and hyperparameter optimization techniques and libraries. This post critically evaluates the common PyTorch, Gymnasium, and NumPy RL stack by comparing it to a faster alternative: JAX/Flax for both of the model training and simulation of environments. A Gridworld example evaluating both training speed and accuracy is utilized to test these packages. Additionally, we complement this example by a comprehensive tracking and monitoring of the training process using MLflow along with a thorough hyperparameters optimization via Optuna. The post concludes with a discussion of the results and final recommendations for optimal use cases of each of these packages.* 

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

# Introduction & Installation

The common workflow for applying Reinforcement Learning to optimize an objective, is to start by defining Markov Decision Process (MDP) quantities, like the state and action spaces, and reward function. Additionally, we need the environment model as a simulation in order to simulate the forward application of our RL agent in model-free algorithms. The training process will alternate then between collecting rollouts and training the agent on them. This means that the run-time of our program is affected by two components: the neural network parameters updates, and the environment simulation process. Openai-Gym, proposed originally in [1], and its successor (Gymnasium [2]) are well-known popular python libraries that provide a structure to build RL simulated environments. For training the agent model itself, libraries like Tensorflow or Pytorch are common. In this post, we are interested in exploring a promising alternative for both of the simulation and the training phases utilizing JAX package [21], and its neural network extension: FLAX [20]. The motivation here is to speed up the training process as well as achieve better optimization.

Based on our tests on Grid-World environment, we found that utilizing JAX for batching the environments can results in huge speed up on our GPU hardware, reaching almost similar levels of performance. We also focused on the hyperparameters search problem, which is more urgent problem in Reinforcement Learning than it is for Supervised Learning, due to the nature of the interactive learning. We utilize **Optuna** [6] implementation of some advanced hyperparameters search methods, and showed its effect on the results, and tracked all trails and experiments utilizing Mlflow [5], providing detailed overview of all the metrics. 

Beside that, we started each implementation section with an concise introduction to each package capabilities and main function, which will suffice to start utilizing them effectively for the readers projects. Lastly, the main experiments results, are shown and discussed and final take-away were stated.


<!-- In the following we will go over all the different libraries, highlighting their practical advantages or disadvantages and their main key difference from their alternatives. Starting with Gymnasium, MLflow, Optuna, and lastly Jax and Flax. We also will show the results of utilizing these libraries on our test program (explained below in Figure 4), focusing on performance (ability to converge optimally) and training time, on our hardware **NVIDIA GeForce RTX 5060 Ti**.-->

The installation of the packages needed in this post with `pip`-python can be done simply as follows:

```bash
pip install gymnasium
pip install mlflow
pip install optuna

#replace with your cuda version 
pip install "jax[cuda12]"
pip install flax
```

# Gymnasium: Standardize your environment 

[Gymnasium](https://gymnasium.farama.org/v0.29.0/) is an update of the popular Gym package developed originally by OpenAI [1]. It contains a set of standard simulated environments with unified interfaces, which undergo regular updates. This standardization is helpful for benchmarking different RL algorithms as well as for readability and collaboration. Among the other advantages motivating the usage of Gym and Gymnasium are:

- The ability to run **vectorized environments** (`VecEnv`): where multiple instances of the same environment are created and their states and actions can be processed in batches, which speeds up the rollout of trajectories in the environments and consequently training the RL agent as well. There's two methods to deploy vectorized environment in Gym: either as *Synchronized* or *Asynchronized* environments.  A comparison of the two is displayed in Table 1 below.

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
<td style="border: 1px solid black"> creates all environments in one thread serially and batch the output (state,reward,done flags) </td>
<td style="border: 1px solid black"> each environment is created with its own subprocess (computational thread) </td>
</tr>
<tr>
<td style="border: 1px solid black"> best used when environment process is simple, and faster than running independent subprocesses for each instance. </td>
<td style="border: 1px solid black"> best used when the environment processes are computationally expensive and there's enough memory for subprocesses. </td>
</tr>
<tr>
<td colspan='2' style="text-align: center;border: 1px solid black">
Input to both functions should be a list of creation functions of environments (for ex. with `lambda`).
</td>
</tr>
<tr >
<td colspan='2' style="text-align: center;border: 1px solid black">
If you set the optional key input (`shared_memory`) to True, then the output observation data will be referenced directly without copying, which can speed up the stepping when its size is large.
</td>
</tr>
</table>

</center>

- **Spaces objects**: used to define the state and action values and distributions. Namely, these spaces represent sets of specific constrains. Example of all the possible sets are shown in Figure 2 imported from `gymnasium.spaces`.

<center>
<br>
<img width="100%"src='/images/rlwithbells/gymspaces.png'>
<br>
Figure 2: Gymnasium basic and compound spaces.
</center>

- **Registry**: Custom environments can be registered within the installation so that they can be instanced directly like a standard Gym package (with `gym.make`). 

- `gymnasium.wrappers` contains useful classes to *modify* a specific environment behavior. Example of these wrappers include: 

    - `ObeservationWrapper`: Modify Observation space
    - `ActionWrapper`: Modify Action space
    - `RewardWrapper`: Modify Reward function
    - `TimeLimit`:  Used for truncating of an episode after a specific number of steps.
    - `Automaticreset`: When the environment reach a terminal state or get truncated, this wrapper reset in the next call to `.step()`. With that, the last observed state will be directly returned. 
    - `RecordEpisodeStatistics`: Important to collect episodic_rewards, which indicates the success or failure of a policy during training.

- If your environment is a subclass of `gymnasium.Env`, then you get the advantage of utilizing automatic testing with the function: `gymnasium.utils.env_checker.check_env`, which performs common tests on the gym environment methods and its spaces.


Additionally, newly introduced changes in Gymnasium over Gym include the following:

- Replacing `done` flag when stepping in the environment with `termination` and `truncation` flags. The difference is simple: *Termination is a natural ending point when the goal of the episode is achieved (for example: the goal is reached). Whilst truncation occurs only after a specific number of steps to avoid running the episode indefinitely.* Figure 3 depicts these differences.

<center>
<br>
<img width="80%"src='/images/rlwithbells/termination.png'>
<br>
Figure 3: difference between terminating (goal achieved) and truncating (time limit reached) a simulated episode.
</center>

- Introducing a new and experimental function for environment creation: `gymnasium.experimental.functional.FuncEnv()`, of purely functional structure (as the environment class is stateless) to reflect the formulation of POMDP (Partial Observable Markov Decision Process) more closely. Additionally, this structure should enable direct compatibility with JAX. 


## Example: Creating custom Gym environment and training it with DQN


We domenstarte the application of Gym (and the rest of the libraries) in this post utilizing a Grid world environment called *Doors* where an agent occupying a cell in that grid is assigned with the task of moving towards a goal cell passing through one of three gaps (doors) in a wall splitting the grid in two, as shown in Figure 4, which shows also the state-action configurations.

<center>
<br>
<img width="80%"src='/images/ilpost/Doors_.png'>
<br>
<br>
<img width="30%"src='/images/ilpost/gifs/expert.gif'>
<br>
Figure 4: Doors environment introduced in [previous post](https://www.rlbyexample.net/posts/hands-on-imitation-learning/). The lower image shows animation of optimal policy to solving it.
</center>



> Note: The full repository of the code is [available here](https://github.com/engyasin/ilsurvey), where the final script utilizing all the libraries [is here](https://github.com/engyasin/ilsurvey/blob/main/dqn_hopt_flax.py). 


We show below parts of the environment creation class in Gymnasium:

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


As the action space for this environment is discrete, we chose to use Deep Q-Network (DQN) training algorithm [3] based on CleanRL [4] implementation to learn the policy. In the following we will show how to track the training metrics and plot them against training time in Mlflow. 
<!-- ### Results -->


# MLflow: Tracking RL experiments 

[**MLflow**](https://mlflow.org/) is popular python library for tracking, versioning, collaborating and deployment of machine learning models. Its main functionality is to show the training metrics either in local server (by running `mlflow ui` in new terminal, with the default port as 5000) or in an online cloud server such as *Databricks*.

The way **MLflow** organizes training is by creating an *experiment* for each machine learning task (for example cat/dogs images classification). Within each experiment we can have many *runs*, which represent training trails for that task (for example different ML approaches for that task). Furthermore, smaller run can be *nested* inside major runs (which we will do for our hyperparameter trails below).

With that structure, **MLflow** allows comprehensive saving of all the testing parameters and metrics, and provides a unified interface to track them. Additionally, **MLflow** has seamless integration with pytorch, tensorflow, and Keras, and it also has many other functionalities and features that fall out of our scope here, but can be viewed at [its website](https://mlflow.org/).

- We can start a new experiment in MLflow by running `mlflow.create_experiment('experiment_name')` representing a new task for training ML model, or continuing working on an old experiment, that is already created, with the code:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"runs/{experiment_name}")

```
Note that you need to track the uri where the server will publish the results (in this case `http://localhost:5000`), while running the local server in another terminal with the command `mlflow ui`.

After that we can start a specific `run` within the experiment, or *multiple children runs* nested within the parent run (by utilizing the `nested` keyword argument). This last case is suitable, for example, if you are doing hyperparameters optimization where each trail can be tracked independently in its own child run. The following code shows that.

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

# Optuna: Optimization of RL hyperparameters 

Training Reinforcement Learning is known to require plenty of hyperparameters to tune, more than its supervised learning counter-parts. This makes it very beneficial for training to apply efficient hyperparameters optimization methods like Bayesian optimization [7] or Hyperband [16]. In the following sections, we will start by reviewing the most prominent methods of hyperparameters optimization, with focus on their implementation utilizing the `Optuna` package.


These hyperparameters in the case of RL, include parameters like: learning rate, episode length, discount factor (in Bellman equation), as well as the agent network depth and architecture.

## Types of Hyperparameters Optimization Methods:

Generally speaking, there's four main branches of hyperparameters optimization methodologies, varying in their complexity and approach, as the following figure shows.


<center>
<br>
<img width="100%"src='/images/rlwithbells/hyperopt.png'>
<br>
Figure 7: The main search methodologies for machine learning models hyperparameters.
</center>


### Uninformed Methods

These methods are the simplest of all, as they manually test directly different samples from the search space. Depending on their sampling strategy, they can be:

- Manual: Samples are chosen manually by the developer.
- Uniform: Samples are chosen uniformly with the range.
- Random: Samples are chosen randomly with the range.

### Bayesian Optimization methods

This category of methods utilize a surrogate model as an approximation of the **objective function** (*the function estimating the learning objective like accuracy or negative loss given the training hyperparameters*). The training data for that model are the values from the past training attempts. While updating the objective approximation model continuously after each training trail, the new set of hyperparameters to be tested will be proposed by another model called: **acquisition function**.

Based on the nature of that surrogate model, Bayesian Optimization (BO) methods [7] can be categorized into:

- Sequential Model-based Algorithmic Configuration (SMAC)[8]: utilizing random forest to approximate the objective function, which makes it suitable for categorical and discrete parameters search.

- Sequential Model-based Bayesian Optimization (SMBO) [9]: utilizing Gaussian Process model, suitable for continuous hyperparameters

- Tree-structured Parzen Estimators (TPE): utilizing random forest, suitable for large search space for both continuous and discrete search, with fast run-time. In `Optuna`, its implementation allow learning interactive relations between different hyperparameters. Its Optuna function  is: `optuna.samplers.TPESampler`.

- MATIS [10]: Gaussian Process-based, utilizing also a Gaussian Mixture Model as its acquisition function.


### Heuristic Search


This branch of methods samples the hyperparameters of its next training iteration in the neighborhood of the best set of hyperparameters found so far. Clearly the definition of this neighborhood has big impact on the search performance, where we have multiple variants:

- Simulated Annealing (SA) [11]: it searches for its next sample around the best or next-to-best set of values found so far, to avoid local minima.

- Genetic Algorithm [12]: It applies evaluations-inspired methods to select its next set of values. Namely, it is based on pairing the best samples found so far of different parameters, or mutating them.

- Particle Swarm Optimization [13]: This method focuses especially on the case of continuous hyperparameters.

- Population-based Training [14]: This method specializes in neural networks optimization, as it searches for both hyperparameters and normal training parameters as well. For example, it gradually adds new layers to the model under training after each training iteration, where the old trained layers are kept. However, it cannot recover the exact best hyperparameters for the best model, as it finds only the final trained model parameters.


### Multi-Fidelity Optimization (MFO)

This branch of methods adds another dimension to solving the problem of hyperparameters optimization, as it allows faster training by early stopping on not-so-promising samples, either by training on parts of the data, or for lower number of epochs (as the case in Optuna). This makes more sense than full training for all samples, as we don't need to invest computational resources in testing many samples of low probability of being optimal, while focusing less on areas of promising performance. The methods here try to shape this idea as recourses management algorithm.  It is also worth noting, that MFO methods can be combined directly with the previous sampling methods, as they address a different aspect of the problem. In Optuna, MFO methods are called **Pruners**,  and the sampling methods are called **Samplers**.

The most popular MFO methods include:

- Coarse to Fine Pruner: as the name suggests, this method starts by light training of many samples candidates, focusing increasingly on more promising subset of samples.

- Successive Halving (SH) [15]: This method distributes the computational resources wisely on the different training trails.

- Hyper Band (HB) [16]: This method defines pairs of candidates numbers and their allocated resources, called *brackets* and starts full training of some of these brackets to avoid early dropping of promising candidates mistakenly, as it can happen in SH due to shallow training. Its Optuna function is: `optuna.pruners.HyperbandPruner`.

- Bayesian Optimization Hyper Band (BOHB): It was noted that better results are obtained when the combination of BO sampler and Hyperband pruner, as the work in [17] details.  In Optuna this can be done by setting the sampler to TPE and the pruner to HB.


# Steps for doing Hyperparameters Optimization in Optuna:

Hyperparameters in RL training programs are many and have various effects on the training process; therefore, the task of tunning them manually require a lot of experience and tests to find good sets of values. That's why utilizing automatic search with well-tested implementations like Optuna [6] is a direct and fast way to save effort in practical applications. Optuna simplifies the process with clear implementation steps, based on its built-in support for most of the methods mentioned previously and it can be directly integrated with libraries like MLflow, Pytorch and JAX. Specifically, these steps can be summarized as follows:

1. Defining the objective function, which returns (in the case of RL) the average episodic return for episodes.
2. Inside that objective function, we define the hyperparameters ranges and types to be optimized using `optuna.trail.suggest_` group of functions.
3. Initializing the optimization object (called the *study*) with `create_study()` and within it, we define our `sampler` and `pruner` methods, in addition to the *direction* (defaults to minimizing).
4. Optionally, we can save the current training session by passing the `storage` argument (to `create_study()`) representing a database url to save the study object in. Additionally to resume training from saved session of trails, you can pass `load_if_exists=True` to that same function. 
5. We start the training with `.optimize()` method of the previous study object, passing the objective function (as callable) and the number of trails.
6. When the optimization ends, the best set of parameters can be shown (in `study.best_params`) and we can save the model.


It is also worth noting that `Optuna` also has a visualization module: `optuna.visualization` whose functions take the optimized study object as input and present many useful plots, like plotting the most influential hyperparameters on the results. This last module requires the installation of `plotly` package.


In the following we show some illustrative code snippet to implement the above steps.

## Training Code Structure in Optuna


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

In our accompanying [code repository](https://github.com/engyasin/ilsurvey), we performed 40 trails of searching for optimal hyperparameters, and visualized the results at the end, regarding:

- Parameters importance with `optuna.visualization.plot_param_importances`

<center>
<br>
<img width="100%"src='/images/rlwithbells/params_importance.png'>
<br>
Figure 8: Hyperparameters estimated relative importance of hyperparameters on the model training performance. The largest two most important parameters are the episode length and the learning rate.
</center>

- 2D heatmaps of interactive hyperparameters importance with `optuna.visualization.plot_contour`

<center>
<br>
<img width="100%"src='/images/rlwithbells/contours.png'>
<br>
Figure 9: 2D heatmaps of interactive pair-wise importance on the performance. We see here clearly that the darker regions are the best performing regions for that parameter.
</center>

- Performance of trails over time with `optuna.visualization.plot_optimization_history`

<center>
<br>
<img width="100%"src='/images/rlwithbells/optimization_history.png'>
<br>
Figure 10: Improvement of trails performance over order of training. We see clearly here that over time, the hyperparameter optimization was beneficial in learning better set of values to results in better performance. With further search we can expect that curve to continue his ascent.
</center>


> Lastly, we note that looking at these figures can help us estimate and understand the effective ranges or combination of ranges that result in the best performance. Possibly leading to more manual enhancement of other program parts which is not under optimization.

<!-- ### Optimizing the hyper parameters of the trained agent -->

<!-- ### Result -->

#  JAX & Flax: Speed up environment rollout and model training

The common option when training RL model utilizing simulated environment is to use Pytorch or Tensorflow for training the RL agent and Numpy/Gym for simulating your environment. However, an increasingly popular alternative to consider, replacing these libraries is a package developed by Google called  **JAX** [*(Just After Execution)*](https://docs.jax.dev/en/latest/index.html). JAX is faster way to run matrix computation efficiently (instead of `numpy`) and to train neural network models (instead of `pytorch`), due to its targeted exploitation of the hardware computational devices like GPUs and TPUs. While JAX can be utilized directly to update the neural network parameters; A JAX-based targeted package, like FLAX, can make life easier, when structuring your model and training algorithm. 

In the following, we will mention some of the key features of JAX, focusing on its Numpy-alternative functionalities, which we will demonstrate later by rewriting the same Doors Gym environment in JAX and comparing its run-time with the original. 

- JAX works by compiling the code with **XLA** *(Accelerated Linear Algebra)* compiler to statically typed expression language called **Jaxpr**. This compiled code run faster on CPUs, GPUs, and TPUs. Practically, after writing your JAX functions, you can compile them by passing them to `jax.jit()` function or by placing the decorator `@jax.jit` right above their definitions.

- JAX replaces most of Numpy functions with similar names so that modifying your numpy code is minimized. Mostly you should only replace `import jax.numpy as np` with `import numpy as np`. However, some other considerations should be also noted, as it is shown below:


> *Note*: JAX arrays, unlike numpy, are immutable. So they cannot by changed inplace. Instead we have to change them with the following code:

```python
import jax
arr = jax.numpy.arange(10)
arr = arr.at[1].add(2) # equivalent to arr[1] += 2 in numpy
```

> *Note*: JAX arrays don't throw an error (`OutofIndex`) if the index is out of its range, but default to giving the last item in the array.

> *Note*: JAX default precision is `float32` unlike Numpy's `float64`

> *Note*: JAX offers alternative functions of Scipy functions with `jax.scipy` 


The following code shows an example of JAX compatible function compiled with jit, measuring its runtime


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

- JAX can autovectorise any function with its `jax.vmap()` function (alteratively with `@jax.vmap` decorator). This is needed if you want to run a function or sequence of inputs: instead of looping through each input alone, you can pass these inputs as *batch* and get huge speed up over pur Python & Numpy code. Practically the input and output will be stacked and concatenated adding another dimension to their matrices (you can chose its place). We show below that this is also faster than Gymnasium way of environment vectorization.


- In JAX we can also vectorize functions across computational recourse, which allow parallel processing. This has the same implementation as `vmap` but by wrapping any function with `jax.pmap()` or the decorator `@jax.pmap`.


> *Note*: JAX execution is Asynchronized by default, this means that the code return directly before calculating the output of a function. To force it to wait, we should append any function call with `.block_until_ready()`.


- In addition to compiling with XLA, JAX can calculate gradients effectively by doing **automatic differentiation** *autodiff* of the calculations of all variables. This is very useful in speeding up training of neural networks.


- Control statements (*for, while, if, switch*) are known as performance bottleneck in Python. In JAX, they can be replaced as follows: 

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

> *Note*: For the code to be correctly compiled or vectorized in JAX, it should be exclusively *functional* only. Object oriented code (like stateful classes) cannot be compiled in JAX . However; stateless classes objects can be used, where they don't save any internal variables (or use them as static variables only). If these variables should be changed, then they are, by definition, part of the state.

> *Note*: This last restriction of functional code shouldn't be seen as a drawback. In fact, functional code is commonly considered more readable and a better structured form of the code. 

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

As you can see from the example above, the environment class is vectorized by definition, where we can pass the matrices of all actions stacked along the first dimension to step through multiple environment simultaneously. Namely, this is initialized in the `.reset()` function, by passing a corresponding number of random keys:

```python

    key = random.PRNGKey(0)
    NUM_ENVS = 24 # vmap
    keys = random.split(key,NUM_ENVS) # generate new keys from existing ones.

```
This vectorization has shown to be extremely advantageous in our tests. To confirm that, we tested the runtime for a range of DOORS environment counts doing the same operations, in JAX, Gym Synchronized, Gym Asynchronized, and JAX with accelerated looping between steps (which is usually slow in Python). The following figure plots the runtime as a function of the number of environments for these three methods.

<center>
<br>
<img width="100%"src='/images/rlwithbells/runtimeEnvs.png'>
<br>
Figure 11: Comparing runtime of different vectorization methods. JAX seems insensitive to number of environments running up to 500. Speeding up the for loop led to super fast performance of 0.07s.
</center>

**JAX-based environments don't seem to slow down by increasing environments instances.** This is very interesting note, because we can now increase our environment counts and speed up the rollout phase in a lot of RL training methods. The results and plotting script is available in the ´display.py´script in the accompanying repository where anyone can test it on its hardware. 
Additionally, we note that Synchronized was faster than the Asynchronized version, as the DOORS environment stepping is relatively simple compared to the overhead of spawning many subprocess.


<!-- compare run time with Gym: Plot (Number of envs, Time): Gym (list), Asyn, Sync, JAX  and how that impact the training -->

## FLAX

FLAX [20] is a JAX-based specialized library for building and training neural networks, which is regarded as faster and more readable library for deep learning than Pytorch or Tensorflow, due to its dependence on JAX.

We can also utilize another JAX-based library called (`optax`) [21], beside FLAX, for creating composable gradient transformation in JAX, while defining the model and training state in FLAX.

The definition of neural networks classes in FLAX is inherited from `flax.linen.Module`, where the forward inference of that network is expressed in its `__call__()` function with the annotation `@flax.linen.compact`. This means that the network creation interface in FLAX is object-oriented but still stateless and interpretable with JIT.

The following code is an example of defining a neural network in Flax, then passing a random input to it, as a necessary step to initialize its parameters. Note also that these parameters are required input for the model inference (with the `.apply()`) as it is stateless class.


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
print(model.tabulate(key2,random_data)) # shows the model structure

```

Another benefit in FLAX here is the automatic vectorization of the network functions, without the need to wrap it with `jax.vmap`, where the batch dimension defaults to the first dimension. 

After defining the network above, we can define the optimizer utilizing `optax`, and the training state class (organizing the training) as follows:

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

With the previous code we can update the parameters of the model, based on the loss. To save the final trained Flax model we write:


<!-- TODO: check if we can log flax model to mlflow and load them? -->

```python
with open(model_path, "wb") as f:
    f.write(flax.serialization.to_bytes(model.params))

# This code saves the model parameters in a data object, To load its parameters again use:

with open(model_path, "r") as f:
    q_state.params = flax.serialization.from_bytes(q_state.params, f.read())

```

> *Note*: `orbax` library is another higher level JAX-based package used to save Flax model automatically. 


<!--### Making the environment functional for JAX vectorization -->


<!-- ### Training DQN with Flax -->


<!-- ### Result : Speed comparison with Pytorch-->


# Final Take-away

<!-- comparing run time and performance -->

Table 2 below shows the performance (measured by the final step-wise mean reward of the last 2000 episodes-return during training (out of 5e5 training episode in total)) and the run-time of the training phase for three variants of training programs:

- Pytorch with Gym (Synchronized environments) [available here](https://github.com/engyasin/ilsurvey/blob/main/dqn_hopt_flax.py)
- FLax with Gym (Synchronized environments) [available here](https://github.com/engyasin/ilsurvey/blob/main/dqn_hopt_flax.py)
- FLax with Gym environment and JAX automatic vectorization on GPU [available here](https://github.com/engyasin/ilsurvey/blob/main/dqn_hopt_flax.py)

> Note that these results are calculated on **NVIDIA GeForce RTX 5060 Ti** as GPU and **AMD Ryzen 5 7600X 6-Core Processor** as CPU with 40 trails for the first and last tests, while the second test hyperparameters are copied from the best case of the last.

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
<td style="border: 1px solid black"> 0.71 </td>
</tr>
<tr>
<td style="border: 1px solid black"> Training Time</td>
<td style="border: 1px solid black"> 22.5 min </td>
<td style="border: 1px solid black">  22.8 min </td>
<td style="border: 1px solid black"> <strong>2.3 min</strong> </td>
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



We note from the results in Table 2, that hyperparameters optimization was helpful in finding a good model that reached good performance (0.73 with 40 trails) with Pytorch. The other programs with JAX and FLAX got close but still a bit worse which can be attributed to random initialization of the different packages. Doing more search trails can possibly lead to better results for all methods. Note also that increasing the buffer size in DQN (or any off-policy method) is important to take full advantage of the rollout speed up, otherwise performance can drop even with very fast environment.

The major improvement was in the training time when we replace normal Numpy operations in the DOORS environment with JAX accelerated and functional code. This can be attributed to the remark that increasing the number of environments do not affect the speed of JAX functional stateless classes. So, **we took advantage of that and increased the number of number of environments 16 times in the JAX-based implementation, which contributed to this huge speed-up of around 10 times.** We expect the possibility of larger speed-up if this number is increased further. The rest of the settings and hyperparameters ranges was the same for all three programs.


With that we conclude with final recommendations of when and why to use each of these packages:

1. Gymnasium: If you want to create new environment, and you care about sharing it and collaborating it with others. Then utilizing Gymnasium is a good decision toward that goal. 
2. MLflow: If you want comprehensive tracking of all of your training metrics and parameters, full display of the hyperparameters in your programs, and deployment over life-time, then utilizing MLflow is suitable and direct way to achieve that.
3. Optuna: If your model is complex and contains a lot of hyperparameters, which are hard to tune manually, (as the case is usually in Reinforcement Learning programs), Optuna can provide implementations of advanced hyperparameters search algorithms with direct compatibility with MLflow.

4. JAX: If your environment simulating is not fast enough and requires lengthy calculations, forming the bottleneck of your runtime, then vectorizing the environment with JAX eps. on GPU or TPU devices can results in a great boost for the training as bigger batches can be sampled faster.

4. FLAX: Being JAX-based library where gradient are calculated faster, Flax can results in high speed up on specialized devices. However; if the model size and dataset size are small (as in our case), this benefit diminishes. As we saw in Table 2, the training time is almost the same for that of Pytorch code. A case where Flax is very beneficial is when the observation space is huge (containing possibly images or videos) with many parameters to train. 


From that we can say that the training pipeline should be examined first to define where the computational bottleneck is, esp. in Model-free Reinforcement Learning with its two phases of rollouts generation and model parameters updating. For the first we recommend accelerated JAX matrices operations and for the second we recommend FLAX autodiff and optimizers.

# More Libraries in JAX 

To avoid reinventing the wheel, we mention below some open source JAX RL implementations covering many cases and algorithms, which can save time by directly editing the needed functionalities.

## Brax 

[Brax](https://github.com/google/brax) [22] is the JAX-based version of MujoCo, developed by Google, [(check our post here for an introduction of MujoCo)](https://www.rlbyexample.net/posts/immerse-yourself-in-reinforcement-learning-and-robotics-with-mujoco/). It shows great speed-up over standard MujoCo, and provides implementations of SAC and PPO RL algorithms

## Dopamine

[Dopamine](https://github.com/google/dopamine) [23] is another Google-developed package providing a JAX implementation for a variety of RL algorithms, allowing fast training and testing on different environments. 


# References

1. Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). Openai gym. arXiv preprint arXiv:1606.01540.
2. Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T., ... & Younis, O. G. (2024). Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032.
3. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
4. Huang, S., Dossa, R. F. J., Ye, C., Braga, J., Chakraborty, D., Mehta, K., & AraÃšjo, J. G. (2022). Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms. Journal of Machine Learning Research, 23(274), 1-18.
5. Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., ... & Zumar, C. (2018). Accelerating the machine learning lifecycle with MLflow. IEEE Data Eng. Bull., 41(4), 39-45.
6. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019, July). Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 2623-2631).
SMBO
7. Frazier, P. I. (2018). A tutorial on Bayesian optimization. arXiv preprint arXiv:1807.02811.
TPE
8. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. Advances in neural information processing systems, 24.
SMAC
9. Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2011, January). Sequential model-based optimization for general algorithm configuration. In International conference on learning and intelligent optimization (pp. 507-523). Berlin, Heidelberg: Springer Berlin Heidelberg.
METIS
10. Li, Z. L., Liang, C. J. M., He, W., Zhu, L., Dai, W., Jiang, J., & Sun, G. (2018). Metis: Robustly tuning tail latencies of cloud systems. In 2018 USENIX Annual Technical Conference (USENIX ATC 18) (pp. 981-992).
SA
11. Kirkpatrick, S., Gelatt Jr, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. science, 220(4598), 671-680.
GA
12. Di Francescomarino, C., Dumas, M., Federici, M., Ghidini, C., Maggi, F. M., Rizzi, W., & Simonetto, L. (2018). Genetic algorithms for hyperparameter optimization in predictive business process monitoring. Information Systems, 74, 67-83.
swarm 
43. Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization. In Proceedings of ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). ieee.
population
14. Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., ... & Kavukcuoglu, K. (2017). Population based training of neural networks. arXiv preprint arXiv:1711.09846.
SH
15. Pietruszka, M., Borchmann, Ł., & Graliński, F. (2021, May). Successive halving top-k operator. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 18, pp. 15869-15870).
HB
16. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal of Machine Learning Research, 18(185), 1-52.
BOHB
17. Falkner, S., Klein, A., & Hutter, F. (2018, July). BOHB: Robust and efficient hyperparameter optimization at scale. In International conference on machine learning (pp. 1437-1446). PMLR.
18. Imambi, S., Prakash, K. B., & Kanagachidambaresan, G. R. (2021). PyTorch. In Programming with TensorFlow: solution for edge computing applications (pp. 87-104). Cham: Springer International Publishing.
19. Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., ... & Zhang, Q. (2021). JAX: Autograd and xla. Astrophysics Source Code Library, ascl-2111.
20. Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., Steiner, A., & Van Zee, M. (2020). Flax: A neural network library and ecosystem for JAX. Version 0.3, 3, 14-26.
21. DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio (2020). The DeepMind JAX Ecosystem https://github.com/google-deepmind
22. Freeman, C. D., Frey, E., Raichuk, A., Girgin, S., Mordatch, I., & Bachem, O. (2021). Brax--a differentiable physics engine for large scale rigid body simulation. arXiv preprint arXiv:2106.13281.
23. Castro, P. S., Moitra, S., Gelada, C., Kumar, S., & Bellemare, M. G. (2018). Dopamine: A research framework for deep reinforcement learning. arXiv preprint arXiv:1812.06110.


<!-- TODO: Turn Code into images -->
<!-- TODO: Make other images -->
<!-- TODO: Add experiments results -->
<!-- TODO: Mention the hardware -->
<!-- TODO: Add references -->


<!-- TODO: Edit by yourself then LLM then yourself -->
<!-- TODO: More on Tree-structured Parzen Estimators -->
 