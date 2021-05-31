# Multi-domain Multi-task Rehearsal for Lifelong Learning
### 

## Abstract
Rehearsal, seeking to remind the model by storing old knowledge in lifelong learning, is one of the most effective ways to
mitigate catastrophic forgetting, i.e., biased forgetting of previous knowledge when moving to new tasks. However, the
old tasks of the most previous rehearsal-based methods suffer from the unpredictable domain shift when training the new
task. This is because these methods always ignore two significant factors. First, the Data Imbalance between the new
task and old tasks that makes the domain of old tasks prone to shift. Second, the Task Isolation among all tasks will make the
domain shift toward unpredictable directions; To address the unpredictable domain shift, in this paper, we propose MultiDomain Multi-Task (MDMT) rehearsal to train the old tasks and new task parallelly and equally to break the isolation
among tasks. Specifically, a two-level angular margin loss is proposed to encourage the intra-class/task compactness and
inter-class/task discrepancy, which keeps the model from domain chaos. In addition, to further address domain shift of the
old tasks, we propose an optional episodic distillation loss on the memory to an chor the knowledge for each old task.Experiments on benchmark datasets validate the proposed approach can effectively mitigate the unpredictable domain shift.
## Requirements

TensorFlow >= v1.9.0.
The code is based on https://github.com/facebookresearch/agem.

## Training

To replicate the results of the paper on a particular dataset, execute (see the Note below for downloading the CUB and AWA datasets):
```bash
$ ./replicate_results.sh <DATASET> <THREAD-ID> 
```

Example runs are:
```bash
$ ./replicate_results.sh MNIST 4     /* Train MEGA on MNIST */

$ ./replicate_results.sh CIFAR 3     /* Train MEGA on CIFAR */

$ ./replicate_results.sh CUB 3 0   /* Train MEGA on CUB */

$ ./replicate_results.sh AWA 7 0    /* Train MEGA on AWA */
```

### Note
For CUB and AWA experiments, download the dataset prior to running the above script. Run following for downloading the datasets:

```bash
$ ./download_cub_awa.sh
```
The plotting code is provided under the folder `plotting_code/`. Update the paths in the plotting code accordingly.

