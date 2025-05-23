
# Why Do More Experts Fail? A Theoretical Analysis of Model Merging

> Pytorch implementation.

## Overview


Model merging dramatically reduces storage and computational resources by combining multiple expert models into a single multi-task model. Although recent model merging methods have shown promising results, they struggle to maintain performance gains as the number of merged models increases. In this paper, we investigate the key obstacles that limit the scalability of model merging when integrating a large number of expert models.
First, we prove that there is an upper bound on model merging. Further theoretical analysis reveals that the limited effective parameter space imposes a strict constraint on the number of models that can be successfully merged. Gaussian Width shows that the marginal benefit of merging additional models diminishes according to a strictly concave function. This implies that the effective parameter space becomes rapidly saturated as the number of merged models increases. Furthermore, using Approximate Kinematics Theory, we prove the existence of a unique optimal threshold beyond which adding more models does not yield significant performance improvements. At the same time, we introduce a straightforward Reparameterized Heavy-Tailed method (RHT) to extend the coverage of the merged model, thereby enhancing its performance. Empirical results on 12 benchmarks, including both knowledge-intensive and general-purpose tasks, validate our theoretical analysis. We believe that these results spark further research beyond the current scope of model merging.

  
## Usage

### Prerequisites
- Python 3.10
- [Pytorch (>=2.1.0) and torchvision](https://pytorch.org/)
- CUDA 12.0 or above

  

### Run the Code

1. Command as follows
~~~~
python ./scripts/RHT.sh
~~~~
