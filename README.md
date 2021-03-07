# GNN-SpaceTimeGraphs

## Abstract
In Intelligent Transport Systems (ITS), traffic forecasting is a crucial tool to improve road security, planning and operation. Before using neural architectures, autoregressive models were employed for time-series forecasting which faced difficulties to model highly non-linear and spatially dependent traffic data.
Speed sensors in road networks are arranged in graph like structures, therefore, spatial and temporal dependencies are often modeled based on traffic graphs. Because relationships of sensors are modeled in space and time concurrently, the effectiveness of each mechanisms needs to be isolated when comparing neural network architectures. Contrary to a formulation where edges in a traffic graph are predefined through physical road connections or closeness in space, there is a trend towards refining the structure of traffic graphs during the learning process.
We propose a series of experiments based on spectral graph convolution using a concept introduced by Zhang et. al (AAAI 2020), which regards the graph laplacian as a learnable parameter. We compare this setup to one that uses a static laplacian.
Additionally we use a latent correlation layer as proposed by Chao et. al (NeurIPS 2020) as another way of learning the laplacian.
To keep the variants of the spectral convolution comparable the temporal modeling component stays fixed.
The contributions of this work can be summarized answering the following two research questions (RQ):
- RQ1: How does learning the graph structure affect the precision of predictions in graph neural networks for traffic forecasting?
- RQ2: Do graph convolution operators benefit from having the graph structure as a learnable parameter? 
We employ two widely used benchmark datasets and compare different setups to answer RQ1 and RQ2. We were able to reproduce results shown by Zhang et. al and extend the comparison to models that utilize a latent correlation layer.

