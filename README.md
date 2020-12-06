# GNN-TiborMaxTiago

## Abstract
For modern transportation systems such as highway- or railway networks, traffic
forecasting is a crucial tool for planning and operation. For instance, traffic
flow prediction is used in navigation systems like "Google-Maps"[cite] to
estimate travel time and propose optimal driving routes. In town planning,
traffic analysis can help to identify where road-infrastructure is overloaded
and thus determine where new roads need to be built. 

Predicting traffic speed on highway- and road networks can be organically
formulated as a graph learning problem. Here, nodes represent sensors that
capture speed of the traffic in different locations of a network. Edges are
denoted by road segments which connect the sensors. Previous works have
employed graph neural networks to solve this learning problem (cite). While
early “Graph Neural Network” (GNN) architectures suffered from high
computational complexity [cite early GCN model], later works concentrated on
capturing spatial and temporal patterns simultaneously [DCRNN, Traffic GCN].

In this work we employ three widely used benchmark traffic dataset to compare
classical graph- and deep learning algorithms specific to the task of traffic
forecasting. As a result of these comparisons, we identify limitations to the
models and propose a novel approach to capture spatial and temporal
dependencies more efficiently. Additionally, we study how the construction of
the graph structure influences the performance of GNNs. Based on these results
we discuss how aforementioned methods can be used to infer traffic flow from
traffic speed prediction.

