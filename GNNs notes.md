# Graph Convolutional Neural Networks

They take absolute advantage of the graph structure.

There is no explicit representation of these mechanisms, and their further understanding is unclear.

The idea of **message-passing** in a graph is a compelling concept. From it, we can derive most of the graph algorithms in existence.



**Message passing in a nutshell:** A node in the graph can send and receive messages by connecting with its neighbors.



1. A particular node will dispatch a message about itself to its neighbors.
2. The neighbors of the initial node received its representation information included in the message. This information is used to update themselves and better understand their environment.



**The essence of the label propagation algorithm**: It can be conceptualized as smoothing the label information across the entire neighborhood. (**label smoothing**)



They can be understood as a simple message passing algorithm, sending the entire input node vector representation to its neighbors. (**feature smoothing**)



**Graph convolutional neural networks step-by-step illustration of the mechanism by a simple computation through one layer**:



1. We take the whole neighborhood node feature vectors for any node in the graph and then aggregate them (We will choose the aggregator operator attending to our needs). The node feature vectors must have identical dimensions, but the aggregator operator is agnostic of how many neighbors might exist. Here for the sake of simplicity, we apply the average. Thus a particular node feature representation is conditional to its neighbors, in this case, following their average. (**preprocessing**)
2. Pass this average feature vector through a dense neural network layer (we multiply it by some matrix and then apply an activation function, **a non-linear function,** to the generated output of this dense layer). The output produced by applying this dense layer will be the new feature vector representation for the initial node.

**Note:** We execute this process for all the nodes in the graph. We repeat the same process as described above for a GCN with more than one layer, but now the input vector for a layer n + 1 is the previously generated feature vector of layer n. (**traditional deep neural network**). 



**Conclusion:** In contrast to a classical neural network binary classification, we have these additional steps of aggregating the neighbors between the layers.



**Worth emphasizing:** The number of GCN layers establishes an **upper bound** on how far a signal can span from a node. The number of layers in the GCN is equivalent to the number of hops messages can reach. Thus if we are concerned about long-range interactions, we must use deep neural networks.