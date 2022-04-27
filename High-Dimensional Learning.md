# Context

1. Basic of Statistical Learning: decomposition of error
2. The Curse of Dimensionality: adversarial phenomena that emerges as the input space become high dimensional
3. Addressing the curse: signals



# Statistical Learning

## Data distribution

1. Distribution function $$v$$ generating $$x_{i}$$

2. A target unknown function $$f^{*}(x_{i})$$ generating the labels $$y_{i}$$

In order to analyse Machine Learning algorithms and provide guarantees, we need assumptions on both $$v$$ and $$f$$. If there aren't assumptions on the data coming either from the distribution or the target function, this implies that there's no way we can actually **generalize**.

## Model

**Synonyms**: _hypothesis class, function approximation_

- Polynomials of degree $$k$$
- Neural Networks

**Complexity measure**: Some type of **norm** or **quantity** that you can evaluate in your **hypothesis** that is meant to divide your hypothesis into into that are **simple** and those that are **complicated**. _e.g_: how many neurons we have in our neural network?

**Sobolev Norm**

**Learner**: Try to find hypothesis with low complexity.

## Error Metric

**Idea**: Notion of **comparing** or measuring errors quantitatively. 

**Loss function**: For example measuring the **square** distance (**M**ean **S**quared **E**rror). 

**Energy functions**.

**Point-wise** measure.

Notion of **average**:

1. **Population average**: **expectation** for the data of this point wise measure. How well we are gonna do.
2. **Empirical average**: commonly named the **training loss**.

**Conclusion**: We are gonna try to reduce the population error but we can only work with the empirical error. **How these 2 notions of average relate to each other?**

**How far is ML from classical statistics**: I can't afford the luxury of **fixing** my hypothesis. In the training era we are gonna be updating our initial guess for $$f^{*}$$.

**Training** and **test** eras can't be related point-wise.

Not see it as a random variable, but rather as a **expectation of a random function**. Training and test eras as **objects**, as **functions**, they should be **close** to each other. **Redemacher complexities**

## Empirical Risk Minimization: The Algorithm

**Supervised learning** sense.

Focus on **low complexity** hypothesis class.

Consider an **algorithm** as an **estimator**.

**Convex optimization**: See also convex **constraints**. The constraint form is **not easy** to use and that's why we can also consider the **penalized** form.

**Penalized form**: Introduces a **Lagrangian multiplier** where the **constraint** now becomes part of the **optimization objective**.

**Hyper parameter**: Lambda nows control indirectly the strength of the **regularization**. 

$$\delta$$ And $$\lambda$$ can be thought as being a **dual guard**.

**Interpolation form**: Popularized by **large** neural networks. Functions hugely **expresive** that we can even **completely fit** the data. This implies an **empirical risk** of value **0**.
**Strong assumption** for this method to work: If and only if we assume that there's **no noise** in the data. This **luxury** only occurs in **certain** data sets.

## Basic Descomposition of Error

**Idea**: It's always the same. We have something that we want to **control**. The way to **introduce** to actually make **progress** is always the same, **add** and **subtract** the appropriated quantity.

**Guarantee** in the performance in the **test set** once an arbitrary hipothesis has been chosen.
$$
\mathcal{R}(\hat{f}) - \textcolor{Yellow}{inf_{f \in \mathcal{F}} \mathcal{R}(f)} =
$$


**Infimun error**: The best we could do a **posteriori**. If we had an **oracle** who could give us as many samples as we want while having theoretical infinity amount of computational resources, we could effectively have selected the **best** hypothesis.

To further interpreter the previous difference, we need to **transform** it into **multiple differences**.

Add and subtract the **best** we can **do** by **resctricting the complexity**
$$
inf_{f \in \mathcal{F}_\delta}\mathcal{R}(f)
$$

$$
= \big ( \mathcal{R}(\hat{f}) - inf_{f \in \mathcal{F}_\delta}\mathcal{R}(f)\big )
+ \textcolor{Red}{(inf_{f \in \mathcal{F}_\delta}\mathcal{R}(f) - inf_{f \in \mathcal{F}} \mathcal{R}(f)}
$$
Now we have one object that we can start to interpret. So this red object is telling us the difference between the error If we minimise it over the **full class** minus the error we get if we only look at the complexities of size $$\delta$$.
$$
inf_{f \in \mathcal{F}} \mathcal{R}(f)
$$
In this term there is **no empirical** information, there isn't any **hats**, it's a **pure** term. Thus is completely connected with what we call **approximation**. If $$\delta$$ gets immensely high, the term is going to become **smaller**. Therefore it can be conceived as the **approximation error**.
$$
\textcolor{Red}{(inf_{f \in \mathcal{F}_\delta}\mathcal{R}(f) - inf_{f \in \mathcal{F}} \mathcal{R}(f)} = \epsilon_{appr}
$$


Now we are going to introduce the **empirical objective**, the **test error** over the "ball":
$$
\hat{\mathcal{R}}(\hat{f}) - inf_{f \in \mathcal{F}_\delta}\mathcal{\hat{R}}(f)
$$
Thus the decomposition progress as follows,
$$
\textcolor{Green}{\big (\hat{\mathcal{R}}(\hat{f}) - inf_{f \in \mathcal{F}_\delta}\mathcal{\hat{R}}(f) \big ) +}
(\mathcal{R}(\hat{f}) - \hat{\mathcal{R}}(\hat{f})) 
+ \big ( inf_{f \in \mathcal{F}_\delta}\mathcal{\hat{R}}(f) - inf_{f \in \mathcal{F}_\delta}\mathcal{R}(f)
\big ) + \textcolor{red}{\epsilon_{appr}}
$$
We also add and subtract the **infimun of the training error**:
$$
inf_{f \in \mathcal{F}_\delta}\mathcal{\hat{R}}(f)
$$
**Interpretation** of the current state of the terms inhabiting the decomposition expression.
$$
\textcolor{Green}{\big (\hat{\mathcal{R}}(\hat{f}) - inf_{f \in \mathcal{F}_\delta}\mathcal{\hat{R}}(f) \big )} = \epsilon_{opt}
$$
It's the **training error** of my hypothesis **minus** the **best** training error in my **ball**. So if we are able to solve the **empirical** base **minimization problem**, how much does this term cost?

This term is going to be **zero**. If we are good at optimising things, then this **green** term is going to be **small**.

**Interpretation**: Optimization error.

**2 terms remain**:

1. $$(\mathcal{R}(\hat{f}) - \hat{\mathcal{R}}(\hat{f})) $$
	Point-wise difference.
2. $$( inf_{f \in \mathcal{F}_\delta}\mathcal{\hat{R}}(f) - inf_{f \in \mathcal{F}_\delta}\mathcal{R}(f))$$ 

**Interpretation**: They really look like something that **relates** the **population** objective with the **training** objective.

We can unify these two terms together and **upper bound** them:
$$
\textcolor{Cyan}{2sup_{f \in \mathcal{F}_{\delta}}|\mathcal{R}(f) - \mathcal{\hat{R}}(f)|}
$$
This term is two times the **largest fluctuation** between the **training** and the **test** over our "ball" of available hypothesis.

**Conceptualization**: We can clearly see that if we have **2 functions** and we want to minimize them, the **difference** between, we can always **upper bound** the **difference**.

Thus our expression over the **error**, or test error, looks as follows:
$$
\textcolor{Green}{\leq \: \epsilon_{opt}}
+ \textcolor{Cyan}{2sup_{f \in \mathcal{F}_{\delta}}|\mathcal{R}(f) - \mathcal{\hat{R}}(f)|}
+ \textcolor{Red}{\epsilon_{appr}} \\
= \textcolor{Green}{\epsilon_{opt}} + \textcolor{Cyan}{\epsilon_{stat}} + \textcolor{Red}{\epsilon_{appr}}
$$
The error made is a **contribution** of **three** different **sources** of error.

1. $$\textcolor{Green}{\epsilon_{opt}}$$ (**Optimization error**): It measures our ability to solve this empirical risk minimization efficiently.
2. $$\textcolor{Cyan}{\epsilon_{stat}}$$ (**Statistical error**): Penalizes uniform fluctuations over the "ball" between the **true function**, the **test** function, and the **random function**, which is the **training** error.
3. $$\textcolor{Red}{\epsilon_{appr}}$$ (**Approximation error**): How well we can approximate the **target function** $$f^{*}$$ with small complexity.

### Missing term

$$
\mathcal{R}(\hat{f}) 
\leq 
\textcolor{Yellow}{inf_{f \in \mathcal{F}} \mathcal{R}(f)} + \textcolor{Green}{\epsilon_{opt}} + \textcolor{Cyan}{\epsilon_{stat}} + \textcolor{Red}{\epsilon_{appr}}
$$

- $$\textcolor{Yellow}{inf_{f \in \mathcal{F}} \mathcal{R}(f)} = 0$$ if $$\mathcal{F}$$ is dense. _e.g._ neural networks with non-polynomial activation (Universal Approximation Theorems) 
- Approximation error: **Exploit** as much as we can the hypothesis like the prior information we have on the target function.
- Statistical error: 
- Optimization error: Solve these problems in a efficient way.


**Conclusion**: If we want to be able to learn in high-dimensions we need to be good at these **three errors** at the same time.

**Appendix questions**: 
Question: Does the **empirical error** always need to have a **convex constraint**?

Answer: For sake of simplicity always assume the hypothesis space to be convex. In the context of neural networks this is synonymous to considering the **last layer** to be hugely **wide** (as wide as you can). Nevertheless even if our hypothesis space **wasn't  convex**, our previous decomposition of error stills **holds**. All the presented equations are **dimension-free**.

There exists one hyper parameter in the generated expression for the descomposition of error $$\delta$$. This implies that the **learner** can effectively **chose** it.

Question: How the hyper parameter $$\delta$$ Depends on the dimensionality of the input space?

Answer: It depends to the **dimensionality**, but it also depends on finer **properties** of the **functional class**.

#### How to simultaneous control all sources of error in the high-dimensional regime?

# The curse of dimensionality

## Statistical perspective

Question: How the terms that emerged in our descomposition of error behave as a function of the dimensionality of the input space?

**Dynamic programming** ->  synonym: High-dimensional statistics.

**Basic Principle of Learning: Intepolation**

Propagate the **information** that we **observed** to the propagation form the neighbors.

**Similarity**: The principle of learning by basically **finding patterns** that are similar, it's something that suffers a lot in high-dimensions.

### Learning Lipschitz Functions: Understand the role of locality in learning

Encapsulates the notion of **locallity**. It's a hypothesis of a function that only depends on locality. Thus the value of the function at one point is going to be close to the value of the function at a neighbour point.
$$
f: \mathcal{X} \subseteq \R^{d} \to \R \: is \: \beta Lipshitz \: if \\
|f(x) - f(x')| \leq \beta \Vert x - x'\Vert
$$
**Statement**: If $$x$$ and $$x'$$ are small then $$f(x)$$ and $$f'(x)$$ are close to each other.

### Number of samples needed to learn given an arbitrary input space with d-dimension

**Setup**:
$$
\{ (x_i, f^{*}(x_{i})\}_{i=1...n}
$$
Our hypothesis space is going to be all the functions that
$$
F = \{ f: \R^{d} \to \R, f \: bounded, f \: Lipshitz\}
$$
This introduced space can indeed be shown to be an [Bannach space](https://ncatlab.org/nlab/show/Banach+space). This conclusion means that the emerged space has a notion of complexity, norm. Now we can define our estimator.

**Estimator**: ERM is the interpolant form.
$$
\hat{f} = argmin_{f \in \mathcal{F}} \Big \{ Lip(f), f(x_i) = f^{*}(x_i) \forall i\Big \}
$$
This implies we are going through all the points.

Question: How do we complete the error between this estimator and the ground truth?

**Pick** $$x \sim v$$
$$
|\hat{f}(x) - f^{*}(x)|
$$
Given a **point cloud**, the $$x_{i}$$ values in our space, and an value $$x$$ 

**Mechanism to compute bounds in machine learning**: Add and subtract trick.

We are going to consider the **point** that is **closest** from the point cloud to $$x$$. We denote this point as $$x_{i0}$$

Now we add and subtract it. Thus the previous expression becomes:
$$
|\hat{f}(x) - f^{*}(x)| \leq |\hat{f}(x) - \hat{f}(x_{i0})| 
+ |\hat{f}(x_{i0} - f^{*}(x_{i0})| + |f^{*}(x_{i0}) - f^{*}(x)|
$$
We need to **bound** these terms. The expression can be simplified by substituting the term
$$
|\hat{f}(x_{i0} - f^{*}(x_{i0})| 
$$
By **0**. Because this point has been picked from the **training set** and by definition we know that our **interpolant** passes through all the points. Thus this term value is 0 by construction. The last term,
$$
|f^{*}(x_{i0}) - f^{*}(x)|
$$
force us to use the hypothesis $$f^{*}$$. $$f^{*}$$ and $$\hat{f}^{*}$$ are both **selected** because we chosen them to **minimize** the  **Lipschitz constant**. In particular it isn't only a Lipschitz constant, but the Lipschitz constant of $$\hat{f}$$ is at most the one of $$f^{*}$$ because $$f^{*}$$ is an interpolant. This is
$$
\leq 2 \Vert x_{i0} - x \Vert
$$
two times the distance between $$x_{i0}$$ minus $$x$$.

In conclusion we have, having fixed the Lipschitz constant to be **one**,
$$
\mathbb{E}_{x} |\hat{f}(x) - f^{*}(x)|^{2} 
\leq 4 \mathbb{E}_{x} \Vert x - x_{io}\Vert^{2}
$$
This defined as the closest point from the constant.

**Optimal transport and the exact Wasserstein loss**

The term $$4\mathbb{E}_{x} \Vert x - x_{io}\Vert^{2}$$ is a well known quantity, called **optimal transport distance**.
$$
4\mathbb{E}_{x} \Vert x - x_{io}\Vert^{2} = 4W_{2}^{2}(v, \hat{v}_{n})
$$


**Wasserstein distance** ([distance between two Gaussians](https://ncatlab.org/nlab/show/Wasserstein+metric)): Given a data distribution, sample of size $$n$$, and a newly introduced point in the sample, is defined as the **minimun** distance from this new point to one of the points inhabitants of the sample. The implications of the growth of the dimensionality over this quantity states as follows:
$$
4\mathbb{E}_{x} \Vert x - x_{io}\Vert^{2} = 4W_{2}^{2}(v, \hat{v}_{n})
\sim n^{-1/d} = \epsilon
$$
Since we want to make the previous expression equal to epsilon, implies that the epsilon needs to be,
$$
4\mathbb{E}_{x} \Vert x - x_{io}\Vert^{2} = 4W_{2}^{2}(v, \hat{v}_{n})
\sim n^{-1/d} = \epsilon \\
\implies n \sim \epsilon^{-d}
$$
**Conclusion**: The lower bound of needed samples it's actually the **necessary** amount of samples in order to properly learn.

## Pure approximation perspective

## Optimization perspective

Read the space. This means we need to just **evaluate** every possible point and **find** the **smallest** value. This of course has **exponential dependency in dimension**.

Exponential blow up of complexity. Thus we need to make **assumptions**.

This is overcome in practice working with spaces that are nearly **convex**. They possess **no bad local minima**. 

Instead of finding a **global minima** we focus about finding a **local minimum**.

Local minimum, formally called **second-order stationary points**.

Question: **How hard is find a local minimum in high-dimensions?**

Answer: Easier than finding a global. Quantitatively in terms of **iteration complexity** and having an **error** of $$\epsilon$$, we need a number of iterations that is of the order of
$$
\tilde{\mathcal{O}}(\beta/\epsilon^{2})
$$
iterations. The notation $$\tilde{\mathcal{O}}$$ means that it's hiding **log factors**. Thus might exists terms which depend on dimension but only **logarithmically**.

**Strong assumption of non bad local minima**: This might not always be the case.
