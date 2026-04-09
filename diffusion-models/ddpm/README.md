# Denoising Diffusion Probabilistic Model (DDPM)

A minimal implementation of DDPM following [Ho et al. (2020)](https://arxiv.org/pdf/2006.11239), with an epsilon-prediction objective and ancestral sampling.

## Theory

A brief walkthrough of the DDPM, following the content from the textbook.

### Forward process (fixed)

The process used in noising a data sample into a noise from Gaussian.

Let a data sample be $\mathbf{x}=\mathbf{x}_0$ and the number of iterations $L$.

Then, the transition is defined by $p(\mathbf{x}_i | \mathbf{x}_{i-1}) := \mathcal{N}(\mathbf{x}_i ; \alpha_i \mathbf{x}_{i-1}, \beta_i^2 \mathbf{I})$,
where $\alpha_i := \sqrt{1 - \beta_i^2}$ and $\beta_i \in (0,1)$ is a pre-defined noise parameter.

Alternatively, the above could be written as $\mathbf{x}_i = \alpha_i \mathbf{x}_{i-1} + \beta_i \boldsymbol{\epsilon}_i$ where $\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

Since an affine transformation of a normal distribution is again a normal distribution,
we can obtain the closed-form expression for the distribution at step $i$ given the data sample $\mathbf{x}_0$ by recursively applying the transitions:
$$\mathbf{x}_i = \overline{\alpha}_i \mathbf{x}_0 + \sqrt{1 - \overline{\alpha}_i^2} \boldsymbol{\epsilon},$$
where $\overline{\alpha}_i := \prod_{k=1}^i \alpha_k$.

Assuming that $\{ \beta_i \}_{i=1}^L$ is an increasing sequence, $\overline{\alpha}_i \rightarrow 0$ as $L \rightarrow \infty$, since
$0 \leq \prod_{k=1}^\infty \sqrt{1 - \beta_k^2} \leq \prod_{k=1}^\infty \sqrt{1 - \beta_1^2} = 0$;
which indicates $p_L(\mathbf{x}_L | \mathbf{x}_0) \rightarrow \mathcal{N}(\mathbf{0}, \mathbf{I})$ as $L \rightarrow \infty$.

### Reverse denoising process (trainable)

The process used in denoising a Gaussian noise into a point from the data distribution.

This is the problem of approximating $p(\mathbf{x}_{i-1} | \mathbf{x}_i)$ using a parametric model $p_{\boldsymbol{\phi}} (\mathbf{x}_{i-1} | \mathbf{x}_i)$.

To avoid computing the intractable true marginal $p_i (\mathbf{x}_i) = \int p_i (\mathbf{x}_i | \mathbf{x}_0) p_{\text{data}}(\mathbf{x}_0) d\mathbf{x}_0$,
DDPMs additionally condition the reverse transition kernel on $\mathbf{x} = \mathbf{x}_0$:
$$
p(\mathbf{x}_{i-1} | \mathbf{x}_i, \mathbf{x})
= p(\mathbf{x}_i | \mathbf{x}_{i-1}, \mathbf{x}) \frac{p(\mathbf{x}_{i-1} | \mathbf{x})}{p(\mathbf{x}_i | \mathbf{x})}
= p(\mathbf{x}_i | \mathbf{x}_{i-1}) \frac{p(\mathbf{x}_{i-1} | \mathbf{x})}{p(\mathbf{x}_i | \mathbf{x})},
$$
where the second equivalence is due to the Markov property of the forward process.

From Theorem 2.2.1 in the textbook, it is known that
$\mathbb{E}_{p_i(\mathbf{x}_i)} \left[ \mathcal{D}_{KL} \left( p(\mathbf{x}_{i-1} | \mathbf{x}_i) \Vert p_{\boldsymbol{\phi}} (\mathbf{x}_{i-1} | \mathbf{x}_i) \right) \right]$ is equivalent to
$\mathbb{E}_{p_{\text{data}}(\mathbf{x})} \mathbb{E}_{p(\mathbf{x}_i|\mathbf{x})} \left[ \mathcal{D}_{KL} \left( p(\mathbf{x}_{i-1} | \mathbf{x}_i, \mathbf{x}) \Vert p_{\boldsymbol{\phi}} (\mathbf{x}_{i-1} | \mathbf{x}_i) \right) \right]$ up to a constant independent of $\boldsymbol{\phi}$,
and the minimizer satisfies
$p^* (\mathbf{x}_{i-1} | \mathbf{x}_i) = \mathbb{E}_{p(\mathbf{x}|\mathbf{x_i})} \left[ p(\mathbf{x}_{i-1} | \mathbf{x}_i, \mathbf{x}) \right] = p(\mathbf{x}_{i-1} | \mathbf{x}_i)$.

Importantly, $p(\mathbf{x}_{i-1} | \mathbf{x}_i, \mathbf{x})$ has the closed-form expression:
$$
p(\mathbf{x}_{i-1} | \mathbf{x}_i, \mathbf{x}) = \mathcal{N} \left( \mathbf{x}_{i-1} ; \boldsymbol{\mu}(\mathbf{x}_i, \mathbf{x}, i), \sigma^2(i) \mathbf{I} \right),
$$
where
$$
\boldsymbol{\mu}(\mathbf{x}_i, \mathbf{x}, i) :=
\frac{\overline{\alpha}_{i-1} \beta_i^2}{1 - \overline{\alpha}_i^2}\mathbf{x}
    + \frac{\left( 1 - \overline{\alpha}_{i-1}^2 \right) \alpha_i}{1 - \overline{\alpha}_i^2} \mathbf{x}_i, \qquad
\sigma^2(i) := \frac{1 - \overline{\alpha}_{i-1}^2}{1 - \overline{\alpha}_i^2} \beta_i^2.
$$

When training, since $\sigma^2(i)$ is fixed, $p_{\boldsymbol{\phi}}$ could be modeled by a learnable mean function $\boldsymbol{\mu}_{\boldsymbol{\phi}}(\cdot, i)$,
where the target objective is the average of KL divergences for all layers, averaged over the data distribution:
$$
\mathcal{L}_{\text{DDPM}} (\boldsymbol{\phi}) :=
    \sum_{i=1}^L {
        \frac{1}{2\sigma^2(i)}
        \mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}}
        \mathbb{E}_{\mathbf{x}_i \sim p(\mathbf{x}_i | \mathbf{x}_0)}
        \left[
            \lVert \boldsymbol{\mu}_{\boldsymbol{\phi}} (\mathbf{x}_i, i) - \boldsymbol{\mu}(\mathbf{x}_i, \mathbf{x}_0, i) \rVert_2^2
        \right]
    }.
$$

#### $\boldsymbol{\epsilon}$-prediction

Practically, instead of predicting the mean function, the equivalent $\boldsymbol{\epsilon}$-prediction is widely used for simplicity.

$\boldsymbol{\mu}(\mathbf{x}_i,\mathbf{x}_0, i)$ can be rewritten as a function of $\boldsymbol{\epsilon}$
from the fact $\mathbf{x}_i = \overline{\alpha}_i \mathbf{x}_0 + \sqrt{1 - \overline{\alpha}_i^2} \boldsymbol{\epsilon}$:
$$
\boldsymbol{\mu}(\mathbf{x}_i,\mathbf{x}_0, i) = \frac{1}{\alpha_i} \left( \mathbf{x}_i - \frac{1 - \alpha_i^2}{\sqrt{1 - \overline{\alpha}_i^2}} \boldsymbol{\epsilon} \right).
$$

This suggests parameterizing $\boldsymbol{\mu}_{\boldsymbol{\phi}}$ using a learnable function $\boldsymbol{\epsilon}_{\boldsymbol{\phi}}$:
$$
\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}_i, i) = \frac{1}{\alpha_i} \left( \mathbf{x}_i - \frac{1 - \alpha_i^2}{\sqrt{1 - \overline{\alpha}_i^2}} \boldsymbol{\epsilon}_{\boldsymbol{\phi}} (\mathbf{x}_i, i) \right).
$$

Further, $\mathcal{L}_{\text{DDPM}}$ simplifies to finding $\boldsymbol{\epsilon}_{\boldsymbol{\phi}} (\mathbf{x}_i, i)$ that minimizes:
$$
\mathcal{L}_{\text{simple}} (\boldsymbol{\phi}) :=
    \mathbb{E}_i \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}
    \left[
        \lVert \boldsymbol{\epsilon}_{\boldsymbol{\phi}} (\mathbf{x}_i, i) - \boldsymbol{\epsilon} \rVert_2^2
    \right].
$$
Dropping the per-term weights (present in the exact derivation) was found to be beneficial for sample quality in [Ho et al. (2020)](https://arxiv.org/pdf/2006.11239), as it up-weights the harder denoising tasks at larger $i$.

#### Sampling

After training, sampling proceeds sequentially from $\mathbf{x}_L \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ back to $\mathbf{x}_0$:
$$
\mathbf{x}_{i-1} \leftarrow
\frac{1}{\alpha_i} \left(
    \mathbf{x}_i - \frac{1 - \alpha_i^2}{\sqrt{1 - \overline{\alpha}_i^2}} \boldsymbol{\epsilon}_{\boldsymbol{\phi}} (\mathbf{x}_i, i)
\right) + \sigma(i) \boldsymbol{\epsilon}_i, \qquad \boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$
