# Noise-Conditional Score Network (NCSN)

A minimal implementation of NCSN, following [Song and Ermon (2019)](https://arxiv.org/abs/1907.05600).

NCSN's key intuition and motivation is:
_perturbing the data with random Gaussian noise of various magnitudes._

## Theory

### Preliminary: Score matching

Energy-based models (EBMs) defines a probability desntiy via an energy function $E_{\boldsymbol{\phi}}(\mathbf{x})$:

$$
p_{\boldsymbol{\phi}} (\mathbf{x}) := \frac{\exp(-E_{\boldsymbol{\phi}}(\mathbf{x}))}{Z_{\boldsymbol{\phi}}},
$$

where $Z_{\boldsymbol{\phi}}$ is the partition function making sure that $\int p_{\boldsymbol{\phi}} (\mathbf{x}) d\mathbf{x} = 1$.

The existence of partition function makes training EBMs tricky, since maximum likelihood training requires computing $\int p_{\boldsymbol{\phi}} (\mathbf{x}) d\mathbf{x}$, which is intractable in most cases.

Score matching, instead, tries to optimize the gradient of the log-density $\mathbf{s}(\mathbf{x}) := \nabla_\mathbf{x} \log p_{\boldsymbol{\phi}} (\mathbf{x})$.
This removes the necessity of computing $Z$, since for $p(x)=\tilde{p}(\mathbf{x}) / Z$ where $Z=\int \tilde{p}(\mathbf{x}) d\mathbf{x}$,

$$
\nabla_\mathbf{x} \log p (\mathbf{x}) = \nabla_\mathbf{x} \log \tilde{p}(\mathbf{x}) - \nabla_\mathbf{x} \log Z = \nabla_\mathbf{x} \log \tilde{p}(\mathbf{x}).
$$

And it is known that it doesn't sacrifice the expressiveness, as $p(\mathbf{x})$ could be recovered up to a constant:

$$
\log p(\mathbf{x}) = \log p(\mathbf{x}_0) + \int_0^1 \mathbf{s}(\mathbf{x}_0 + t(\mathbf{x} - \mathbf{x}_0)) dt,
$$

via the fundamental theorem of calculus.

#### Training with score matching

In score matching,
the true log-score function $\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p_{\text{data}} (\mathbf{x})$ is approximated using a neural network
$\mathbf{s}_{\boldsymbol{\phi}} (\mathbf{x})$.

Instead of maximum likelihood training that requires computing $Z$,
training with score matching defines the training objective by aligning the score with the data score:

$$
\mathcal{L}_{\text{SM}} (\boldsymbol{\phi}) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \left[ \lVert \mathbf{s}_{\boldsymbol{\phi}} (\mathbf{x}) - \mathbf{s}(\mathbf{x}) \rVert_2^2 \right].
$$

In the above, although the data score $\mathbf{s}(\mathbf{x}) = \nabla_\mathbf{x} \log p_{\text{data}} (\mathbf{x})$ is unknown,
but [Hyvärinen (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) used the integration by parts technique to get an equivalent objective without the term:

$$
\mathcal{L}_{\text{SM}} (\boldsymbol{\phi}) = \tilde{\mathcal{L}}_{\text{SM}} (\boldsymbol{\phi}) + C,
$$

where

$$
\tilde{\mathcal{L}}_{\text{SM}} (\boldsymbol{\phi}) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \text{Tr}(\nabla_\mathbf{x} \mathbf{s}(\mathbf{x})) + \frac{1}{2}\lVert \mathbf{s}_{\boldsymbol{\phi}} (\mathbf{x}) \rVert_2^2 \right].
$$

#### Sampling with Langevin dynamics

Sampling from the trained model $\mathbf{s}_{\boldsymbol{\phi}}$ is done by (discrete or continuous-time) Langevin dynamics:

$$
\mathbf{x}_{n+1} = \mathbf{x}_n - \eta \mathbf{s}_{\boldsymbol{\phi}} (\mathbf{x}_n) + \sqrt{2\eta} \boldsymbol{\epsilon}_{n}
$$

in the discrete case or

$$
d\mathbf{x}(t) = \mathbf{s}_{\boldsymbol{\phi}} (\mathbf{x}(t)) dt + \sqrt{2} d\mathbf{w}(t)
$$

in the continuous case where $\mathbf{w}(t)$ denotes a standard Brownian motion.

### Preliminary: Denoising score matching

While the score matching defines a tractable objective $\mathcal{L}_{\text{SM}}$,
it still requiers computing the trace of the Jacobian, which has complexity of $\mathcal{O}(D^2)$ where $D$ is the dimensionality of data.

To solve the issue, [denoising score matching (DSM)](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)
proposed injecting noise into the data via a known conditional distribution $p_\sigma (\tilde{\mathbf{x}} | \mathbf{x})$,
and made the neural network $\mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma)$ approximate the score of the perturbed distribution

$$
p_\sigma (\tilde{\mathbf{x}}) = \int p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) p_{\text{data}} (\mathbf{x}) d\mathbf{x}.
$$

Rewriting the score matching objective:

$$
\mathcal{L}_{\text{SM}} (\boldsymbol{\phi}; \sigma) := \frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[ \lVert \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma) - \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \rVert_2^2 \right].
$$

Conditioning on $\mathbf{x} \sim p_{\text{data}}$, the DSM objective is defined by:

$$
\mathcal{L}_{\text{DSM}} := \frac{1}{2} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} \sim p_\sigma(\cdot | \mathbf{x})}
\left[ \lVert \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma) - \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) \rVert_2^2  \right],
$$

which is tractable since we assume $p_\sigma (\tilde{\mathbf{x}} | \mathbf{x})$ is a known distribution whose log-gradient has a closed form equation.

An example of a such distribution is the Gaussian noise with variance $\sigma^2$:

$$
p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}} ; \mathbf{x}, \sigma^2 \mathbf{I})
$$

whose log-gradient $\nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) = (\mathbf{x} - \tilde{\mathbf{x}}) / \sigma^2$.

The SM and DSM objectives are identical up to a constant: $\mathcal{L}_{\text{SM}} (\boldsymbol{\phi}; \sigma) = \mathcal{L}_{\text{DSM}} (\boldsymbol{\phi}; \sigma) + C$.

<details>
<summary>Proof</summary>

$$
\begin{align*}
&\mathcal{L}_{\text{SM}} (\boldsymbol{\phi}; \sigma)
= \frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[ \lVert \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma) - \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \rVert_2^2 \right] \\
&= \frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[ \lVert \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma) \rVert_2^2
    + \frac{1}{2} \lVert \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \rVert_2^2
    - \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \right\rangle \right] \\
&=\frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[ \lVert \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma) \rVert_2^2 \right]
    + \frac{1}{2}  \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[ \lVert \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \rVert_2^2 \right]
    - \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[\left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \right\rangle \right]
\end{align*}
$$

The second term is independent to $\boldsymbol{\phi}$; let's denote it as $C_1$.

The third term, $\mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[\left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \right\rangle \right]$, could be expressed as:

$$
\begin{align*}
&\mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[ \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \right\rangle \right] \\
&= \int_{\tilde{\mathbf{x}}} p_\sigma(\tilde{\mathbf{x}}) \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \right\rangle d\tilde{\mathbf{x}} \\
&= \int_{\tilde{\mathbf{x}}} p_\sigma(\tilde{\mathbf{x}}) \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma), \frac{\nabla_{\tilde{\mathbf{x}}} p_\sigma (\tilde{\mathbf{x}})}{p_\sigma{}(\tilde{\mathbf{x}})} \right\rangle d\tilde{\mathbf{x}} \\
&= \int_{\tilde{\mathbf{x}}} \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma), \nabla_{\tilde{\mathbf{x}}} p_\sigma (\tilde{\mathbf{x}}) \right\rangle d\tilde{\mathbf{x}} \\
&= \int_{\tilde{\mathbf{x}}} \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma), \nabla_{\tilde{\mathbf{x}}} \int_\mathbf{x} p_{\text{data}}(\mathbf{x}) p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) d\mathbf{x} \right\rangle d\tilde{\mathbf{x}} \\
&= \int_{\tilde{\mathbf{x}}} \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma), \int_\mathbf{x} p_{\text{data}}(\mathbf{x}) \nabla_{\tilde{\mathbf{x}}} p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) d\mathbf{x} \right\rangle d\tilde{\mathbf{x}} \\
&= \int_{\tilde{\mathbf{x}}} \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma), \int_\mathbf{x} p_{\text{data}}(\mathbf{x}) p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} | \mathbf{x})  d\mathbf{x} \right\rangle d\tilde{\mathbf{x}} \\
&= \int_{\tilde{\mathbf{x}}} \int_\mathbf{x}  p_{\text{data}}(\mathbf{x}) p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} | \mathbf{x})  \right\rangle d\mathbf{x} d\tilde{\mathbf{x}} \\
&= \int_\mathbf{x} \int_{\tilde{\mathbf{x}}} p_{\text{data}}(\mathbf{x}) p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} | \mathbf{x})  \right\rangle d\tilde{\mathbf{x}} d\mathbf{x}  \\
&= \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} \sim p_\sigma(\cdot | \mathbf{x})} \left[ \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} | \mathbf{x})  \right\rangle \right].
\end{align*}
$$

Thus,

$$
\begin{align*}
&\mathcal{L}_{\text{SM}} (\boldsymbol{\phi}; \sigma) \\
&=\frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[ \lVert \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma) \rVert_2^2 \right]
    - \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} \sim p_\sigma(\cdot | \mathbf{x})} \left[ \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} | \mathbf{x})  \right\rangle \right]
    + C_1 \\
&=\frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[ \lVert \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma) \rVert_2^2 \right]
    - \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} \sim p_\sigma(\cdot | \mathbf{x})} \left[ \left\langle \mathbf{s}_{\boldsymbol{\phi}} (\tilde{\mathbf{x}}; \sigma),  \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} | \mathbf{x})  \right\rangle \right] \\
& \qquad + \frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \lVert \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}| \mathbf{x}) \rVert_2^2
    - \frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \lVert \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}} | \mathbf{x}) \rVert_2^2
    + C_1 \\
&= \mathcal{L}_{\text{DSM}} (\boldsymbol{\phi}; \sigma) + C_1 - \frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})}\lVert \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}| \mathbf{x}) \rVert_2^2.
\end{align*}
$$

Again, since the last term doesn't depend on $\boldsymbol{\phi}$, we can treat it as a constant with respect to parameters, which completes the proof,
with $C = \frac{1}{2}  \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \left[ \lVert \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}) \rVert_2^2 \right] - \frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}})} \lVert \nabla_{\tilde{\mathbf{x}}} \log p_\sigma (\tilde{\mathbf{x}}| \mathbf{x}) \rVert_2^2$.

</details>

### NCSN

NCSN could be seen as a DSM with multiple noise levels, i.e. rather than having a single $p_\sigma (\tilde{\mathbf{x}} | \mathbf{x})$,
it introduces multiple noises: $p_{\sigma_i} (\mathbf{x}_{\sigma_i} | \mathbf{x})$, where $0 < \sigma_1 < \sigma_2 < \cdots < \sigma_L$.

Sampling is done by applying Langevin sampling at each noise level, starting from $i=L$;
i.e. determining coarse features with larger $\sigma$, and then generating detailed features with smaller $\sigma$.

The training objective is basically identical to the one from DSM, but with different noise level per level:

$$
\mathcal{L}_{\text{NCSN}} (\boldsymbol{\phi}) := \lambda(\sigma_i) \mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma_i),
$$

where $\lambda(\sigma_i) > 0$ is a weighting function for each $\sigma_i$.

Sampling is done by annealed Langevin dynamics;
starting from $\mathbf{x}^{\sigma_L} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$,
apply the Langevin sampling for $N_l$ iterations at level $l \in [L, 2]$
with the trained score function $\mathbf{s}_{\boldsymbol{\phi}^\times}(\cdot, \sigma_l)$, to obtain $\mathbf{x}^{\sigma_{l-1}}$,
until reaching $\mathbf{x}^{\sigma_1}$.

From above, we can figure out that NCSN has slow sampling speed,
requiring $\mathcal{O}(LK)$ operations where $K$ is the number of iterations per level.
