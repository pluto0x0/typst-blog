#import "../index.typ": template, tufted
#show: template.with(title: "Fundamentals of Reinforcement Learning")

#import "@preview/mmdr:0.2.1": mermaid
#show raw.where(lang: "mermaid"): it => mermaid(it.text)

#import "@preview/lemmify:0.1.8": *
#let (
  theorem, lemma, corollary,
  remark, proposition, example,
  proof, rules: thm-rules
) = default-theorems("thm-group", lang: "en")
#show: thm-rules

#set heading(numbering: "1.")

= Introduction to Reinforcement Learning

== Basic Definitions

An RL problem is defined by the following components:

- *State*: $s in S$, the set of all possible configurations of the environment.
- *Action*: $a in A$, the set of all actions the agent can take.
- *Reward*: $r: S times A -> RR$, the immediate feedback signal for taking an action in a state.
- *Transition*: $P(s'|s, a)$, the probability of transitioning to state $s'$ given state $s$ and action $a$.
- *Policy*: $pi: S -> A$, a mapping from states to actions. The optimal policy is denoted $pi^*$.

Two central quantities measure the long-term quality of a policy:

- *Value function*: $V^pi (s) = EE [sum_(t = 1)^oo gamma^(t - 1) r_t|s_1 = s, pi]$ where $r_t$ is the reward at time step $t$ and $gamma in (0, 1)$ is the discount factor. It measures the expected cumulative discounted reward starting from state $s$ under policy $pi$.
- *Q function*: $Q^pi (s, a) = EE [sum_(t = 1)^oo gamma^(t - 1) r_t|s_1 = s, a_1 = a, pi]$, which additionally conditions on the first action $a$.

== Motivating Examples

=== Shortest Path

#figure(
  image("img/reinforcement-learning-lecture-1.png", width: 60%),
  caption: [Shortest Path],
)

- nodes: states
- edges: actions

A greedy algorithm is not optimal in general, because locally optimal choices may lead to globally suboptimal paths.

*Bellman Equation* (Dynamic Programming):

$ V^* (d) = min{3 + V^* (g), 2 + V^* (f)} $

=== Stochastic Shortest Path

When transitions are stochastic, the problem becomes a *Markov Decision Process (MDP)*: the next state depends probabilistically on the current state and action.

#figure(
  image("img/reinforcement-learning-lecture-1-1.png", width: 60%),
  caption: [Stochastic Shortest Path],
)

*Bellman Equation*

$ V^* (c) = min {4 + 0.7 V^* (d) + 0.3 V^* (e), thin 2 + V^* (e)} $

The optimal policy $pi^*$ is the one that minimizes the expected cost at every state.

=== Model-based RL

In model-based RL, the transition probabilities are unknown. The agent must learn them by *trial-and-error*.

#figure(
  image("img/reinforcement-learning-lecture-1-2.png", width: 60%),
  caption: [a trajectory: $s_0->c->e->F->G$],
)

The agent recovers the underlying graph structure by collecting multiple *trajectories*. It then uses empirical frequencies to estimate transition probabilities.

#tufted.margin-note[This approach assumes that states and actions are visited sufficiently uniformly.]

=== Exploration Problem

Random exploration can be inefficient, especially in environments with sparse rewards:

#figure(
  image("img/reinforcement-learning-lecture-1-5.png", width: 30%),
  caption: [example: video game],
)

=== Video Game

Objective: maximize the cumulative reward.

$ EE[sum_(t = 1)^oo r_t|pi] "or" EE[sum_(t=1)^oo gamma^(t - 1) r_t|pi] $

Problem: the state space is too large to enumerate.

#figure(
  image("img/reinforcement-learning-lecture-1-4.png", width: 60%)
)

There are states that the RL agent has never seen, and therefore the agent needs *generalization* to handle unseen states.

=== Contextual Bandits

- Even if the algorithm is good, if it makes bad actions at the beginning, it will not get good data.
- Keeping bad actions (e.g.~guessing wrong label on image classification) prevents the agent from discovering the right action.
  - This is a key difference from supervised learning, where the correct labels are always available.
- #link("https://en.wikipedia.org/wiki/Multi-armed_bandit")[Multi-armed bandit]

= The RL Protocol

For round $t = 1, 2, ...,$ the learner interacts with the environment as follows:

- For time step $h=1, 2, ..., H$, the learner
  - Observes state $x_h^((t))$
  - Chooses action $a_h^((t))$
  - Receives reward
    $r_h^((t))~R(x_h^((t)), a_h^((t)))$
  - The next state $x_(h + 1)^((t))$ is generated as a function of
    $x_h^((t))$ and $a_h^((t))$ (or sometimes, all previous states
    and actions within round $t$)

Recall $V^* , Q^* , V^pi, Q^pi$. The optimal value function satisfies:

$
  V^* (s) = max_(a in A)(underbrace(R(s, a) + gamma EE_(s'~P(dot|s, a)) [V^*  (s')], Q^* (s, a)))
$

= Bellman Equations and Dynamic Programming

== Bellman Operator

The Bellman operator $cal(T)$ maps a function $f : S times A -> RR$ to another function of the same type:

$
  forall f : S times A -> RR,\
 (cal(T) f)(s, a) =(R(s, a) + gamma EE_(s'~P(dot|s, a))[max_(a') f(s', a')])\
  "where" cal(T) : RR^(S A) -> RR^(S A) .
$

The key property is that $Q^*$ and $V^*$ are *fixed points* of the Bellman operator:

$
  Q^*  = cal(T) Q^* \
  V^*  = cal(T) V^* \
$

i.e.~$Q^* $ and $V^* $ are fixed points of $cal(T)$.

== Value Iteration Algorithm (VI)
<value-iteration-algorithm-vi>

The *greedy policy* with respect to the optimal Q-function is:

$ pi^* (s) = arg max_(a in A) Q^* (s, a) $

Value Iteration computes a sequence of functions converging to $Q^*$:

$ f_0, f_1, f_2, ... -> Q^*  $

At each step, the greedy policy with respect to $f_k$ is defined as:

$ pi_(f_k)^* (s) = arg max_(a in A) f_k (s, a) $

The following claim bounds the suboptimality of the greedy policy:

$||V^*  - V^(pi_f)||<= frac(2||f - Q^* ||_oo, 1 - gamma) $

Define the Bellman operator on value functions:

$ (cal(T) f)(s) = max_(a in A) (R(s, a) + gamma E_(s'~P(dot|s, A)) [f (s')]) $

#footnote[
  Note: the $cal(T)$ acting on $Q^*$ (mapping $RR^(S A) -> RR^(S A)$) and the $cal(T)$ acting on $V^*$ (mapping $RR^S -> RR^S$) are *not the same* operator, though they share the same notation.
]

The algorithm starts from the zero function and iterates:

$ f_0 = arrow(0) in RR^(S A) $

For $k = 1, 2, 3, ...,$ compute:

$ f_k = cal(T) f_(k - 1) $

Concretely, this means: $forall s, a : f_1(s, a) <- cal(T) f_0(s, a)$.

There are two variants of performing this iteration:

- *Synchronous iteration*: use all function values from the old version $f_(k-1)$ during the update.
- *Asynchronous iteration*: update entries one at a time, using the most recent values.

=== Convergence of Value Iteration

#lemma(name: [$gamma$-contraction])[

  $cal(T)$ is a *$gamma$-contraction* under $||dot||_oo$, where $||f||_oo := max_(s, a) |f(s, a)|$. This means:
  $||cal(T) f - cal(T) f'||_oo <= gamma||f - f'||_oo $
]

#proof(name: [$gamma$-contraction])[

  It suffices to prove
  $
    & lr(|(cal(T) f - cal(T) f')(s, a)|)\
    = & lr(|(R(s, a) + gamma EE_(s'~P(dot |s, a))[max_(a') f(s', a')]) - (R(s, a) + gamma EE_(s'~P(dot| s, a))[max_(a') f(s', a')])|)\
    = & gamma lr(|EE_(s'~P(dot|s, a))[max_(a') f(s', a') - max_(a') f'(s', a')]|)
  $

  WLOG assume, $max_(a') f(s' , a') > max_(a') f' (s' , a')$ and $exists a^* : max_a f(s', a) = f(s', a^*)$, then

  $ & gamma lr(|EE_(s'~P(dot|s, a))[max_(a') f(s', a') - max_(a') f'(s', a')]|)\
  & gamma lr(|EE_(s'~P(dot|s, a))[f(s', a^* ) - max_(a') f'(s', a')]|)\
  <= & gamma lr(|EE_(s'~P(dot|s, a))[f(s', a^* ) - f'(s', a^* )]|)\
  <= & gamma lr(|f(s', a^* ) - f'(s', a^* )|)\
  <= & gamma||f - f'||_oo $
]

Using the $gamma$-contraction lemma, we can prove the convergence of Value Iteration:

#proof(name: "Convergence of VI")[
  $
      &||f_k - Q^* ||_oo \
    = &||cal(T) f_(k - 1) - Q^* ||_oo \
    = &||cal(T) f_(k - 1) - cal(T) Q^* ||_oo \
    <=^("lemma") & gamma||f_(k - 1) - Q^* ||_oo \
  $

  Applying this recursively from $f_0$:

  $ =>||f_k - Q^* ||_oo <= gamma^k||f_0 - Q^* ||_oo "where" gamma in(0, 1) $
]

== $V^* $ Iteration

$V^*$ Iteration is the analogous algorithm applied to value functions instead of Q-functions:

$
  f_0 = arrow(0)\
  f_k <- cal(T) f_(k - 1)
$

The resulting $f_k$ equals the optimal value over $k$-horizon policies:

$ f_k (s) = max_("all possible " pi) EE [sum_(t = 1)^k gamma^(t - 1) r_t|s_1 = s, pi] $
#footnote[
  This follows from the definition of the Bellman operator $cal(T)$.
]

The convergence rate is:

$||f_k - V^* || lt.tilde gamma^k $

Step 1: $f_k <= V^* $ (the $k$-horizon optimum cannot exceed the infinite-horizon optimum).

Step 2:

$
  f_k >= & EE [sum_(t = 1)^oo gamma^(t - 1) r_t|s_1 = s, pi^* ] - EE [sum_(t = k + 1)^oo gamma^(t - 1) r_t|s_1 = s, pi^* ]\
  >= & V^*  - gamma^k V_max qed
$

#footnote[
  This argument assumes that once the optimal policy $pi^*$ reaches the goal, it never leaves. The tail reward beyond step $k$ is bounded by $gamma^k V_max$.
]

== Example: Restaurant Choice

```mermaid
graph TD;
    A([start])
    A -->|+0| B([Japanese])
    A -->|+0| C([Italian])
    B -->|+2| D([Ramen])
    B -->|+2| E([Sushi])
    C -->|+1| F([Steak])
    C -->|+3| G([Pasta])
```

The optimal policy is to head toward `Pasta`, which yields the maximum cumulative reward of $3$.

#footnote[
  This example is a finite horizon case. To make it infinite horizon
  discount, add a terminal state $T$ :

  ```mermaid
  graph TD;
      A([start])
      A -->|+0| B([Japanese])
      A -->|+0| C([Italian])
      B -->|+2| D([Ramen])
      B -->|+2| E([Sushi])
      C -->|+1| F([Steak])
      C -->|+3| G([Pasta])

      D -.->|+0| T((T))
      E -.->|+0| T((T))
      F -.->|+0| T((T))
      G -.->|+0| T((T))
      T --->|+0| T((T))
  ```
]

To find $V^* (s)$, update $V$ values from the leaf nodes upward to the root state.

= Policy Iteration

== Policy Iteration by Example

=== Iteration \#0

Define initial policy $pi_0$ :

```mermaid
graph TD;
    A([start])
    A -.-|+0| B([Japanese])
    A ==>|+0| C([Italian])
    B ==>|+2| D([Ramen])
    B -.-|+2| E([Sushi])
    C ==>|+1| F([Steak])
    C -.-|+3| G([Pasta])
```

Then compute the corresponding $Q^(pi_0)(s, a)$ :

```mermaid
graph TD;
    A([start])
    A -->|+2| B([Japanese])
    A -->|+1| C([Italian])
    B -->|+2| D([Ramen])
    B -->|+2| E([Sushi])
    C -->|+1| F([Steak])
    C -->|+3| G([Pasta])
```


=== Iteration \#1

Update the policy greedily: $ pi_1 (s) : = arg max_(a in A) Q^(pi_0) (s, a) $

```mermaid
graph TD;
    A([start])
    A ==>|+2| B([Japanese])
    A -.-|+1| C([Italian])
    B ==>|+2| D([Ramen])
    B -.-|+2| E([Sushi])
    C -.-|+1| F([Steak])
    C ==>|+3| G([Pasta])
```

$Q^(pi_1)$:

```mermaid
graph TD;
    A([start])
    A -->|+2| B([Japanese])
    A -->|+3| C([Italian])
    B -->|+2| D([Ramen])
    B -->|+2| E([Sushi])
    C -->|+1| F([Steak])
    C -->|+3| G([Pasta])
```

=== Iteration \#2

$ pi_2 (s) : = arg max_(a in A) Q^(pi_1) (s, a) $

```mermaid
graph TD;
    A([start])
    A -.-|+2| B([Japanese])
    A ==>|+3| C([Italian])
    B ==>|+2| D([Ramen])
    B -.-|+2| E([Sushi])
    C -.-|+1| F([Steak])
    C ==>|+3| G([Pasta])
```

=== Discussion

The policy was switched to Japanese in iteration \#1, then switched back to Italian in iteration \#2. This illustrates that the policy updates propagate information *upward* from the leaves: the leaf-level choices are corrected first, and the root-level decision adjusts once the downstream values are accurate.

== Monotone Policy Improvement

Policy iteration is guaranteed to improve monotonically:

$ forall k, forall s : V^(pi_k)(s) >= V^(pi_(k - 1))(s) $

$ "if" pi_(k - 1) eq.not pi^* , exists s : V^(pi_k)(s) > V^(pi_(k - 1))(s) $

Since the number of distinct deterministic policies is finite, the total number of iterations is bounded:

$ "#iterations" <= |A|^(|S|) $

#tufted.margin-note[Monotone policy improvement produces *exact* solutions, while value iteration produces approximate solutions that converge in the limit.]

=== Proof of Monotone Improvement

We want to show $Q^(pi_(k + 1)) >= Q^(pi_k)$. The proof relies on three lemmas.

*Lemma 1*:

$ Q^(pi_k) = cal(T)^(pi_k) Q^(pi_k) <= cal(T) Q^(pi_k) $

because

$
  &(cal(T)^pi f)(s, a) = R(s, a) + gamma EE_(s'~P(dot|s, a)) [f (s', pi)]\
  <= &(cal(T) f)(s, a) = R(s, a) + gamma EE_(s'~P(dot|s, a)) [max_(a') f (s', a')]
$

*Lemma 2*:

$ cal(T) Q^(pi_k) = cal(T)^(pi_(k + 1)) Q^(pi_k) $

*Lemma 3*:

$ forall f >= f', cal(T)^pi f >= cal(T)^pi f' $

Combining Lemmas 1, 2, and 3:

$
  Q^(pi_k) & <= cal(T)^(pi_(k + 1)) Q^(pi_k)\
  => cal(T)^(pi_(k + 1)) Q^(pi_k) & <= cal(T)^(pi_(k + 1)) cal(T)^(pi_(k + 1)) Q^(pi_k)\
  & dots.v\
  => Q^(pi_k) & <= (cal(T)^(pi_(k + 1)))^oo Q^(pi_k) = Q^(pi_(k + 1))
$

The last equality holds because $Q^(pi_(k + 1))$ is the fixed point of $cal(T)^(pi_(k + 1))$.

=== Summary

In policy iteration, we apply the greedy improvement at every step. The number of iterations is finite because the policy improves strictly at each step and the total number of deterministic policies is bounded.

== Performance-Difference Lemma

#tufted.margin-note[The Performance-Difference Lemma is a fundamental tool in RL theory. Many deep RL algorithms rely on this lemma.]

$forall pi, pi', s$,

$ V^(pi')(s) - V^pi(s) = frac(1, 1 - gamma) E_(s'~d_s^(pi')) [Q^pi(s', pi') - V^pi(s')] $

Applying the lemma to the policy iteration steps:

$
  V^(pi_(k + 1))(s) - V^(pi_k)(s) = frac(1, 1 - gamma) E_(s'~d_s^(pi_(k+1))) [Q^(pi_k)(s', pi_(k + 1)) - V^(pi_k)(s')]
$

Since $V^(pi_k)(s') = Q^(pi_k)(s', pi_k)$ and $pi_(k+1)$ is the greedy policy with respect to $Q^(pi_k)$, the term $Q^(pi_k)(s', pi_(k+1)) - V^(pi_k)(s') >= 0$ for all $s'$, so the RHS is non-negative. This gives another proof of monotone policy improvement.

=== Proof of the Lemma

_(Proof omitted.)_

= Planning and Learning

*Planning*:

- Given the MDP model, compute the optimal policy.
- The MDP model $(S, A, P, R, gamma)$ is fully known.

*Learning*:

- The MDP model is unknown.
- The agent must collect data from the MDP in the form of tuples $(s, a, r, s')$.
- Data is limited, e.g., in adaptive medical treatment and dialog systems.
- Examples: Go, chess, etc.
- Learning can be useful *even if the final goal is planning*:
  - Especially when $|S|$ is large and/or only a blackbox simulator is available.
  - E.g., AlphaGo, video game playing, simulated robotics.

= Monte-Carlo Policy Evaluation

Given a policy $pi$, we want to estimate its performance:
$J(pi) := EE_(s~d_0) [V^pi(s)]$ , where $d_0$ is the initial
state distribution.

Monte-Carlo outputs some scalar $v$; accuracy is measured by
$|v - J(pi)|$. The estimate is obtained by sampling different trajectories:

Data: trajectories starting from $s_1~d_0$ using $pi$ (i.e.,
$a_t = pi (s_t)$ ).

$
  {(s_1^((i)), a_1^((i)), r_1^((i)), s_2^((i)), ..., s_H^((i)), a_H^((i)), r_H^((i)))}_(i = 1)^n
$

#tufted.margin-note[This is called *on-policy* evaluation: the data is collected using the exact same policy being evaluated. Otherwise, it is *off-policy*.]

Estimator:

$ 1 / n sum_(i = 1)^n sum_(t = 1)^H gamma^(t - 1) r_t^((i)) $

#footnote[
  Guarantee: with probability at least
  $1 - delta, |v - J(pi)| <= frac(R_max, 1 - gamma) sqrt(frac(1, 2 n) ln 2 / delta)$
  (larger $n$ means higher accuracy).
  Notably, this bound is *independent* of the size of the state space.
]

== Comment on Monte-Carlo

Monte-Carlo is a zeroth-order (ZO) optimization method, which is not
efficient compared to higher-order methods:

- *First order*: gradient / first derivative (in DL/ML,
  *SGD*)
- *Second order*: Hessian matrix / second derivative

= Model-based RL with a Sampling Oracle (Certainty Equivalence)

#tufted.margin-note[The term "Certainty Equivalence" refers to treating the estimated model as if it were the true model.]

Assume we can sample $r~R(s, a)$ and
$s'~P(s, a)$ for any $(s, a)$.

Collect $n$ samples per
$(s, a)$: $(r_i , s'_i)_(i = 1)^n$ .
Total sample size: $n |S times A|$.

Estimate an empirical MDP $hat(M)$ from data:

- $hat(R)(s, a) := 1 / n sum_(i = 1)^n r_i, quad hat(P) (s'|s, a) := 1 / n sum_(i = 1)^n II [s'_i = s']$
- i.e., treat the empirical frequencies of states appearing in
  ${ s'_i }_(i = 1)^n$ as the true distribution.

Plan in the estimated model and return the optimal policy.

Transition tuples: $(s_i, a_i, r_i, s_(i + 1))$. Use
$s_i, a_i$ to identify the current state and action, use $r_i$ for reward,
and $s_(i + 1)$ for the next-state transition. These tuples can be extracted from trajectories.

== Finding Policy on Estimated Environment

*True* environment: $M =(S, A, P, R, gamma)$

*Estimated* environment:
$hat(M) =(S, A, hat(P), hat(R), gamma)$

- Notation: $pi_(hat(M)), thin V_(hat(M)), thin ...$

Performance measurement:

- In the *true* environment, use
  $||V^*  - V^(pi_f)||$ where $f approx Q^* $.

- In the *estimated* environment, use $||V_M^*  - V_M^(pi_(hat(M))^* )||$, i.e.~measure the performance of the optimal policy of the estimated environment when deployed in the real environment.

== Empirical Bellman Update
<model-based-rl-with-a-sampling-oracle-certainty-equivalence-contd>

To find $Q_(hat(M))^* $ with empirical $hat(R)$ and $hat(P)$, we run value iteration in the estimated model:

$ f_0 in RR^(S A), quad f_k = hat(cal(T)) f_(k - 1) . $

where

$
  (hat(cal(T)) f)(s, a) &= hat(R) (s, a) + gamma EE_(s^prime ~ hat(P) (dot.op|s, a)) underbrace([max_(a^prime) f(s^prime comma a^prime)], V_f (s^prime)) \
&= 1/n sum_(i = 1)^n r_i + gamma lr(chevron.l hat(P) (dot.op|s, a), V_f chevron.r) \
&= 1/n sum_(i = 1)^n r_i + gamma sum_(s^prime) (1/n sum_(i = 1)^n II [s_i^prime = s^prime]) dot.op V_f (s^prime) \
&= 1/n sum_(i = 1)^n r_i + gamma/n sum_(i = 1)^n (sum_(s^prime) II [s_i^prime = s^prime] V_f (s^prime)) \
&= 1/n sum_(i = 1)^n r_i + gamma dot.op 1/n sum_(i = 1)^n V_f (s_i^prime) \
&= 1/n sum_(i = 1)^n (r_i + gamma max_(a^prime) f(s_i^prime , a^prime)) \
$

This is called the *Empirical Bellman Update*.

== Computational Complexity

=== Value Iteration

For the original
#link(<value-iteration-algorithm-vi>)[Value Iteration],
the computational complexity per iteration is

$ |S| times |A| times |S| $

$|S| times |A|$ entries to update, each requiring an expectation over $|S|$ next states.

=== Empirical Bellman Update

For the Empirical Bellman Update, the computational complexity per iteration is

$ |S| times |A| times n $

where $n$ is the number of empirical samples per state-action pair.

= Value Prediction

== The Value Prediction Problem

Given a fixed policy $pi$, we want to estimate $V^pi$ and $Q^pi$.

/ On-policy Learning: #block[
    Data used to evaluate or improve policy $pi$ is generated by $pi$ itself.
  ]

/ Off-policy Learning: #block[
    Data used to evaluate or improve policy $pi$ is generated by some other policy.
  ]

When the action is always chosen by a fixed policy, the MDP reduces to a
Markov chain plus a reward function over states, also known as a *Markov
Reward Process (MRP)*.

== Monte-Carlo Value Prediction
<monte-carlo-value-prediction>

For each state $s$, roll out $n$ trajectories using policy $pi$:

- For episodic tasks, roll out until termination.
- For continuing tasks, roll out to a length (typically
  $H = O(1 \/(1 - gamma))$ ) such that omitting the
  future rewards has minimal impact ("small truncation error").
- Let $hat(V)^pi(s)$ (written as $V(s)$ below) be the average
  discounted return.

*Online Monte-Carlo*

- For $i = 1, 2, ...$ as the index of trajectories:
  - Draw a starting state $s_i$ from the exploratory initial
    distribution, roll out a trajectory using $pi$ from $s_i$ , and let
    $G_i$ be the (random) discounted return.
  - Let $n (s_i)$ be the number of times $s_i$ has appeared as an
    initial state. If $n (s_i) = 1$ (first time seeing this state), let
    $V (s_i) <- G_i$ (where
    $G_t = sum_(t' = t)^(t + H) gamma^(t' - t) r_(t')$ ).
  - Otherwise,
    $V (s_i) <- frac(n (s_i) - 1, n (s_i)) V (s_i) + frac(1, n (s_i)) G_i$

No need to store the full trajectory.

More generally, the update rule can be written as:

$ V(s_i) <-(1 - alpha) V(s_i) + alpha G_i $

or equivalently,

$ V (s_i) <- V (s_i) + alpha (G_i - V (s_i)) $

where $alpha$ is the *learning rate* and $G_i$ is the *target*.

#footnote[
  This can be interpreted as stochastic gradient descent. If we have i.i.d.
  real random variables $v_1, v_2, ..., v_n$ , the average is the
  solution of the least-square optimization problem:

  $ min_v frac(1, 2 n) sum_(i = 1)^n (v - v_i)^2 $
]

== Every-visit Monte-Carlo

Suppose we have a continuing task and cannot set the starting
state arbitrarily.

In this case, we have a single *long* trajectory with length $N$:

$ s_1, a_1, r_1, s_2, a_2, r_2, s_3, a_3, r_3, ... $

- We can extract $N \/ H$ truncated segments, each of length
  $H = O(1 \/(1 - gamma))$.
- Alternatively, we can shift the $H$-length window by 1 each time and get
  $N - H + 1 approx N$ overlapping segments.

This "walk" through the state space should have non-zero probability on
each state, i.e.~do not starve any state of visits.

What if a state occurs multiple times on a trajectory?

- Approach 1 (*first-visit*): only the first occurrence is used.
- Approach 2 (*every-visit*): all occurrences are used.

= Temporal Difference Learning

== TD(0)

Again, suppose we have a single long trajectory
$s_1, a_1, r_1, s_2, a_2, r_2$ ,
$s_3, a_3, r_3, s_4, ...$ in a continuing task.

TD(0) performs the following update at each time step:

For
$t = 1, 2, ..., quad V (s_t) <- V (s_t) + alpha (r_t + gamma V (s_(t + 1)) - V (s_t))$

/ TD: #block[
    Temporal Difference.
  ]

/ TD error: #block[
    $r_t + gamma V (s_(t + 1)) - V (s_t)$
  ]

The update rule is the same as Monte-Carlo, except that the "target" is
$r_t + gamma V (s_(t + 1))$ instead of $G_t$. This target is similar to the
#link(<model-based-rl-with-a-sampling-oracle-certainty-equivalence-contd>)[Empirical Bellman Update].

Recall that in
#link(<monte-carlo-value-prediction>)[Monte-Carlo],
the "target" is $G_t = sum_(t' = t)^(t + H) gamma^(t' - t) r_(t')$ and
is *independent* of the current value function. In TD(0),
the target $r_t + gamma V (s_(t + 1))$ *depends* on the current value
function $V$.

Compared to value iteration:

$ V_(k + 1)(s) := EE_(r, s'|s, pi) [r + gamma V_k (s')] $

The empirical approximation is:

$ approx 1 / n sum_(i = 1)^n (r_i + gamma V_k (s'_i)) $

This is an approximate Value Iteration process. Note that the
whole iteration through $i = 1, ..., n$ constitutes only one iteration (one
$V_k$), so an outer loop is needed for $V$ to approximate
the true $V^pi$.

=== Understanding TD(0)

The approximate Value Iteration process above is similar to TD(0) but
slightly different: it uses a value function $V$ (which stays constant
during updates) to compute $V'$, which is a new function. After long
enough, we have $V' = cal(T)^pi V$ and do $V <- V'$ , then repeat
the process. This eventually converges to $V^pi$.

But in TD(0), we use $V$ to update itself. The difference is
"synchronous" (Value Iteration) vs "asynchronous" (TD(0)).

#tufted.margin-note[TD(0) is less stable than the synchronous version because updates immediately affect subsequent computations within the same sweep.]

== TD( $lambda$ ): Unifying TD(0) and MC

TD($lambda$) interpolates between TD(0) and Monte-Carlo by using multi-step bootstrap targets:

- 1-step bootstrap (TD(0)): $r_1 + gamma V(s_(i + 1))$
- 2-step bootstrap: $r_1 + gamma r_(i + 1) + gamma^2 V(s_(i + 2))$
- 3-step bootstrap:
  $r_1 + gamma r_(i + 1) + gamma^2 r_(i + 2) + gamma^3 V(s_(i + 3))$
- …
- $oo$ -step bootstrap:
  $r_1 + gamma r_(i + 1) + gamma^2 r_(i + 2) + gamma^3 r_(i + 3) + ...$
  recovers Monte-Carlo.

=== Proof of Correctness

E.g. for the 2-step bootstrap, using the #link("https://en.wikipedia.org/wiki/Law_of_total_expectation")[Law of Total Expectation]:

$
  & EE[r_1 + gamma r_(t + 1) + gamma^2 V(s_(t + 2))|s_t]\
  = & EE[r_t + gamma(r_(t + 1) + gamma V(s_(t + 2)))|s_t]\
  = & EE[r_t] + gamma EE_(s_(t + 1) |s_t) #scale(x: 120%, y: 120%)[\[] EE[(r_(t + 1) + gamma V(s_(t + 2)))| s_t, s_(t + 1)] #scale(x: 120%, y: 120%)[\]]\
  = & EE[r_t + gamma(cal(T)^pi V)(s_(t + 1))|s_t]\
  = &((cal(T)^pi)^2 V)(s)
$

=== TD( $lambda$ )

For the $n$-step bootstrap, give a $(1 - lambda) lambda^n$ weight:

- $lambda = 0$ : Only $n=1$ receives the full weight. This is TD(0).
- $lambda -> 1$ : Approaches Monte-Carlo.

=== Forward View and Backward View

*Forward view*: looking forward from each time step, combine future returns with exponentially decaying weights.

$
 (1 - lambda) dot (r_1 + gamma V(s_2) - V(s_1))\
 (1 - lambda) lambda dot (r_1 + gamma r_2 + gamma^2 V(s_3) - V(s_1))\
 (1 - lambda) lambda^2 dot (r_1 + gamma r_2 + gamma^2 r_3 + gamma^3 V(s_4) - V(s_1))\
  ...
$

*Backward view*: looking backward from each reward, distribute credit to past states with exponentially decaying eligibility.

$ 1 dot (r_1 + gamma V(s_2) - V(s_1))\
lambda gamma dot (r_2 + gamma V(s_3) - V(s_2))\
lambda^2 gamma^2 dot (r_3 + gamma V(s_4) - V(s_3))\
... $

= Value Prediction with Function Approximation
<value-prediction-with-function-approximation>

*Tabular representation vs. function approximation*: Function approximation can handle infinite or very large state spaces where we cannot enumerate all states.

*Linear function approximation*: Design features $phi.alt(s) in RR^d$ ("featurizing states") and approximate $V^pi(s) approx theta^top phi.alt(s) + b$. The parameter $theta$ is shared across all states.

#tufted.margin-note[In the following, the bias $b$ is ignored because it can be absorbed by appending a constant $1$ to the feature vector.]

A tabular value function can be interpreted as a special case: the feature vector is $phi.alt(s) in RR^(|S|)$, an indicator vector $[0, ..., 0, 1, 0, ..., 0]$ where the position of the $1$ indicates the state.

== Example: Tetris Game

#figure(
  image("img/reinforcement-learning-lecture-14-1.png", width: 30%),
  caption: [Tetris Game]
)

The state space is exponentially large: each cell can be occupied or not.

Featurization: count the number of blocks on each column. In the example, the feature is $(4,4,5,4,3,3,3)$.

== Monte-Carlo Value Prediction with Function Approximation

$ V^pi (s) = EE [G|s] = arg min_(f : S -> RR) EE [(f(s) - G)^2] $

This is a regression problem: find the function $f$ that best predicts the return $G$ from state $s$.

#footnote[
  Why is the conditional expectation the argmin? See @lemma:bias-variance.
]

The same idea applies to non-linear value function approximation. More
generally, think of function approximation as searching
over a restricted *function space* $cal(F)$, whose members
are functions that map states to real values.

E.g. the function space of linear value function approximation:

$ cal(F) = {V_theta : theta in RR^(d)} ", where " V_theta (s) = theta^top phi.alt(s) $

- This is typically only a small subset of all possible functions.
- Using "all possible functions" is equivalent to tabular representation.
- Equivalently, tabular MC value prediction can be recovered by choosing
  $phi.alt$ as the identity features $phi.alt(s) = {II [s = s']}_(s' in S)$.

We find the best approximation by minimizing the empirical loss:

$ min_(V_theta in cal(F)) 1 / n sum_(i = 1)^n (V_theta (s_i) - G_i)^2 $

SGD: uniformly sample $i$ and update:

$ theta <- theta - alpha dot (V_theta (s_(i)) - G_(i)) dot nabla V_theta (s_(i)) $

== Interpreting TD(0) with Tabular Representation

In the tabular setting, the TD(0) update is equivalent to:

$ theta <- theta + alpha (G_t - phi.alt (s_t)^top theta) phi.alt (s_t) . $

Here $theta$ is the tabular value function and $phi.alt$ is
$\[ 0, ..., 0, 1, 0, ..., 0]$ , as mentioned
#link(<value-prediction-with-function-approximation>)[above].

== TD(0) with Linear Approximation

In TD(0), we perform:

$ V (s_t) <- V (s_t) + alpha (r_t + gamma V (s_(t + 1)) - V (s_t)), $

which, across all time steps $t$, corresponds to:

$ V_(k + 1) <- cal(T)^pi V_k . $

i.e.

$ V_(k + 1)(s) = EE_pi [r + gamma V_k (s')|s] $

Similar to MC with function approximation, we can *rewrite the expectation as a regression problem*:

$
  V_(k + 1) (s) =
  arg min_(f : S -> RR) EE_pi [(f(s) -(r + gamma V_k (s')))^2] \
  approx arg min_(V_theta in cal(F)) 1/n sum_(i = 1)^n (V_theta (s_i) - r_i - gamma V_k (s'_i))^2 .
$

And the SGD update is:

$ theta <- theta + alpha (r_t + gamma V_k (s_(t + 1)) - V_theta (s_t)) nabla V_theta (s_t) $


Recall the Bellman Equation:

$
  (cal(T)^pi f)(s, a) & = R(s, a) + gamma EE_(s'~P(dot|s, a)) [f (s', pi)] \
                 & = EE [r + gamma dot f (s', pi)|s, a] .
$

which empirically equals:

$ 1 / n sum_(i = 1)^n (r_i + gamma Q_(k-1) (s'_i, pi)) . $

With tuples $(s_t, a_t, r_t, s_(t + 1))$ from the long
trajectory, applying the running average:

$ Q_k (s_t, a_t) <- Q_k (s_t, a_t) + alpha(r_t + gamma Q_(k - 1)(s_(t + 1), pi) - Q_k (s_t, a_t)) $

= Control Methods

== SARSA

SARSA updates Q-values using the action actually taken in the next state:

$ Q (s_t, a_t) <- Q (s_t, a_t) + alpha (r_t + gamma Q (s_(t + 1), a_(t + 1)) - Q (s_t, a_t)) $

Notice that SARSA is not applicable for deterministic policies, because it
requires a non-zero probability distribution over *all*
state-action pairs ( $forall(s, a) in S times A$ ), but a deterministic policy assigns probability 1 to a single action per state.

=== SARSA with $epsilon.alt$-greedy Policy

How are the state-action data pairs chosen in SARSA?

At each time step $t$, with probability $epsilon.alt$, choose $a$ from the
action space uniformly at random. Otherwise, $a_t = arg max_a Q(s_t , a)$.

#footnote[
  When sampling an $(s, a, r, s', a')$ tuple along the trajectory, the first action in
  the tuple was actually generated with the previous version of $Q$, so SARSA is not 100% "on-policy".
]

=== Does SARSA Converge to the Optimal Policy?

The cliff example (pg 132 of Sutton & Barto):

- Deterministic navigation with high penalty when falling off the cliff.
- The optimal policy walks near the cliff edge.
- Unless $epsilon.alt$ is extremely small, SARSA will learn to avoid the cliff.

#figure(
  image("img/reinforcement-learning-lecture-15.png", width: 60%),
  caption: [The cliff example],
)

The optimal path runs along the side of the cliff, but on this path the
$epsilon.alt$-greedy exploration will frequently cause the agent to fall off (incurring a large penalty). Therefore, SARSA converges to the safer path instead.

=== Softmax
<softmax>

The $epsilon.alt$-greedy exploration can be replaced by a *softmax* policy: choose action $a$ with
probability

$ frac(exp(Q(s, a) \/ T), sum_(a') exp(Q(s, a') \/ T)) $

where $T$ is the temperature parameter.

== Q-learning

Q-learning uses the *maximum* Q-value at the next state, regardless of the action actually taken:

$ Q (s_t, a_t) <- Q (s_t, a_t) + alpha (r_t + gamma max_(a') Q (s_(t + 1), a') - Q (s_t, a_t)) $

Q-learning is *off-policy*: the update always
takes $max_(a') Q (s_(t + 1), a')$ regardless of what the behavior policy actually chooses. This means Q-learning converges to the optimal policy regardless of the exploration strategy.

E.g.~in the cliff setting, the optimal policy can always be found, no matter
the choice of $epsilon.alt$.

=== Exercise: Multi-step Q-learning?

Does the target
$r_t + gamma r_(t + 1) + gamma^2 max_(a') Q (s_(t + 2), a')$ work? If
not, why?

No.~Because it leads to

$ Q <- cal(T)^pi cal(T) Q $

#footnote[
  The resulting $cal(T)^pi cal(T) ... cal(T)^pi cal(T) Q$ does converge to a fixed point, but it is the optimal policy for a *different* MDP: one where on odd steps the agent follows $pi$, and on even steps the agent is free to choose any action.
]

== Q-learning with Experience Replay

So far most algorithms we have seen are "one-pass":

- Each data point is used once and then discarded.
- Number of updates $=$ number of data points.
- Concern 1: We need many updates for optimization to converge. Can we separate optimization from data collection?
- Concern 2: We need to reuse data if sample size is limited.

*Experience replay* addresses both concerns: store all transition tuples $(s, a, r, s')$ in a replay buffer, sample (with replacement) a tuple randomly from the buffer, and apply the Q-learning update rule.

- Number of updates $>>$ number of data points.
- Each time a new tuple is collected, add it to the buffer and perform several updates.

#tufted.margin-note[Experience replay is not applicable for on-policy methods (e.g.~SARSA), because the data distribution must match the current policy.]

= Mean Squared Bellman Error

Consider minimizing the squared error between $V_theta (s)$ and its bootstrap target:

$ EE_(s, r, s') [(V_theta (s) - r - gamma V_theta (s'))^2] $

In TD(0), we perform
$V_theta (s) <- V_theta (s) + alpha(r + gamma V_theta (s') - V_theta (s))$.

What if we instead directly minimize the mean squared Bellman error (MSBE)?

By @lemma:bias-variance, the MSBE can be decomposed as the sum of two parts:

- $EE_s [(V_theta (s) - (cal(T)^pi V_theta)(s))^2]$
  — this is the $L^2$-norm Bellman error, which is desirable to minimize.
- $gamma^2 EE_s ["Var"_(s'|s, pi(s)) [V_theta (s')]]$
  — this term penalizes value functions with large variance, which is undesirable.
  - This is acceptable for deterministic environments where the variance is always $0$.

#lemma(name: "Bias-Variance Decomposition")[
  Let $X, Y$ be two random variables that follow some joint distribution over $cal(X) times RR$.

  Let $f : cal(X) -> RR$ be a real-valued function. Then:
  $ EE [(Y - f(X))^2] - EE [(f(X) - EE[Y|X])^2] = EE [(Y - EE[Y|X])^2]. $
] <lemma:bias-variance>


#proof(name: "Bias-Variance Decomposition")[
  It suffices to prove
  $
    EE [(Y - f(X))^2 -(f(X) - EE [Y|X])^2 -(Y - EE [Y|X])^2] = 0 \
    "i.e." quad EE [(E [Y | X] - Y)(E [Y | X] - f(X))] = 0 .
  $
  Given $E[Y|X]$ is a function of $X$ , let $g(X) := E[Y|X] - f(X)$,
  then it suffices to prove
  $ EE[EE[Y|X] g(X)] = EE[Y g(X)]. $

  Then:
  $
    "LHS" & = sum_(x_i) EE[Y|X = x_i] g(x_i) P_X (x_i)\
    & = sum_(x_i) g(x_i) P_X (x_i) sum_(y_i) y_i frac(P_(X, Y)(x_i, y_i), P_X (x_i))\
    & = sum_(x_i) sum_(y_i) g(x_i) y_i P_(X, Y)(x_i, y_i)\
    & = "RHS"
  $
]

#footnote[
  Let $f : cal(X) -> RR$ be an estimator from $X$ to $Y$. This
  decomposition shows that the squared error ($ell_2$ loss)
  $EE [(Y - f(X))^2]$ is at least
  $EE [(Y - EE[Y|X])^2]$ for all $f$,
  and thus cannot be made arbitrarily small.
]
<bias-variance-note>

== Solution: Double Sampling

If we have a simulator, for each $s$ in the data, draw another independent
next-state transition.

Minimize objective:

$ EE \[(V_theta (s) - r - gamma V_theta (s'_A)) lr((V_theta (s) - r - gamma V_theta (s'_B)]) $

This is called "double sampling" and is the basis of Baird's residual algorithm (Bellman residual minimization).

= Convergence of TD with Function Approximation

TD with function approximation can diverge in general. The cause is perhaps surprising:

- Is it because of randomness in SGD?
  - No. Even the batch (full-gradient) version does not converge.
- Is it because of sophisticated, non-linear function approximation?
  - No. Even linear function approximation does not converge.
- Is it because the function class does not capture $V^pi$?
  - No. Even if $V^pi$ can be exactly represented in the function class
    ("realizable"), it still does not converge.

== Example: Divergence of TD with Function Approximation

Consider an MDP: $1 -> 2 -> ... -> 9 -> 10$ with reward $~"Ber"(0.5)$.

Value iteration converges to the true values:

#figure(
  table(
    columns: 6,
    align: (auto, auto, auto, auto, auto, auto),
    table.header([\#Iter], [1], [2], […], [9], [10]),
    table.hline(),
    [1], [], [], [], [], [0.501],
    [2], [], [], [], [0.501], [0.501],
    […], [], [], [], [], [],
    [10], [0.501], [0.501], [0.501], [0.501], [0.501],
  ),
  kind: table,
)

Now suppose the function space has two possible value vectors at each state:

#figure(
  table(
    columns: 7,
    align: (auto, auto, auto, auto, auto, auto, auto),
    table.header([1], [2], [3], […], [8], [9], [10]),
    table.hline(),
    [0.5], [0.5], [0.5], […], [0.5], [0.5], [0.5],
    [1.012], [0.756], [0.628], […], [0.504], [0.502], [0.501],
  ),
  kind: table,
)

(0.5 and *0.502* have the same distance to *0.501*; 0.5 and *0.504* have the same distance to *0.502*; ...)

Then the projected Bellman iteration yields:

#figure(
  table(
    columns: 6,
    align: (auto, auto, auto, auto, auto, auto),
    table.header([\#Iter], [1], [2], […], [9], [10]),
    table.hline(),
    [1], [], [], [], [], [0.501],
    [2], [], [], [], [0.502], [0.501],
    […], [], [], [], [], [],
    [10], [1.012], [0.756], […], [0.502], [0.501],
  ),
  kind: table,
)

The value deviates from 0.501 as the iteration progresses. Intuitively, the function space is a *plane* (or low-dimensional subspace). The result of each Bellman update generally does not lie on this plane, so we must project back. These projections can amplify errors across iterations.

= Off-Policy Evaluation with Importance Sampling
<importance-sampling>

== Importance Sampling

Suppose we can only sample $x~q$ but want to estimate
$EE_(x~p) f(x)$.

The *Importance Sampling* (IS) estimator (also called the inverse propensity score estimator) is:

$ frac(p(x), q(x)) f(x) $

Unbiasedness:

$ EE_(x~q) [frac(p(x), q(x)) f(x)] = sum_x q(x) (frac(p(x), q(x)) f(x)) = sum_x p(x) f(x) = EE_(x~p)[f(x)] $

== Application in Contextual Bandits (CB)

- The data point is a tuple $(x, a, r)$.
- The function of interest is $(x, a, r) mapsto r$.
- The target distribution is
  $x~d_0, a~pi, r~R(x, a)$.
  - Let the joint density be $p(x, a, r)$.
- The data distribution is
  $x~d_0, a~pi_b, r~R(x, a)$.
  - Let the joint density be $q(x, a, r)$.
- IS estimator:
  $frac(p(x, a, r), q(x, a, r)) dot r$.
- Write down the densities:
  - $p(x, a, r) = d_0(x) dot pi(a|x) dot R(r|x, a)$
  - $q(x, a, r) = d_0(x) dot pi_b (a|x) dot R(r|x, a)$
  - To compute the importance weight, you don't need knowledge of $d_0$ or
    $R$! You just need $pi_b$ (or even just $pi_b (a|x)$ ,
    the "proposal probability").
- Let $rho$ be a shorthand for $pi(a|x) \/ pi_b(a|x)$ , so the estimator is
  $rho dot r$.
- $pi_b$ needs to "cover" $pi$:
  - i.e., whenever $pi(a|x) > 0$, we need $pi_b (a|x) > 0$.
- A special case:
  - $pi$ is deterministic, and $pi_b$ is uniformly random
    $(pi_b (a|x) equiv 1 \/ |A|)$.
  - $frac(II[a = pi(x)], 1 \/ |A|) r$
    - Only look at actions that match what $pi$ wants to take and
      discard other data points.
    - If match, $rho = |A|$; mismatch: $rho = 0$.
  - On average: only $1 \/ |A|$ portion of the data is useful.

=== A Note about Using IS

- Shifting rewards does not matter for planning purposes in fixed-horizon problems.
- However, when applying IS, shifting rewards *does* impact the variance of
  the estimator.
- Special case:
  - deterministic $pi$ , uniformly random $pi_b$ ,
  - reward is deterministic and constant: regardless of $(x, a)$ ,
    reward is always 1.
  - We know the value of any policy is 1.
  - On-policy MC has 0 variance.
  - IS still has high variance!
- Where does the variance come from?

$
  & 1 / n sum_(i = 1)^n frac(II [a^((i)) = pi (x^((i)))], 1 \/ |A|) dot r^((i)) = sum_(i = 1)^n frac(II [a^((i)) = pi (x^((i)))] dot r^((i)), n \/ |A|)\
  & = frac(1, n \/ |A|) sum_(i : a^((i)) = pi (x^((i)))) r^((i))
$

Because $n \/ |A|$ is the *expected* number of samples where
$a^((i))$ matches $pi$, not the true number of matched samples, which
causes variance.

Solution: use the true number of matched samples as the denominator:

$ frac(1, |{ i : a^((i)) = pi(x^((i))) }|) sum_(a^((i)) = pi(x^((i)))) r_i $

This is called the *self-normalized IS estimator*.

== Multi-step IS in MDPs
<multi-step-is-in-mdps>

- Data: trajectories starting from $s_1~d_0$ using $pi_b$
  (i.e., $a_t~pi_b (s_t)$ ). For simplicity, assume the process
  terminates in $H$ time steps.
$ {(s_1^((i)), a_1^((i)), r_1^((i)), s_2^((i)), ..., s_H^((i)), a_H^((i)), r_H^((i)))}_(i = 1)^n $
- Want to estimate $J(pi) := EE_(s~d_0) [V^pi(s)]$.
- Same idea as in the bandit case: apply IS to the entire trajectory.

Define $tau$ as the whole trajectory. The function of interest is
$tau mapsto sum_(t = 1)^H gamma^(t - 1) r_t$.
- Let the distribution of trajectory induced by $pi$ be $p(tau)$.
- Let the distribution of trajectory induced by $pi_b$ be $q(tau)$.
- IS estimator:
  $frac(p(tau), q(tau)) dot sum_(t = 1)^H gamma^(t - 1) r_t$.

How to compute $p(tau) \/ q(tau)$ ?

- $p(tau) = d_0 (s_1) dot pi (a_1|s_1) dot P (s_2|s_1, a_1) dot pi (a_2|s_2) ... P (s_H|s_(H - 1), a_(H - 1)) dot pi (a_H|s_H)$
- $q(tau) = d_0 (s_1) dot pi_b (a_1|s_1) dot P (s_2|s_1, a_1) dot pi_b (a_2|s_2) ... P (s_H|s_(H - 1), a_(H - 1)) dot pi_b (a_H|s_H)$

All $P(dot|dot)$ and $d_0$ terms cancel out.

Let $rho_t = frac(pi (a_t|s_t), pi_b (a_t|s_t))$ , then
$frac(p(tau), q(tau)) = product_(t = 1)^H rho_t = : rho_(1 : H)$.

=== Examine the Special Case

- $pi$ is deterministic, and $pi_b$ is uniformly random
  $(pi_b (a|x) equiv 1 \/ |A|)$.

- $rho_t = frac(II [a_t = pi (s_t)], 1 \/ |A|)$

- Only look at trajectories where *all* actions happen to match what $pi$
  wants to take.
  - If match: $rho = |A|^H$; mismatch: $rho = 0$.

- On average, only $1 \/ |A|^H$ portion of the data is useful. This exponential scaling is the *curse of horizon* in importance sampling.

= Policy Gradient Methods

Given a parametrized policy $pi_theta$, optimize
$J (pi_theta) := EE_(s~d_0) [med V^(pi_theta)(s)]$
where $d_0$ is the initial state distribution.

Key properties of the policy gradient approach:

- Use Gradient Ascent ($nabla_theta J (pi_theta)$).
- An unbiased estimate can be obtained from a #strong[single on-policy
    trajectory].
- No need for knowledge of $P$ and $R$ of the MDP.
- Similar to #link(<importance-sampling>)[Importance Sampling].

Note that when we write $pi$, we mean $pi_theta$, and
$nabla$ means $nabla_theta$.

Why use policy gradient?

- Goal: find a good policy directly.
- Value-based RL is indirect: it first estimates value functions and then derives a policy.
- PG is not based on the Bellman equation, and it is possible for a good policy not to satisfy the Bellman equation exactly.

== Example of Policy Parametrization

Linear + softmax:

- Featurize state-action: $phi.alt : S times A -> RR^d$
- Policy (softmax):
  $pi(a|s) prop e^(theta^top phi.alt(s, a))$

Recall in SARSA, we also used
#link(<softmax>)[softmax]
with temperature $T$. But in PG, we don't need a separate temperature parameter. Why?

- In SARSA, the softmax policy is based on the $Q$ function, which has a fixed scale.
- In PG, $theta^top phi.alt(s, a)$ is an arbitrary parametrization, so the scale of $theta$ already absorbs the temperature.

== PG Derivation

- The trajectory induced by $pi$:
  $tau := (s_1, a_1, r_1, ..., s_H, a_H, r_H)$ and
  $tau~pi$.
- Let $R(tau) := sum_(t = 1)^H gamma^(t - 1) r_t$.

$ J(pi) := EE_pi [sum_(t = 1)^H gamma^(t - 1) r_t] = EE_(tau~pi)[R(tau)] $

$
    & nabla J (pi) \
  = & nabla sum_(tau in(S times A)^H) P^pi(tau) R(tau) \
  = & sum_tau (nabla P^pi(tau)) R(tau) \
  = & sum_tau frac(P^pi(tau), P^pi(tau)) nabla P^pi(tau) R(tau) \
  = & sum_tau P^pi(tau) nabla log P^pi(tau) R(tau) \
  = & EE_(tau~pi) [nabla log P^pi(tau) R(tau)]
$

and

$
  & nabla_theta log P^(pi_theta)(tau) = nabla_theta log (d_0 (s_1) pi (a_1|s_1) P (s_2| s_1, a_1) pi (a_2|s_2) ...)\
  & = cancel(nabla log d_0(s_1)) + nabla log pi (a_1|s_1) + cancel(nabla log P (s_2|s_1, a_1)) + nabla log pi (a_2|s_2) + ...\
  & = nabla log pi (a_1|s_1) + nabla log pi (a_2|s_2) + ...
$

Note that this form is similar to that discussed in
#link(<multi-step-is-in-mdps>)[Importance Sampling]:
the transition probabilities $P$ and the initial distribution $d_0$ cancel out.

Given that
$pi(a|s) = frac(e^(theta^top phi.alt(s, a)), sum_(a') e^(theta^top phi.alt(s, a')))$,
we can compute:

$
  & nabla log pi(a|s)\
  = & nabla (log (e^(theta^top phi.alt (s, a))) - log (sum_(a') e^(theta^top phi.alt (s, a'))))\
  = & phi.alt(s, a) - frac(sum_(a') e^(theta^top phi.alt(s, a')) phi.alt(s, a'), sum_(a') e^(theta^top phi.alt (s, a')))\
  = & phi.alt(s, a) - EE_(a'~pi)[phi.alt(s, a')]
$

Note that the expectation of this quantity over $a~pi$ is $0$:

$
  EE_(a~pi)[phi.alt(s, a) - EE_(a'~pi)[phi.alt(s, a')]] = 0
$

*Conclusion*:

So far we have:

$ nabla J(pi) = EE_pi [(sum_(t = 1)^H gamma^(t - 1) r_t) (sum_(t = 1)^H nabla log pi(a_t|s_t))] $

Using the relation $EE_pi [nabla log pi(a_t|s_t)] = sum_(a_t) nabla pi(a_t|s_t) = nabla 1 = 0$,
and the fact that for $t' < t$, the reward $r_(t')$ is independent of
$nabla log pi(a_t|s_t)$:

$ EE_pi [nabla log pi(a_t|s_t) r_(t')] = EE_pi [nabla log pi(a_t|s_t)] EE_pi [r_(t')] = 0 $

We can therefore drop past rewards and rewrite:

$ nabla J(pi) = EE_pi [sum_(t = 1)^H (nabla log pi(a_t|s_t) sum_(t' = t)^H gamma^(t' - 1) r_(t'))] $

== PG and Value-Based Methods

So far we have:

$ nabla J(pi) = EE_pi [sum_(t = 1)^H (nabla log pi(a_t|s_t) sum_(t' = t)^H gamma^(t' - 1) r_(t'))] . $

By adding a condition on the expectation:

$
  nabla J(pi) &=
  EE_(s_t , a_t ~ pi) [
    EE_pi [
      sum_(t=1)^H (
        nabla log pi(a_t|s_t)
        sum_(t'=t)^H gamma^(t'-1) r_(t')
      )
      | s_t, a_t
    ]
  ] \

  &=
  sum_(t = 1)^H
    EE_(s_t , a_t ~ d_t^pi) [
      nabla log pi(a_t | s_t)
      underbrace(
        EE_pi [
          sum_(t' = t)^H gamma^(t' - 1) r_(t')
          | s_t, a_t
        ]
        , gamma^(t - 1) Q^pi (s_t, a_t)
      )
  ] \

  &=
  sum_(t = 1)^H gamma^(t - 1)
  EE_underbrace(s comma a ~ d_t^pi, d^t) [
    nabla log pi(a_t|s_t) Q^pi (s, a)
  ] \

  &=
  1/(1 - gamma)
  EE_(s ~ d^pi , a ~ pi(s))
  [ Q^pi (s, a) nabla log pi(a|s) ]
$

=== Blending PG and Value-Based Methods

Instead of using the MC estimate $sum_(t' = t)^H gamma^(t' - 1) r_(t')$ for
$Q^pi (s_t, a_t)$, use an approximate value function
$hat(Q)(s_t, a_t)$, often trained by TD, e.g. expected SARSA:

$ Q(S_t , A_t) <- Q(S_t , A_t) + alpha [R_(t + 1) + gamma EE_pi [Q(S_(t + 1) , A_(t + 1)) | S_(t + 1)] - Q(S_t , A_t)] $

=== Actor-Critic

The parametrized *policy* is called the *actor*, and the
*value-function* estimate is called the *critic*.

#figure(image("img/Taxonomy-of-model-free-RL-algorithms-by-Schulman-43.png", width: 60%),
  caption: [
    Actor-Critic. #link("https://www.researchgate.net/profile/Nikolas-Wilhelm/publication/344770842/figure/fig1/AS:948651484516356@1603187547778/Taxonomy-of-model-free-RL-algorithms-by-Schulman-43.png")[(credit)]
  ]
)

=== Baseline in PG

For any function $f : S -> RR$,

$ nabla J(pi) = frac(1, 1 - gamma) EE_(s~d^pi, a~pi(s)) [(Q^pi(s, a) - f(s)) nabla log pi(a|s)] $

because $f(s)$ is independent of $a$, and $EE_(a~pi) [nabla log pi(a|s)] = 0$.

A common choice is $f = V^pi(s)$, which gives the *advantage function*:

$ nabla J(pi) = frac(1, 1 - gamma) EE_(s~d^pi, a~pi(s)) [A^pi(s, a) nabla log pi(a|s)] $

where $A^pi(s,a) = Q^pi(s,a) - V^pi(s)$. The baseline does not change the *expectation* of the gradient but lowers its *variance*, leading to more stable training.
