#import "../index.typ": template, tufted
#show: template.with(title: "Fundamentals of Reinforcement Learning")
#import "@preview/mmdr:0.2.1": mermaid
#show raw.where(lang: "mermaid"): it => mermaid(it.text)

== 1

Among all probability distributions over $\[ a, b] in RR$ , which
distribution has the highest variance? How large is that variance?

$ P(x) = cases(delim: "{", 1 / 2 & x = a, b, 0 & "otherwise") $

then

$ "Var"(x) = EE ((x - frac(a + b, 2))^2) = (frac(a + b, 2))^2 $

== 2

Let $X, Y$ be two random variables that follow some joint
distributions over $cal(X) times RR$ . Let $f : cal(X) -> RR$
be a real-valued function. Prove that

$ EE [(Y - f(X))^2] - EE [(f(X) - EE[Y|X])^2] = EE [(Y - EE[Y|X])^2] . $

=== Proof
<proof>

It suffies to prove

$
  EE [(Y - f(X))^2 -(f(X) - EE [Y divides X])^2 -(Y - EE [Y divides X])^2] = 0 \
  "i.e." quad EE [(E [Y | X] - Y)(E [Y | X] - f(X))] = 0 .
$

Given $E[Y|X]$ is a function of $X$ , let
$g(X) := E[Y|X] - f(X)$ then it suffies to prove

$ EE[EE[Y|X] g(X)] = EE[Y g(X)] . $

then

$
  "LHS" & = sum_(x_i) EE[Y|X = x_i] g(x_i) P_X (x_i)\
  & = sum_(x_i) g(x_i) P_X (x_i) sum_(y_i) y_i frac(P_(X, Y)(x_i, y_i), P_X (x_i))\
  & = sum_(x_i) sum_(y_i) g(x_i) y_i P_(X, Y)(x_i, y_i)\
  & = "RHS" qed 
$

=== Notes
<notes>

Let $f : cal(X) -> RR$ be a estimator from $X$ to $Y$ , this
equation shows that square error ( $l_2$ loss)
$EE [(Y - f(X))^2]$ is at least
$EE [(Y - EE[Y|X])^2]$ for $forall f$ and thus
cannot be arbitrarily small.

== 3

Let $A in RR^(n times n)$ be a positive-definite real symmetric
matrix, and $b in RR^n$ be a vector. $lambda$ is the largest
eigenvalue of $A$ , that is,


$lambda = max_(z :||z||_2 = 1)||A z||_2 . quad(1)$


Let $x^* $ be the solution to $x^*  = A x^*  + b$ .
Define $x_0 = 0$ and for $t > 0$ , $x_t := A x_(t - 1) + b$ . Prove that
$||x_t - x^* ||_2 <= lambda^t||x^* ||_2$
.

\(Hint: show that
$||x_t - x^* ||_2 <= lambda||x_(t - 1) - x^* ||_2$
). Also, you do not need to know any additional properties about the
largest eigenvalue of matrix; the proof is elementary given Eq. (1).)

=== Proof

substitude

$ b = x^*  - A x^* , $

then

$ x_t = A x_(t - 1) + b = A x_(t - 1) + x^*  - A x^* , $

and it suffies to prove

$
 ||x_t - x^* ||_2 <= lambda||x_(t - 1) - x^* ||_2\
  "i.e." quad||A x_(t - 1) - A x^* ||_2 <= lambda||x_(t - 1) - x^* ||_2\
$

With Equation (1),

$
 ||A(x_(t - 1) - x^* )||_2 <= lambda||x_(t - 1) - x^* ||_2 qed
$

== 4

Prove that
$gamma^(frac(log(1 \/ epsilon.alt), 1 - gamma)) <= epsilon.alt$
when $gamma, epsilon.alt in(0, 1)$ .

\(Hint: use the fact that $(1 - 1 \/ x)^x < 1 \/ e$ when $x > 1$ )

=== Proof

==== Lemma

$(1 - 1 \/ x)^x < 1 \/ e$ when $x > 1$

It suffies to prove

$ x log (1 - 1 / x) < - 1 . $

Substitude $u := 1 - 1 \/ x$ , then

$ log(u) < u - 1 $

holds.

For original proposition, substitude $u := frac(1, 1 - gamma)$ and
therefore $gamma = 1 - 1 / u$ , then It suffies to prove

$ (1 - 1 / u)^(u log(1 \/ epsilon.alt)) <= epsilon.alt $

with the lemma,

$
  (1 - 1 / u)^(u log(1 \/ epsilon.alt)) < (1 / e)^(log(1 \/ epsilon.alt)) = epsilon.alt qed
$

== example: Shortest Path

#figure(
  image("img/reinforcement-learning-lecture-1.png", width: 60%),
  caption: [Shortest Path],
)

- nodes: stats
- edges: actions

Greedy is not optimal.

*Bellman Equation* (Dynamic Programing):

$ V^* (d) = min{3 + V^* (g), 2 + V^* (f)} $

== Stochastic Shortest Path

Markov Decision Process (MDP)

#figure(
  image("img/reinforcement-learning-lecture-1-1.png", width: 60%),
  caption: [Stochastic Shortest Path],
)

*Bellman Equation*

$ V^star.op (c) = min {4 + 0.7 V^star.op (d) + 0.3 V^star.op (e), thin 2 + V^star.op (e)} $

optimal policy : $pi^*$

== Model-based RL

The states are unknown. Learn by *trial-and-error*

#figure(
  image("img/reinforcement-learning-lecture-1-2.png", width: 60%),
  caption: [a trajectory: $s_0->c->e->F->G$],
)

Need to recover the graph by collecting multiple *trajectories*.

Use imperial frequency to find probabilities.

Assume states & actions are visited uniformly.

=== exploration problem

Random exploration can be inefficient:

#figure(
  image("img/reinforcement-learning-lecture-1-5.png", width: 30%),
  caption: [example: video game],
)

== example: video game

Objective: maximize the reward

$ EE[sum_(t = 1)^oo r_t|pi] "or" EE[sum_(t=1)^oo gamma^(t - 1) r_t|pi] $

Problem: the graph is too large

#figure(
  image("img/reinforcement-learning-lecture-1-4.png", width: 60%)
)

There are states that the RL model have never seen, therefore need
*generalization*

=== Contextual bandits

- Even if the algorithm is good, if mamke bad actions at beginning, will
  not get good data.
- Keep taking bad actions (e.g.~guessing wrong label on image
  classification), don't know right action.
  - Compared with superivsed learning
- #link("https://en.wikipedia.org/wiki/Multi-armed_bandit")[Multi-armed bandit]

== RL steps

For round t = 1, 2, …,

- For time step h=1, 2, …, H, the learner
  - Observes $x_h^((t))$
  - Chooses $a_h^((t))$
  - Receives
    $r_h^((t))~R(x_h^((t)), a_h^((t)))$
  - Next $x_(h + 1)^((t))$ is generated as a function of
    $x_h^((t))$ and $a_h^((t))$ (or sometimes, all previous x's
    and a's within round t)

Recall $V^* , Q^* , V^pi, Q^pi$ .

$
  V^* (s) = max_(a in A)(underbrace(R(s, a) + gamma EE_(S'~P(dot|s, a)) [V^*  (s')], Q^* (s, a)))
$

== Bellman Operator

$
  forall f : S times A -> RR,\
 (cal(T) f)(s, a) =(R(s, a) + gamma EE_(s'~P(dot|s, a))[max_(a') f(s', a')])\
  "where" cal(T) : RR^(S A) -> RR^(S A) .
$

then

$
  Q^*  = cal(T) Q^* \
  V^*  = cal(T) V^* \
$

i.e.~$Q^* $ and $V^* $ are fixpoints of $cal(T)$ .

== Value Interation Algorithm (VI)
<value-interation-algorithm-vi>

$ "funtion " f_0 = arrow(0) in RR^(S A) $

Interation to calculate fix points:

for $k = 1, 2, 3, ...$ ,

$ f_k = cal(T) f_(k - 1) $

How to do interation

$ forall s, a : f_1(s, a) <- cal(T) f_0(s, a) $

- synchronized iteration: use all functions values from old version
  during the update.
- asynchronized iteration

== Convergence of VI

lemma: $cal(T)$ is a $gamma$ *\-contraction* under
$||dot||_oo$ where
$||dot||_oo := max_(x in(dot)) x$ .

which means

$||cal(T) f - cal(T) f'||_oo <= gamma||f - f'||_oo $

Proof.

$
    &||f_k - Q^* ||_oo \
  = &||cal(T) f_(k - 1) - Q^* ||_oo \
  = &||cal(T) f_(k - 1) - cal(T) Q^* ||_oo \
  <=^("lemma") & gamma||f_(k - 1) - Q^* ||_oo \
$

$
  =>||f_k - Q^* ||_oo <= gamma^k||f_(k - 1) - Q^* ||_oo "where" gamma in(0, 1) qed
$

Proof of lemma:

It suffies to prove

$
  & lr(|(cal(T) f - cal(T) f')(s, a)|)\
  = & lr(|(R(s, a) + gamma EE_(s'~P(dot |s, a))[max_(a') f(s', a')]) - (R(s, a) + gamma EE_(s'~P(dot| s, a))[max_(a') f(s', a')])|)\
  = & gamma lr(|EE_(s'~P(dot|s, a))[max_(a') f(s', a') - max_(a') f'(s', a')]|)
$

WLOG assume, $max_(a') f(s' , a') > max_(a') f' (s' , a')$ and $exists a^* : max_a f(s', a) = f(s', a^*)
$

$ & gamma lr(|EE_(s'~P(dot|s, a))[max_(a') f(s', a') - max_(a') f'(s', a')]|)\
= & gamma lr(|EE_(s'~P(dot|s, a))[f(s', a^* ) - max_(a') f'(s', a')]|)\
<= & gamma lr(|EE_(s'~P(dot|s, a))[f(s', a^* ) - f'(s', a^* )]|)\
<= & gamma lr(|f(s', a^* ) - f'(s', a^* )|)\
<= & gamma||f - f'||_oo $ 

greedy policy:

$
  pi^star.op (s) = arg max_(a in A) Q^star.op (s, a)
$

sequence of function:

$ f_0, f_1, f_2, ... -> Q^*  $

define

$ pi_(f_k)^star.op (s) = arg max_(a in A) f_k (s, a) $

Claim:

$||V^*  - V^(pi_f)||<= frac(2||f - Q^*  | |_oo, 1 - gamma) $

define operator $cal(T)$ :

$ (cal(T) f)(s) = max_(a in A) (R(s, a) + gamma E_(s'~P(dot|s, A)) [f (s')]) $

#quote(block: true)[
  Note: the $cal(T)$ in $cal(T) Q^* $ and $cal(T) V^* $ are *not the same*.
]

== $V^* $ Iteration

$
  f_0 = arrow(0)\
  f_k <- cal(T) f_(k - 1)
$

then

$ f_k (s) = max_("all possible " pi) EE [sum_(t = 1)^k gamma^(t - 1) r_t|s_1 = s, pi] $

#quote(block: true)[
  This is derived my the definaion of operator $cal(T)$.
]

Claim:

$||f_k - V^* ||lt.tilde gamma^k $

step 1: $f_k <= V^* $

step 2:

$
  f_k >= & EE [sum_(t = 1)^oo gamma^(t - 1) r_t|s_1 = s, pi^* ] - EE [sum_(t = k + 1)^oo gamma^(t - 1) r_t|s_1 = s, pi^* ]\
  >= & V^*  - r^k V_max qed
$


c

#quote(block: true)[
  this means, once reached goal $pi^* $ , never leave.
]

== example

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

Optimal policy is heading `Pasta`.

#quote(block: true)[
  This example is a finite horizon case. To make it infinite horizon
  discount, add a state $T$ :

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

To find $V^* (s)$ , update $V$ value from leaf upwards to root
state.

=== policy iteration (example)

==== interation \#0

define initial $pi_0$ :

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

then the corresponding $Q^(pi_0)(s, a)$ :

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


==== interation \#1

$ pi_1 (s) : = arg max_(a in A) Q^(pi_0) (s, a) $

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

$Q^(pi_i)$:

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

==== interation \#2

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

==== Comment

Policy $pi$ was switched to Japanese for once, and switched back to
Italian at the end.

Also, the policy updates upwards.

== Monotone Policy improvement

$ forall k, forall s : V^(pi_k) >= V^(pi_(k - 1)) $

$ "if" pi_(k - 1) eq.not pi^* , exists s : v^(pi_k)(s) > V^(pi_(k - 1))(s) $

$ => "#iteration" <= |A|^(|S|) $

#quote(block: true)[
  Monotone Policy improvement produces exact solutions, while value
  iteration produces approxmitate solutions, {: .prompt-tip }
]

Proof of: $Q^(pi_(k + 1)) >= Q^(pi_k)$

lemma 1:

$ Q^(pi_k) = cal(T)^(pi_k) Q^(pi_k) <= cal(T) Q^(pi_k) $

beacuse

$
  &(cal(T)^pi f)(s, a) = R(s, a) + gamma EE_(s'~p(dot|s, a)) [f (s', pi)]\
  <= &(cal(T) f)(s, a) = R(s, a) + gamma EE_(s'~p(dot|s, a)) [max_(a') f (s', a')]
$

lemma 2:

$ cal(T) Q^(pi_k) = cal(T)^(pi_(k + 1)) Q^(pi_k) $

lemma 3:

$ forall f >= f', cal(T)^pi f >= cal(T)^pi f' $

with lemma 1,2,3,

$
  Q^(pi_k) & <= cal(T)^(pi_(k + 1)) Q^(pi_k)\
  => cal(T)^(pi_(k + 1)) Q^(pi_k) & <= cal(T)^(pi_(k + 1)) cal(T)^(pi_(k + 1)) Q^(pi_k)\
  & ....v\
  => Q^(pi_k) & <= (cal(T)^(pi_(k + 1)))^oo Q^(pi_k) = Q^(pi_(k + 1))
$

== because $Q^(pi_(k + 1))$ is the fixed point of $cal(T)^(pi_(k + 1))$ .

== recap

in policy iteration, appply greedy algo very time.

\#steps are finite.

== another proof

=== performance-difference lemma (P-D lemma)

#quote(block: true)[
  this is a fundamental tool in RL. many deep RL models relies on this
  lemma {: .prompt-info }
]

$forall pi, pi', s$,

$ V^(pi')(s) - V^pi(s) = frac(1, 1 - gamma) E_(s'~d_s^(pi')) [Q^pi(s', pi') - V^pi(s')] $

apply the lemma in the policy iteration steps:

$
  V^(pi_(k + 1))(s) - V^(pi_k)(s) = frac(1, 1 - gamma) E_(s'~?) [Q^(pi_k)(s', pi_(k + 1)) - V^(pi_k)(s')]
$

and

$ V^(pi_k)(s') = Q^(pi_k)(s', pi_k) $

and RHS $>= 0$ is trivial. QED

=== Proof of lemma

= The Learning Setting

== planning and learning

Planning:

- given MDP model, how to compute optimal policy
- The MDP model is known

Learning:

- MDP model is unknown
- collect data from the MDP: $(s, a, r, s')$ .
- Data is limited. e.g., adaptive medical treatment, dialog systems
- Go, chess, …
- Learning can be useful *even if the final goal is planning*
  - especially when $|S|$ is large and/or only blackbox simulator
  - e.g., AlphaGo, video game playing, simulated robotics

== Monte-Carlo policy evaluation

Given $pi$ , estimate
$J(pi) := EE_(s~d_0) [V^pi(s)]$ ( $d_0$ is initial
state distribution) is the *actual* expectation of reward.

Monte-Carlo outputs some scalar $v$ \; accuracy measured by
$|v - J(pi)|$ . (by sampling different trajectories):

Data: trajectories starting from $s_1~d_0$ using $pi$ (i.e.,
$a_t = pi (s_t)$ ).

$
  {(s_1^((i)), a_1^((i)), r_1^((i)), s_2^((i)), ..., s_H^((i)), a_H^((i)), r_H^((i)))}_(i = 1)^n
$

#quote(block: true)[
  this is called on-policy: evaluating a policy with data collected from
  the exactly same policy.

  Othwise, it is off-policy. {: .prompt-info }
]

Estimator:

$ 1 / n sum_(i = 1)^n sum_(t = 1)^H gamma^(t - 1) r_t^((i)) $

#quote(block: true)[
  Guarantee: w.p. at least
  $1 - delta, |v - J(pi)| <= frac(R_max, 1 - gamma) sqrt(frac(1, 2 n) ln 2 / delta)$
  (larger n, higher accuracy)

  It is *independent* to the size of state space {: .prompt-tip }
]

=== Comment on Monte-Carlo

Monte-Carlo is a Zeroth-order (ZO) optimization method, which is not
efficient.

- *first order*: gradient / first derivative (in DL/ML,
  *SDG*)
- *second order*: Hessian matrix / second derivative

== Model-based RL with a sampling oracle (Certainty Equivalence)

#quote(block: true)[
  Assuming the reward / probability is determined (constant) via sampling.
  {: .prompt-info }
]

Assume we can sample $r~R(s, a)$ and
$s'~P(s, a)$ for any $(s, a)$

Collect $n$ samples per
$(s, a)$: $(r_i , s'_i)_(i = 1)^n$ .
Total sample size $n |S times A|$

Estimate an empirical MDP $hat(M)$ from data

- $hat(R)(s, a) := 1 / n sum_(i = 1)^n r_i, quad hat(P) (s'|s, a) := 1 / n sum_(i = 1)^n II [s'_i = s']$
- i.e., treat the empirical frequencies of states appearing in
  ${ s'_i }_(i = 1)^n$ as the true distribution.

Plan in the estimated model and return the optimal policy

transition tuples: $(s_i, a_i, r_i, s_(i + 1))$ . Use
$s_i, a_i$ to identify current state and action, use $r_i$ for reward
and $s_(i + 1)$ for transition.

extract transition tuples from trajectories.

=== finding policy on estimated environment

*true* environment: $M =(S, A, P, R, gamma)$

*estimated* environment:
$hat(M) =(S, A, hat(P), hat(R), gamma)$

- notation: $pi_(hat(M)), thin V_(hat(M)), thin ...$

performance measurement:

- in the *true* environment, use
  $||V^*  - V^(pi_f)||$ where $f approx Q^* $

- in *estimated* environment, use $||V_M^*  - V_M^(pi_(hat(M))^* )||$ , i.e.~measure the optimal policy of estimated environment in the real environment.

== Model-based RL with a sampling oracle (Certainty Equivalence) *Cont'd*
<model-based-rl-with-a-sampling-oracle-certainty-equivalence-contd>

To find $Q_(hat(M))^* $ with empirical $hat(R)$ and $hat(P)$ :

$ f_0 in RR^(S A), quad f_k in hat(cal(T)) f_(k - 1) . $

where

$
  (hat(cal(T)) f)(s, a) &= hat(R) (s, a) + gamma EE_(s^prime ~ hat(P) (dot.op divides s, a)) underbrace([max_(a^prime) f(s^prime comma a^prime)], V_f (s^prime)) \
&= 1/n sum_(i = 1)^n r_i + gamma lr(chevron.l hat(P) (dot.op divides s, a), V_f chevron.r) \
&= 1/n sum_(i = 1)^n gamma_i + gamma sum_(s^prime) (1/n sum_(i = 1)^n II [s_i^prime = s^prime]) dot.op V_f (s^prime) \
&= 1/n sum_(i = 1)^n r_i + gamma/n sum_(i = 1)^n (sum_(s^prime) II [s_i^prime = s^prime] V_f (s^prime)) \
&= 1/n sum_(i = 1)^n r_i + gamma dot.op 1/n sum_(i = 1)^n V_f (s_i^prime) \
&= 1/n sum_(i = 1)^n (r_i + gamma max_(a^prime) f(s_i^prime , a^prime)) \
$

is call the *Empirical Bellman Update*.

=== Computational Complexity

==== Value Interation

For original
#link(<value-interation-algorithm-vi>)[value iteration],
the Computational Complexity is

$ |S| times |A| times |S| $

$|S| times |A|$ for each $f(s,a)$ and $|S|$ for expectation.

==== Empirical Bellman Update

For Empirical Bellman Update, the Computational Complexity is

$ |S| times |A| times n $

Empirical sampling for $n$ times.

== the Value Prediction Problem

Given $pi$ , wnat to know $V^pi$ and $Q^pi$ .

/ On-policy Learning: #block[
    data used to improve policy $pi$ is generated by $pi$ .
  ]

/ Off-policy Learning: #block[
    data used to improve policy $pi$ is generated by some other policies.
  ]

When action is always chosen by a fixed policy, the MDP reduces to a
Markov chain plus a reward function over states, also known as Markov
Reward Processes (MRP)

=== Monte-Carlo Value Prediction
<monte-carlo-value-prediction>

For each $s$ , roll out $n$ trajectories using policy $pi$

- For episodic tasks, roll out until termination
- For continuing tasks, roll out to a length (typically
  $H = O(1 \/(1 - gamma))$ ) such that omitting the
  future rewards has minimal impact ("small truncation error")
- Let $hat(V)^pi(s)$ (will just write $V(s)$ ) be the average
  discounted return

*Online Monte-Carlo*

- For $i = 1, 2, ...$ as the index of trajectories
  - Draw a starting state $s_i$ from the exploratory initial
    distribution, roll out a trajectory using $pi$ from $s_i$ , and let
    $G_i$ be the (random) discounted return
  - Let $n (s_i)$ be the number of times $s_i$ has appeared as an
    initial state. If $n (s_i) = 1$ (first time seeing this state), let
    $V (s_i) <- G_i$ (where
    $G_t = sum_(t' = t)^(t + H) gamma^(t' - t) r_(t')$ )
  - Otherwise,
    $V (s_i) <- frac(n (s_i) - 1, n (s_i)) V (s_i) + frac(1, n (s_i)) G_i$

No need to store the trajectory.

More generally,

$ V(s_i) <-(1 - alpha) V(s_i) + alpha G_i $

or

$ V (s_i) <- V (s_i) + alpha (G_i - V (s_i)) $

where $alpha$ is known as learning rate, and $G_i$ as the target.

#quote(block: true)[
  It can be interpreted as stochastic gradient descent. If we have i.i.d.
  real random variables $v_1, v_2, ..., v_n$ , the average is the
  solution of the least-square optimization problem:

  $ min_v frac(1, 2 n) sum_(i = 1)^n (v - v_i)^2 $
]

=== Every-visit Monte-Carlo

Suppose we Have a continuing task. What/if we cannot set the starting
state arbitrarily?

i.e.~we have a single *long* trajectory with length $N$

$ s_1, a_1, r_1, s_2, a_2, r_2, s_3, a_3, r_3, ... $

- we can truncate $N \/ H$ truncations with length
  $H = O(1 \/(1 - gamma))$ from the long trajectory.
- we can shift the $H$ -length window by 1 each time and get
  $N - H + 1 approx N$ truncations.

This "walk" through the state space should have non-zero probability on
each state, i.e.~do not starve every states.

What if a state occures multiple times on a trajectory?

- approach 1: only the first occurance is used
- approach 2: all the occurances are used

== Alternative Approach: TD(0)

Again, suppose we have a single long trajectory
$s_1, a_1, r_1, s_2, a_2, r_2$ ,
$s_3, a_3, r_3, s_4, ...$ in a continuing task

TD(0): for
$t = 1, 2, ..., V (s_t) <- V (s_t) + alpha (r_t + gamma V (s_(t + 1)) - V (s_t))$

/ TD: #block[
    temporal difference
  ]

/ TD\_error: #block[
    $r_t + gamma V (s_(t + 1)) - V (s_t)$
  ]

Same as Monte-Carlo update rule, excepts that the "target" is
$r_t + gamma V (s_(t + 1))$ , which is similar to the
#link(<model-based-rl-with-a-sampling-oracle-certainty-equivalence-contd>)[empirical Bellman update].

Recall that in
#link(<monte-carlo-value-prediction>)[Monte-Carlo],
the "target" is $G_t = sum_(t' = t)^(t + H) gamma^(t' - t) r_(t')$ and
is *independent* to the current value function. While in TD(0),
the target $r_t + gamma V (s_(t + 1))$ is dependent to the current value
function $V$ . i.e.

Compared to value iteration:

$ V_(k + 1)(s) := EE_(r, s'|s, pi) [r + gamma V_k (s')] $

and the equation above is

$ approx 1 / n sum_(i = 1)^n (r_i + r V_k (s'_i)) $

which is an approximate Value Iteration process, and notice that the
whole iteraton through $i = 1, ..., n$ is only 1 iteration (a
$V_k$ ), so an outside loop is needed if we want to $V$ approximates
real $V^pi$ .

=== Understanding TD(0)

The "approximate" Value Iteration process above is similar to TD(0) but
slightly different: it uses a value function $V$ (which stays constant
during updates) to update $V'$ which is another function. After long
enough, we have $V' = cal(T)^pi V$ and do $V <- V'$ , then repeat
the process. Finally converges to $V^pi$ .

But in TD(0), we uses $V$ to update itself. The difference is
"synchronous" vs "asynchronous".

#quote(block: true)[
  TD(0) is less stable {: .prompt-info } 
]

== TD( $lambda$ ): Unifying TD(0) and MC

- 1-step bootstrap (TD(0)): $r_1 + gamma V(s_(i + 1))$
- 2-step bootstrap: $r_1 + gamma r_(i + 1) + gamma^2 V(s_(i + 2))$
- 3-step bootstrap:
  $r_1 + gamma r_(i + 1) + gamma^2 r_(i + 2) + gamma^3 V(s_(i + 3))$
- …
- $oo$ -step bootstrap:
  $r_1 + gamma r_(i + 1) + gamma^2 r_(i + 2) + gamma^3 r_(i + 3) + ...$
  is Monte-Carlo.

=== Proof of TD( $lambda$ )'s correctness

E.g. in 2-step bootstrap,

With #link("https://en.wikipedia.org/wiki/Law_of_total_expectation")[Law of total expectation],

$
  & EE[r_1 + gamma r_(t + 1) + gamma^2 V(s_(t + 2))|s_t]\
  = & EE[r_t + gamma(r_(t + 1) + gamma V(s_(t r)))|s_t]\
  = & EE[r_t] + gamma EE_(s_(t + 1) |s_t) #scale(x: 120%, y: 120%)[\[] EE[(r_(t + 1) + gamma V(s_(t r)))| s_t, s_(t + 1)] #scale(x: 120%, y: 120%)[\]]\
  = & EE[r_t + gamma(cal(T)^pi)(s_(t + 1))|s_t]\
  = &((cal(T)^pi)^2 V)(s)
$

=== TD( $lambda$ )

For n-step bootstrap, give a $(1 - lambda) lambda^n$ weight.

- $lambda = 0$ : Only n=1 gives the full weight. TD(0).
- $lambda -> 1$ : (almost) Monte-Carlo.

==== forward view and backward view

Forward view

$
 (1 - lambda) dot (r_1 + gamma V(s_2) - V(s_1))\
 (1 - lambda) lambda dot (r_1 + gamma r_2 + gamma^2 V(s_3) - V(s_1))\
 (1 - lambda) lambda^2 dot (r_1 + gamma r_2 + gamma^2 r_3 + gamma^3 V(s_4) - V(s_1))\
  ...
$

and so on.

Backward view

$ 1 dot (r_1 + gamma V(s_2) - V(s_1))\
lambda gamma dot (r_2 + gamma V(s_3) - V(s_2))\
lambda^2 gamma^2 dot (r_3 + gamma V(s_4) - V(s_3))\
... $ 

== Value Prediction with Function Approximation
<value-prediction-with-function-approximation>

tabular representation vs. function approximation:
#block[
  function approximation can handle infinite state space (can't enumerate
  through all states).
]

linear function approximation:
#block[
  design features $phi.alt(s) in RR^d$ ("featurizing states"), and
  approximate $V^pi(s) approx theta^top phi.alt(s) + b$, where $theta$ should be fixed among features (in the following parts, $b$ is *ignored* because it can be reached by appending a $1$ to the feature vector)
]

#block[
   tabular value function can be interpreted as feature vector $in RR^S$ : $[0, ..., 0, 1, 0, ..., 0]$ where the position of the $1$ indicates the state.
]

=== Example: Tetris Game

#figure(image("img/reinforcement-learning-lecture-14-1.png"), caption: [
  Tetris Game
])

The state space is exponentially large: each block be occupied / not
occupied.

Featurize: \# of blocks on each column. In the example, the feature is $(4,4,5,4,3,3,3)$

=== Monte-Carlo Vaule Prediction

$ V^pi (s) = EE [G divides s] = arg min_(f : S -> RR) EE [(f(s) - G)^2] $

Is a regression problem.

#quote(block: true)[
  Why the expectation is the argmin? See
  #link(<notes>)[here]
]

The same idea applies to non-linear value function approximation More
generally & abstractly, think of function approximation as searching
over a restricted *function space*, which is a set whose members
are functions that map states to real values.

E.g. a function space of linear value function approximation:

$
  cal(F) = {V_theta : theta in RR^(d)} ", where " V_theta (s) = theta^top phi.alt(s)
$

- typically only a small subset of all possible functions
- Using "all possible functions" = tabular!
- Equivalently, tabular MC value prediction can be recovered by choosing
  $phi.alt$ as the identity features $phi.alt(s) = {II [s = s']}_(s' in S)$

Find the function:

$ min_(V_theta in cal(F)) 1 / n sum_(i = 1)^n (V_theta (s_i) - G_i)^2 $

SGD: uniformly sample $i$ and

$
  theta <- theta - alpha dot (V_theta (s_(i)) - G_(i)) dot nabla V_theta (s_(i))
$

=== Interprete Td(0) with Linear Approximation

TD(0) iteration is equivalent to

$ theta <- theta + alpha (G_t - phi.alt (s_t)^top theta) phi.alt (s_t) . $

Here $theta$ is the tabular value function and $phi.alt$ is
$\[ 0, ..., 0, 1, 0, ..., 0]$ , as mentioned
#link(<value-prediction-with-function-approximation>)[here].

=== TD(0) with Linear Approximation

In TD(0), we do

$ V (s_t) <- V (s_t) + alpha (r_t + gamma V (s_(t + 1)) - V (s_t)), $

which, with all steps on $t$ , gets

$ V_(k + 1) <- cal(T)^pi V_k . $

i.e.

$ V_(k + 1)(s) = EE_pi [r + gamma V_k (s')|s] $

Similar to Linear Approximation, *rewriting expectation with a regression problem*,

$
  V_(k + 1) (s) =
  arg min_(f : s -> RR) EE_pi [(f(s) -(r + gamma V_s))^2] \
  approx arg min_(V_theta in scr(F)) 1/n sum_(i = 1)^n (V_theta (s_i) - r_i - gamma V_k (s^prime))^2 .
$

And the SGD steps should be

$ theta <- theta + alpha (V_theta (s_t) - r_t - gamma V_k (s_(t + 1))) nabla V_theta (s_t) $


Recall the Bellman Equation:

$
  (T^pi f)(s, a) & = R(s, a) + gamma EE_(s'~P(k, a)) [f (s', pi)] \
                        & = EE [r + gamma dot f (s', pi)|s, a] .
$

with empirically equals to:

$ 1 / n sum_(i = 1)^n (r_i + gamma theta_(k_1) (s'_i, pi)) . $

with tuples $(s_t, a_t, r_t, s_(t + 1))$ in the long
trajectory, applying the running average:

$
  Q_k (s_t, a_t) <- Q_k (s_t, a_t) + alpha(r_t + gamma Q_(k - 1)(s_(t + 1), pi) - Q_k (s_t, a_t))
$

== SARSA

$ Q (s_t, a_t) <- Q (s_t, a_t) + alpha (r_t + gamma Q (s_t + 1, a_t + 1) - Q (s_t, a_t)) $

Notice that SARSA is not applicable for deterministic policy, because it
requires a non-zero probability distribution over *all*
st0ate-action pairs ( $forall(s, a) in S times A$ ), but the only
possible action for a certain state is determined by the policy.

=== SARSA with $epsilon.alt$-greedy policy

How are the $s, a$ data pairs picked in SARSA?

At each time step t, with probability $epsilon.alt$ , choose a from the
action space uniformly at random. otherwise, $a_t = arg max_a Q(s_t , a)$

#quote(block: true)[
  When sampling s-a-r-s-a tuple along the trajectory, the first action in
  the tuple is actually generated with last version of $Q$ , so we can say
  SARSA is not 100% "on policy".
]

=== Does SARSA converge to optimal policy?

The cliff example (pg 132 of Sutton & Barto)

- Deterministic navigation, high penalty when falling off the clif
- Optimal policy: walk near the cliff
- Unless epsilon is super small, SARSA will avoid the cliff

#figure(
  image("img/reinforcement-learning-lecture-15.png", width: 60%),
  caption: [cliff example],
)

The optimal path is along the side of the cliff, but on this path, the
$epsilon.alt$ -greedy SARSA will often see large penalty (falling off
the cliff) and therefore, choose the safe path instead.

=== softmax
<softmax>

$epsilon.alt$-greedy can be replaged by softmax: chooses action a with
probability

$ frac(exp(Q(s, a) \/ T), sum_(a') exp(Q(s, a') \/ T)) $

== where $T$ is temperature.

== Q-learning

Update rule:

$ Q (s_t, a_t) <- Q (s_t, a_t) + alpha (r_t + gamma max_(a') Q (s_(t + 1), a') - Q (s_t, a_t)) $

Q-learning is off-policy: how we take actions have nothing to do with
our current Q-estimate (or its greedy policy). i.e.~Q-learning always
taks $max_(a') Q (s_(t + 1), a')$ no matter what the real policy is.

e.g.~in the cliff setting, the optimal can always be found, no matter
the choice of $epsilon.alt$ .

=== Exercise: Multi-step Q-learning?

Does the target
$r_t + gamma r_(t + 1) + gamma^2 max_(a') Q (s_(t + 2), a')$ work? If
not, why?

No.~Because it leads to

$ Q <- cal(T)^pi cal(T) Q $

#quote(block: true)[
  This resulting $cal(T)^pi cal(T) ... cal(T)^pi cal(T) Q$ is also a
  optimal policy, but for another MDP, i.e. on odd steps, follow $pi$ , on
  even steps, free to decide.
]

== Q-learning with experience replay

So far most algorithms we see are "one-pass"

- i.e., use each data point once and discard them
- \# updates $=$ \# data points
- Concern 1: We need many updates for optimization to converge Can we separate optimization from data collection?
- Concern 2: Need to reuse data if sample size is limited Sample (with replacement) a tuple randomly from the bag, and apply the Q-learning update rule.
- \# updates $>>$ \# data points

Each time get a new tuple, put in bag, and do updates for several times.

== Not applicable for on-policy controls (e.g.~SARSA).

== A Question

$ EE_(s, r, s') [(V_theta (s) - r - gamma V_theta (s'))^2] $

We do
$V_theta (s) <- V_theta (s) + alpha(r - gamma V_theta (s') - V_theta (s))$
in TD(0).

What if we minimize the square error between $V_theta (s)$ and its
target, i.e.
$EE_(s, r, s') [(V_theta (s) - r - gamma V_theta (s'))^2]$
?

No correct. It can be #link(<proof>)[decomposed] as the sum of 2 parts:

- $EE_s [(V_theta (s) - (cal(T)^pi V_theta)(s))^2]$
  - good. It's L-2 norm Bellman Error.
- $gamma^2 EE_s ["Var"_(s'|s, pi(s)) [V_theta (s')]]$
  - Not good. It penalize policy with large variance.
  - OK for deterministic environment because the variance is always $0$
    in this case.

=== Solution

If we have a simulator, for each $s$ in data, draw another independent
state transition.

Minimize objective

$ EE \[(V_theta (s) - r - gamma V_theta (s'_A)) lr((V_theta (s) - r - gamma V_theta (s'_B)]) $

"Double sampling" and Baird's residual algorithm (Bellman residual
minimization).

== Convergence

- TD with function approximation can diverge in general
- Is it because of…
  - Randomness in SGD?
    - Nope. Even the batch version doesn't converge
  - Sophisticated, non-linear func approx?
    - Nope. Even linear doesn't converge.
  - That our function class does not capture V”?
    - Nope. Even if V” can be exactly represented in the function class
      ("realizable"), it still does not converge.

=== example

For a MDP: $1 -> 2 -> ... -> 9 -> 10$ with reward $~"Ber"(0.5)$

iterations

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

Assume the function space has to possible values at each state:

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

then

#figure(
  table(
    columns: 6,
    align: (auto, auto, auto, auto, auto, auto),
    table.header([\#Ite], [1], [2], […], [9], [10]),
    table.hline(),
    [1], [], [], [], [], [0.501],
    [2], [], [], [], [0.502], [0.501],
    […], [], [], [], [], [],
    [10], [1.012], [0.756], […], [0.502], [0.501],
  ),
  kind: table,
)

Value deviates from 0.501 as iteration goes.

Say the function space is a *plane*, than the results of each
iteration (bellman operator) is not on the plane, instead, their
*projections* are picked.

== Importance Sampling
<importance-sampling>

We can only sample $x~q$ but want to estimate
$EE_(x~p) f(x)$

Importance Sampling (or importance weighted, or inverse propensity
yscore Ps estimator):

$ frac(p(x), q(x)) f(x) $

Unbiasedness:

$ EE_(x~q) [frac(p(x), q(x)) f(x)] = sum_x q(x) (frac(p(x), q(x)) f(x)) = sum_x p(x) f(x) = EE_(x~p)[f(x)] $

== Application in contextual bandit (CB)

- The data point is a tuple $(x, a, r)$
- The function of interest is $(x, a, r) mapsto r$
- The distribution of interest is
  $x~d_0, a~pi, r~R(x, a)$
  - Let the joint density be $p(x, a, r)$
- The data distribution is
  $x~d_0, a~pi_b, r~R(x, a)$
  - Let the joint density be $q(x, a, r)$
- IS estimator:
  $frac(p(x, a, r), q(x, a, r)) dot r$
- Write down the densities
  - $p(x, a, r) = d_0(x) dot pi(a|x) dot R(r|x, a)$
  - $q(x, a, r) = d_0(x) dot pi_b (a|x) dot R(r|x, a)$
  - To compute importance weight, you don't need knowledge of $mu$ or
    $R$ ! You just need $pi_b$ (or even just $pi_b (a|x)$ ,
    "proposal prob.")
- Let $rho$ be a shorthand for $pi(a|x)$ , so estimator is
  $rho dot r$
- $pi_b$ need to "cover" $pi$
  - i.e., whenever $pi(a|x) > 0$ , we need\$ \_b(a x)\>0\$
- A special case:
  - $pi$ is deterministic, and $pi_b$ is uniformly random
    $(pi_b (a|x) equiv 1 \/ |A|)$
  - $frac(II[a = pi(x)], 1 \/ |A|) r$
    - only look at actions that match what $pi$ wants to take and
      discard other data points
    - If match, $rho = |A|$ \; mismatch: $rho = 0$
  - On average: only $1 \/ |A|$ portion of the data is useful

=== A note about using IS

- We know that shifting rewards do not matter (for planning purposes)
  for fixed-horizon problems
- However, when you apply IS, shifting rewards do impact the variance of
  the estimator
- Special case:
  - deterministic $pi$ , uniformly random $pi_b$ ,
  - reward is deterministic and constant: regardless of $(x, a)$ ,
    reward is always 1 (without any randomness)
  - We know the value of any policy is 1
  - On-policy MC has 0 variance
  - IS still has high variance!
- Where does variance come from?

$
  & 1 / n sum_(i = 1)^n frac(II [a^((i)) = pi (x^((i)))], 1 \/ |A|) dot r^((i)) = sum_(i = 1)^n frac(II [a^((i)) = pi (x^((i)))] dot r^((i)), n \/ |A|)\
  & = frac(1, n \/ |A|) sum_(i : a^((i)) = pi (x^((i)))) r^((i))
$

Because $n \/ |A|$ is the *expectaton* of \# of sampling
$a^((i))$ matches $pi$ but not the true \# of matched samples, which
causes variance.

Solution: use true \# of matched samples as denometer,

$ frac(1, |{ i : a^((i)) = pi^((i)) }|) sum_(a^((i)) = pi^((i))) r_i $

== Multi-step IS in MDPs
<multi-step-is-in-mdps>

- Data: trajectories starting from $s_1~d_0$ using $pi_b$
  (i.e., $a_t~pi_b (s_t)$ ) (for simplicity, assume process
  terminates in $H$ time steps)
$ {(s_1^((i)), a_1^((i)), r_1^((i)), s_2^((i)), ..., s_H^((i)), a_H^((i)), r_H^((i)))}_(i = 1)^n $
- Want to estimate $J(pi:= EE_(s~d_0) [V^pi(s)]$
- Same idea as in bandit: apply IS to the entire trajectory

= = =

- define $tau$ as the whole trajectory. The function of interest is
  $tau mapsto sum_(t = 1)^H gamma_partial^(t - 1) r_t$ .
- Let the distribution of trajectory induced by $pi$ be $p(tau)$
- Let the distribution of trajectory induced by $pi_b$ be $q(tau)$
- IS estimator:
  $frac(p(tau), q(tau)) dot sum_(t = 1)^H gamma^(t - 1) r_t$

How to compute $p(tau) \/ q(tau)$ ?

- $p(tau) = d_0 (s_1) dot pi (a_1|s_1) dot P (s_2|s_1, a_1) dot pi (a_2|s_2) ... P (s_H|s_(H - 1), a_(H - 1)) dot pi (a_H|s_H)$
- $q(tau) = d_0 (s_1) dot pi_b (a_1|s_1) dot P (s_2|s_1, a_1) dot pi_b (a_2|s_2) ... P (s_H|s_(H - 1), a_(H - 1)) dot pi_b (a_H|s_H)$

Here all $P(dot|dot)$ terms are cancelled out.

Let $rho_t = frac(pi (d_t|s_t), pi_b (a_t|s_t))$ , then
$frac(p(tau), q(tau)) = product_(t = 1)^H rho_t = : rho_(1 : H)$

=== Examine the special case again

- $pi$ is deterministic, and $pi_b$ is uniformly random
  $(pi_b (a|x) equiv 1 \/ |A|)$

- $rho_t = frac(II [a_t = pi (s_t)], 1 \/ |A|)$

- only look at trajectories where all actions happen to match what $pi$
  wants to take
  - Only if match, $rho = |A|^H$ \; mismatch: $rho = 0$

- == On average: only $1 \/ |A|^H$ portion of the data is useful
  <on-average-only-1-ah-portion-of-the-data-is-useful>

== Policy Gradient

Given policy $pi_theta$, optimize
$J (pi_theta) := EE_(s~d_0) [med V^(pi_theta)(s)]$
where $d_0$ is the initial state distribution.

- Use Gradient Ascent ($nabla_theta J (pi_theta)$)
- an unbiased estimate can be obtained from a #strong[single on-policy
    trajectory]
- no need of the knowledge of $P$ and $R$ of the MDP
- Similar to #link(<importance-sampling>)[IS]

Note that when we use $pi$, we mean $pi_theta$ here, and
$nabla$ means $nabla_theta$.

About PG:

- Goal: we want to find good policy.
- Value-based RL is indirect
- PG isn't based on value function
  - It's possible a good policy don't match Bellman Equation.

=== Example of policy parametrization

Linear + softmax:

- Featurize state-action: $phi.alt : S times A -> RR^d$
- Policy (softmax):
  $pi(a|s) prop e^(theta^top phi.alt(s, a))$

Recall in SARSA, we also used
#link(<softmax>)[softmax]
with temperature $T$. But in PG, we don't need it. Why?

- In SARSA, softmax policy based on $Q$ function ------ $Q$ function cannot
  be arbitrary.
- In PG, $phi.alt(s, a)$ is arbitrary function ------ $T$ is
  included.

=== PG Derivation

- The trajectory inducded by $pi$:
  $tau := (s_1, a_1, r_1, ..., s_H, a_H, r_H)$ and
  $tau~pi$.
- Let $R(tau) := sum_(t = 1)^H gamma^(t - 1) r_t$

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
#link(<multi-step-is-in-mdps>)[Importance Sampling].

Given that
$pi(a|s) = frac(e^(theta^top phi.alt(s, a)), sum_(a') e^(theta^top phi.alt(s, a')))$
(denoted by $pi(a|s)$).

$
  & nabla log pi(a|s)\
  = & nabla (log (e^(theta^top phi.alt (s, a'))) - log (sum_(a') e^(theta^top phi.alt (s, a'))))\
  = & phi.alt(s, a) - frac(sum_(a') e^(theta^top phi.alt(s, a')) phi.alt(s, a'), sum_(a') e^(theta^top phi.alt (s, a')))\
  = & phi.alt(s, a) - EE_(a'~pi)[phi.alt(s, a')]
$

Note that the expectation of the quantity above is $0$. i.e.

$
  EE_(a~pi)[phi.alt(s, a) - EE_(a'~pi)[phi.alt(s, a')]] = 0
$

*Couclusion*:

So far we have

$ nabla J(pi) = EE_pi [(sum_(t = 1)^H gamma^(t - 1) r_t) (sum_(t = 1)^H nabla log pi(a_t|s_t))] $

With the relation discussed above, we say
$EE_pi [nabla log pi(a_t|s_t)] = sum_(a_t) nabla pi(a_t|s_t) = nabla 1 = 0$

So, for $t' < t$, $r_(t')$ is independent to
$nabla log pi(a_t|s_t)$, we have

$ EE_pi [nabla log pi(a_t|s_t) r_(t')] = EE_pi [nabla log pi(a_t|s_t)] EE_pi [r_(t')] = 0 $

We can therefore rewrite the $nabla J(pi)$ as

$ nabla J(pi) = EE_pi [sum_(t = 1)^H (nabla log pi(a_t|s_t) sum_(t' = t)^H gamma^(t' - 1) r_(t'))] $

=== PG and Value-Based Method

So far we have

$ nabla J(pi) = EE_pi [sum_(t = 1)^H (nabla log pi(a_t|s_t) sum_(t' = t)^H gamma^(t' - 1) r_(t'))] . $

add a condition on expectation:

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
        , gamma^(t - 1) Q_pi (s_t, a_t)
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

=== Blend PG and Value-Based Methods

Instead of using MC estimate $sum_(t' = t)^H gamma^(t' - 1) r_t$ for
$Q^pi (s_t, a_t)$, use an approximate value-function
$hat(Q)_(s_t, a_t)$, often trained by TD, e.g. expected SARSA:

$ Q(S_t , A_t) <- Q(S_t , A_t) + alpha [R_(t + 1) + gamma EE_pi [Q(S_(t + 1) , A_(t + 1)) | S_(t + 1)] - Q(S_t , A_t)] $

==== Actor-critic

The parametrized *policy* is called the *actor*, and the
*value-function* estimate is called the *critic*.

#figure(image("img/Taxonomy-of-model-free-RL-algorithms-by-Schulman-43.png", width: 60%),
  caption: [
    Actor-Critic. #link("https://www.researchgate.net/profile/Nikolas-Wilhelm/publication/344770842/figure/fig1/AS:948651484516356@1603187547778/Taxonomy-of-model-free-RL-algorithms-by-Schulman-43.png")[(credit)]
  ]
)

==== Baseline in PG

for any $f : S -> RR$,

$ nabla J(pi) = frac(1, 1 - gamma) EE_(s~d^pi, a~pi(s)) [(Q^pi(s, a) - f(s)) nabla log pi(a|s)] $

because $f(s)$ and $nabla log pi(s|a)$ are
*independent*.

Choose $f = V^pi(s)$ and

$ nabla J(pi) = frac(1, 1 - gamma) EE_(s~d^pi, a~pi(s)) [A^pi(s, a) nabla log pi(a|s)] $

where $A$ is the advantage function. Bseline don't change the *expectation* of Gradient but lower the *variance*.

== Policy Gradient

Given policy $pi_theta$, optimize
$J (pi_theta) := EE_(s~d_0) [med V^(pi_theta)(s)]$
where $d_0$ is the initial state distribution.

- Use Gradient Ascent ($nabla_theta J(pi_theta)$)
- an unbiased estimate can be obtained from a #strong[single on-policy
    trajectory]
- no need of the knowledge of $P$ and $R$ of the MDP
- Similar to #link(<importance-sampling>)[IS]

Note that when we use $pi$, we mean $pi_theta$ here, and
$nabla$ means $nabla_theta$.

About PG:

- Goal: we want to find good policy.
- Value-based RL is indirect
- PG isn't based on value function
  - It's possible a good policy don't match Bellman Equation.
