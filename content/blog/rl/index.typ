#import "../index.typ": template, tufted
#show: template.with(title: "Fundamentals of Reinforcement Learning")
#import "@preview/mmdr:0.2.1": mermaid

== example: Shortest Path
<example-shortest-path>
#figure(
  image("img/reinforcement-learning-lecture-1.png", width: 100%),
  caption: [Shortest Path],
)

- nodes: states
- edges: actions

Greedy is not optimal.

#strong[Bellman Equation] (Dynamic Programing):

$ V^* (d) = min { 3 + V^* (g) , 2 + V^* (f) } $

== Stochastic Shortest Path
<stochastic-shortest-path>
Markov Decision Process (MDP)

#figure(
  image("img/reinforcement-learning-lecture-1-1.png", width: 80%),
  caption: [Stochastic Shortest Path],
)

#strong[Bellman Equation]

$ V^* (c) = min { 4 + 0.7 times V^* (d) + 0.3 times V^* (e) , thin 2 + V^* (e) } $

optimal policy : $pi^*$

== Model-based RL
<model-based-rl>
The states are unknown. Learn by #strong[trial-and-error]

#figure(
  image("img/reinforcement-learning-lecture-1-2.png", width: 80%),
  caption: [a trajectory: $s_0->c->e->F->G$],
)

Need to recover the graph by collecting multiple #strong[trajectories].

Use imperial frequency to find probabilities.

Assume states & actions are visited uniformly.

=== exploration problem
<exploration-problem>
Random exploration can be inefficient:

#figure(
  image("img/reinforcement-learning-lecture-1-5.png", height: 30%),
  caption: [example: video game],
)

== example: video game
<example-video-game>
Objective: maximize the reward

$
  bb(E) [sum_(t = 1)^oo r_t divides pi] #h(0em) upright("or") #h(0em) bb(E) [sum_(t = 1)^oo gamma^(t - 1) r_t divides pi]
$

Problem: the graph is too large

#box(image("img/reinforcement-learning-lecture-1-4.png"))=300x

There are states that the RL model have never seen, therefore need
#strong[generalization]

=== Contextual bandits
<contextual-bandits>
- Even if the algorithm is good, if mamke bad actions at beginning, will
  not get good data.
- Keep taking bad actions (e.g.~guessing wrong label on image
  classification), don't know right action.
  - Compared with superivsed learning
- #link("https://en.wikipedia.org/wiki/Multi-armed_bandit")[Multi-armed bandit]

== RL steps
<rl-steps>
For round t = 1, 2, …,

- For time step h=1, 2, …, H, the learner
  - Observes $x_h^((t))$
  - Chooses $a_h^((t))$
  - Receives
    $r_h^((t)) tilde.op R (x_h^((t)) , a_h^((t)))$
  - Next $x_(h + 1)^((t))$ is generated as a function of
    $x_h^((t))$ and $a_h^((t))$ (or sometimes, all previous x's
    and a's within round t)

= The Learning Setting
<the-learning-setting>
== planning and learning
<planning-and-learning>
Planning:

- given MDP model, how to compute optimal policy
- The MDP model is known

Learning:

- MDP model is unknown
- collect data from the MDP: $(s , a , r , s')$ .
- Data is limited. e.g., adaptive medical treatment, dialog systems
- Go, chess, …
- Learning can be useful #strong[even if the final goal is planning]
  - especially when $\| S \|$ is large and/or only blackbox simulator
  - e.g., AlphaGo, video game playing, simulated robotics

== Monte-Carlo policy evaluation
<monte-carlo-policy-evaluation>
Given $pi$ , estimate
$J (pi) := bb(E)_(s tilde.op d_0) [V^pi (s)]$ ( $d_0$ is initial
state distribution) is the #emph[actual] expectation of reward.

Monte-Carlo outputs some scalar $v$ \; accuracy measured by
$\| v - J (pi) \|$ . (by sampling different trajectories):

Data: trajectories starting from $s_1 tilde.op d_0$ using $pi$ (i.e.,
$a_t = pi (s_t)$ ).

$ {(s_1^((i)) , a_1^((i)) , r_1^((i)) , s_2^((i)) , dots.h , s_H^((i)) , a_H^((i)) , r_H^((i)))}_(i = 1)^n $

#quote(block: true)[
  this is called on-policy: evaluating a policy with data collected from
  the exactly same policy.

  Othwise, it is off-policy.
]

Estimator:

$ 1 / n sum_(i = 1)^n sum_(t = 1)^H gamma^(t - 1) r_t^((i)) $

#quote(block: true)[
  Guarantee: w.p. at least
  $1 - delta , \| v - J (pi) \| lt.eq frac(R_max, 1 - gamma) sqrt(frac(1, 2 n) ln 2 / delta)$
  (larger n, higher accuracy)

  It is #strong[independent] to the size of state space
]

=== Comment on Monte-Carlo
<comment-on-monte-carlo>
Monte-Carlo is a Zeroth-order (ZO) optimization method, which is not
efficient.

- #strong[first order]: gradient / first derivative (in DL/ML,
  #strong[SDG])
- #strong[second order]: Hessian matrix / second derivative

== Model-based RL with a sampling oracle (Certainty Equivalence)
<model-based-rl-with-a-sampling-oracle-certainty-equivalence>
#quote(block: true)[
  Assuming the reward / probability is determined (constant) via sampling.
 
]

Assume we can sample $r tilde.op R (s , a)$ and
$s' tilde.op P (s , a)$ for any $(s , a)$

Collect $n$ samples per
\$(s, a):\\\\{\\left(r\_i, s\_i^{\\prime}\\right)\\\\}\_{i=1}^n\$ .
Total sample size $n \| S times A \|$

Estimate an empirical MDP $hat(M)$ from data

- $hat(R) (s , a) := 1 / n sum_(i = 1)^n r_i , quad hat(P) (s' divides s , a) := 1 / n sum_(i = 1)^n bb(I) [s'_i = s']$
- i.e., treat the empirical frequencies of states appearing in
  ${ s'_i }_(i = 1)^n$ as the true distribution.

Plan in the estimated model and return the optimal policy

transition tuples: $(s_i , a_i , r_i , s_(i + 1))$ . Use
$s_i , a_i$ to identify current state and action, use $r_i$ for reward
and $s_(i + 1)$ for transition.

extract transition tuples from trajectories.

=== finding policy on estimated environment
<finding-policy-on-estimated-environment>
#strong[true] environment: $M = (S , A , P , R , gamma)$

#strong[estimated] environment:
$hat(M) = (S , A , hat(P) , hat(R) , gamma)$

- notation: $pi_(hat(M)) , thin V_(hat(M)) , thin dots.h$

performance measurement:

- in the #strong[true] environment, use
  $parallel V^* - V^(pi_f) parallel$ where $f approx Q^*$
- in #strong[estimated] environment, use
  $parallel V_M^* - V_M^(pi_(hat(M))^*) parallel$ ,
  i.e.~measure the optimal policy of estimated environment in the real
  environment.

== Model-based RL with a sampling oracle (Certainty Equivalence) #emph[Cont'd]
<model-based-rl-with-a-sampling-oracle-certainty-equivalence-contd>
To find $Q_(hat(M))^*$ with empirical $hat(R)$ and $hat(P)$ :

$ f_0 in bb(R)^(S A) , quad f_k in hat(cal(T)) f_(k - 1) . $

where

\$\$
\$\$

is call the #strong[Empirical Bellman Update].

=== Computational Complexity
<computational-complexity>
==== Value Interation
<value-interation>
For original
#link("reinforcement-learning-lecture-6/#value-interation-algorithm-vi")[value iteration],
the Computational Complexity is

$ \| S \| times \| A \| times \| S \| $

\$| S| | A| $f o r e a c h$ f(s,a) $a n d$ | S |\$ for expectation.

==== Empirical Bellman Update
<empirical-bellman-update>
For Empirical Bellman Update, the Computational Complexity is

$ \| S \| times \| A \| times n $

Empirical sampling for $n$ times.

== the Value Prediction Problem
<the-value-prediction-problem>
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
  $H = upright(O) (1 \/ (1 - gamma))$ ) such that omitting the
  future rewards has minimal impact ("small truncation error")
- Let $hat(V)^pi (s)$ (will just write $V (s)$ ) be the average
  discounted return

#strong[Online Monte-Carlo]

- For $i = 1 , 2 , dots.h$ as the index of trajectories
  - Draw a starting state $s_i$ from the exploratory initial
    distribution, roll out a trajectory using $pi$ from $s_i$ , and let
    $G_i$ be the (random) discounted return
  - Let $n (s_i)$ be the number of times $s_i$ has appeared as an
    initial state. If $n (s_i) = 1$ (first time seeing this state), let
    $V (s_i) arrow.l G_i$ (where
    $G_t = sum_(t' = t)^(t + H) gamma^(t' - t) r_(t')$ )
  - Otherwise,
    $V (s_i) arrow.l frac(n (s_i) - 1, n (s_i)) V (s_i) + frac(1, n (s_i)) G_i$

No need to store the trajectory.

More generally,

$ V (s_i) arrow.l (1 - alpha) V (s_i) + alpha G_i $

or

$ V (s_i) arrow.l V (s_i) + alpha (G_i - V (s_i)) $

where $alpha$ is known as learning rate, and $G_i$ as the target.

#quote(block: true)[
  It can be interpreted as stochastic gradient descent. If we have i.i.d.
  real random variables $v_1 , v_2 , dots.h , v_n$ , the average is the
  solution of the least-square optimization problem:

  $ min_v frac(1, 2 n) sum_(i = 1)^n (v - v_i)^2 $
]

=== Every-visit Monte-Carlo
<every-visit-monte-carlo>
Suppose we Have a continuing task. What/if we cannot set the starting
state arbitrarily?

i.e.~we have a single #strong[long] trajectory with length $N$

$ s_1 , a_1 , r_1 , s_2 , a_2 , r_2 , s_3 , a_3 , r_3 , dots.h $

- we can truncate $N \/ H$ truncations with length
  $H = O (1 \/ (1 - gamma))$ from the long trajectory.
- we can shift the $H$ -length window by 1 each time and get
  $N - H + 1 approx N$ truncations.

This "walk" through the state space should have non-zero probability on
each state, i.e.~do not starve every states.

What if a state occures multiple times on a trajectory?

- approach 1: only the first occurance is used
- approach 2: all the occurances are used

== Alternative Approach: TD(0)
<alternative-approach-td0>
Again, suppose we have a single long trajectory
$s_1 , a_1 , r_1 , s_2 , a_2 , r_2$ ,
$s_3 , a_3 , r_3 , s_4 , dots.h$ in a continuing task

TD(0): for
$t = 1 , 2 , dots.h , V (s_t) arrow.l V (s_t) + alpha (r_t + gamma V (s_(t + 1)) - V (s_t))$

/ TD: #block[
    temporal difference
  ]

/ TD\_error: #block[
    $r_t + gamma V (s_(t + 1)) - V (s_t)$
  ]

Same as Monte-Carlo update rule, excepts that the "target" is
$r_t + gamma V (s_(t + 1))$ , which is similar to the
#link("reinforcement-learning-lecture-11/#model-based-rl-with-a-sampling-oracle-certainty-equivalence-contd")[empirical Bellman update].

Recall that in
#link("reinforcement-learning-lecture-11/#monte-carlo-value-prediction")[Monte-Carlo],
the "target" is $G_t = sum_(t' = t)^(t + H) gamma^(t' - t) r_(t')$ and
is #strong[independent] to the current value function. While in TD(0),
the target $r_t + gamma V (s_(t + 1))$ is dependent to the current value
function $V$ . i.e.

Compared to value iteration:

$ V_(k + 1) (s) := bb(E)_(r , s' \| s , pi) [r + gamma V_k (s')] $

and the equation above is

$ approx 1 / n sum_(i = 1)^n (r_i + r V_k (s'_i)) $

which is an approximate Value Iteration process, and notice that the
whole iteraton through $i = 1 , dots.h.c , n$ is only 1 iteration (a
$V_k$ ), so an outside loop is needed if we want to $V$ approximates
real $V^pi$ .

=== Understanding TD(0)
<understanding-td0>
The "approximate" Value Iteration process above is similar to TD(0) but
slightly different: it uses a value function $V$ (which stays constant
during updates) to update $V'$ which is another function. After long
enough, we have $V' = cal(T)^pi V$ and do $V arrow.l V'$ , then repeat
the process. Finally converges to $V^pi$ .

But in TD(0), we uses $V$ to update itself. The difference is
"synchronous" vs "asynchronous".

#quote(block: true)[
  TD(0) is less stable
]

== TD( $lambda$ ): Unifying TD(0) and MC
<td-lambda-unifying-td0-and-mc>
- 1-step bootstrap (TD(0)): $r_1 + gamma V (s_(i + 1))$
- 2-step bootstrap: $r_1 + gamma r_(i + 1) + gamma^2 V (s_(i + 2))$
- 3-step bootstrap:
  $r_1 + gamma r_(i + 1) + gamma^2 r_(i + 2) + gamma^3 V (s_(i + 3))$
- …
- $oo$ -step bootstrap:
  $r_1 + gamma r_(i + 1) + gamma^2 r_(i + 2) + gamma^3 r_(i + 3) + dots.h.c$
  is Monte-Carlo.

=== Proof of TD( $lambda$ )'s correctness
<proof-of-td-lambda-s-correctness>
E.g. in 2-step bootstrap,

With
#link("https://en.wikipedia.org/wiki/Law_of_total_expectation")[Law of total expectation],

$
  & bb(E) \[ r_1 + gamma r_(t + 1) + gamma^2 V (s_(t + 2)) \| s_t \]\
  = & bb(E) \[ r_t + gamma (r_(t + 1) + gamma V (s_(t r))) \| s_t \]\
  = & bb(E) \[ r_t \] + gamma bb(E)_(s_(t + 1) \| s_t) #scale(x: 120%, y: 120%)[\[] bb(E) \[ (r_(t + 1) + gamma V (s_(t r))) \| s_t , s_(t + 1) \] #scale(x: 120%, y: 120%)[\]]\
  = & bb(E) \[ r_t + gamma (cal(T)^pi) (s_(t + 1)) \| s_t \]\
  = & ((cal(T)^pi)^2 V) (s)
$

=== TD( $lambda$ )
<td-lambda>
For n-step bootstrap, give a $(1 - lambda) lambda^n$ weight.

- $lambda = 0$ : Only n=1 gives the full weight. TD(0).
- $lambda arrow.r 1$ : (almost) Monte-Carlo.

==== forward view and backward view
<forward-view-and-backward-view>
Forward view

$
  (1 - lambda) dot.op (r_1 + gamma V (s_2) - V (s_1))\
  (1 - lambda) lambda dot.op (r_1 + gamma r_2 + gamma^2 V (s_3) - V (s_1))\
  (1 - lambda) lambda^2 dot.op (r_1 + gamma r_2 + gamma^2 r_3 + gamma^3 V (s_4) - V (s_1))\
  dots.h.c
$

, and so on.

Backward view

$
  1 dot.op (r_1 + gamma V (s_2) - V (s_1))\
  lambda gamma dot.op (r_2 + gamma V (s_3) - V (s_2))\
  lambda^2 gamma^2 dot.op (r_3 + gamma V (s_4) - V (s_3))\
  dots.h.c
$

== Value Prediction with Function Approximation
<value-prediction-with-function-approximation>
/ tabular representation vs.~function approximation: #block[
    function approximation can handle infinite state space (can't enumerate
    through all states).
  ]

/ linear function approximation: #block[
    design features $phi.alt (s) in bb(R)^d$ ("featurizing states"), and
    approximate
    $upright(V)^pi (upright(s)) approx theta^top phi.alt (upright(s)) + b$
    , where $theta$ should be fixed among features (in the following parts,
    $b$ is #strong[ignored] because it can be reached by appending a $1$ to
    the feature vector). \> tabular value function can be interpreted as
    feature vector $in bb(R)^S$ : \>
    $\[ 0 , dots.h.c , 0 , 1 , 0 , dots.h.c , 0 \]$ where the position
    of the $1$ indicates the state.
  ]

=== Example: Tetris Game
<example-tetris-game>
#figure(image("img/reinforcement-learning-lecture-14-1.png", width: 60%), caption: [
  Tetris Game
])

The state space is exponentially large: each block be occupied / not
occupied.

Featurize: \# of blocks on each column. In the example, the feature is
\$ (4;4;5;4;3;3;3;)\$

=== Monte-Carlo Vaule Prediction
<monte-carlo-vaule-prediction>
\$\$
V^\\pi(s)=\\mathbb{E}\[G \\mid s\]=\\argmin\_{f:S\\to \\mathbb R} \\mathbb{E}\\left\[(f(s)-G)^2\\right\]
\$\$

Is a regression problem.

#quote(block: true)[
  Why the expectation is the argmin? See
  #link("./reinforcement-learning-homework-0/#notes")[here] {: .prompt-tip
  }
]

The same idea applies to non-linear value function approximation More
generally & abstractly, think of function approximation as searching
over a restricted #strong[function space], which is a set whose members
are functions that map states to real values.

E.g. a function space of linear value function approximation:

$
  cal(F) = {upright(V)_theta : theta in bb(R)^(upright(d))} upright(", where ") upright(V)_theta (upright(s)) = theta^top phi.alt (upright(s))
$

- typically only a small subset of all possible functions
- Using "all possible functions" = tabular!
- Equivalently, tabular MC value prediction can be recovered by choosing
  $phi.alt$ as the identity features
  \$\\phi(\\mathrm{s})=\\left\\\\{\\mathbb{I} \\left\[\\mathrm{\~s}=\\mathrm{s}^{\\prime}\\right\]\\right\\\\}\_{\\mathrm{s}^{\\prime} \\in \\mathrm{S}}\$

Find the function:

$ min_(V_theta in cal(F)) 1 / n sum_(i = 1)^n (V_theta (s_i) - G_i)^2 $

SGD: uniformly sample $i$ and

$
  theta arrow.l theta - alpha dot.op (upright(V)_theta (upright(s)_(upright(i))) - upright(G)_(upright(i))) dot.op nabla upright(V)_theta (upright(s)_(upright(i)))
$

=== Interprete Td(0) with Linear Approximation
<interprete-td0-with-linear-approximation>
TD(0) iteration is equivalent to

$ theta arrow.l theta + alpha (G_t - phi.alt (s_t)^top theta) phi.alt (s_t) . $

Here $theta$ is the tabular value function and $phi.alt$ is
$\[ 0 , dots.h.c , 0 , 1 , 0 , dots.h.c , 0 \]$ , as mentioned
#link(<value-prediction-with-function-approximation>)[here].

=== TD(0) with Linear Approximation
<td0-with-linear-approximation>
In TD(0), we do

$ V (s_t) arrow.l V (s_t) + alpha (r_t + gamma V (s_(t + 1)) - V (s_t)) , $

which, with all steps on $t$ , gets

$ V_(k + 1) arrow.l cal(T)^pi V_k . $

i.e.

$ V_(k + 1) (s) = bb(E)_pi [r + gamma V_k (s') divides s] $

Similar to Linear Approximation, #strong[rewriting expectation with a
  regression problem],

\$\$
\$\$

And the SGD steps should be

$ theta arrow.l theta + alpha (V_theta (s_t) - r_t - gamma V_k (s_(t + 1))) nabla V_theta (s_t) $

Recall the Bellman Equation:

$
  (T^pi f) (s , a) & = R (s , a) + gamma bb(E)_(s' tilde.op P (k , a)) [f (s' , pi)] \
                   & = bb(E) [r + gamma dot.op f (s' , pi) divides s , a] .
$

with empirically equals to:

$ 1 / n sum_(i = 1)^n (r_i + gamma theta_(k_1) (s'_i , pi)) . $

with tuples $(s_t , a_t , r_t , s_(t + 1))$ in the long
trajectory, applying the running average:

$ Q_k (s_t , a_t) arrow.l Q_k (s_t , a_t) + alpha (r_t + gamma Q_(k - 1) (s_(t + 1) , pi) - Q_k (s_t , a_t)) $

== SARSA
<sarsa>
$ Q (s_t , a_t) arrow.l Q (s_t , a_t) + alpha (r_t + gamma Q (s_t + 1 , a_t + 1) - Q (s_t , a_t)) $

Notice that SARSA is not applicable for deterministic policy, because it
requires a non-zero probability distribution over #strong[all]
st0ate-action pairs ( $forall (s , a) in S times A$ ), but the only
possible action for a certain state is determined by the policy.

=== SARSA with $epsilon.alt$ -greedy policy
<sarsa-with-epsilon--greedy-policy>
How are the $s , a$ data pairs picked in SARSA?

At each time step t, with probability $epsilon.alt$ , choose a from the
action space uniformly at random. otherwise,
\$a\_t = \\argmax\_a Q(s\_t, a)\$

#quote(block: true)[
  When sampling s-a-r-s-a tuple along the trajectory, the first action in
  the tuple is actually generated with last version of $Q$ , so we can say
  SARSA is not 100% "on policy".
]

=== Does SARSA converge to optimal policy?
<does-sarsa-converge-to-optimal-policy>
The cliff example (pg 132 of Sutton & Barto)

- Deterministic navigation, high penalty when falling off the clif
- Optimal policy: walk near the cliff
- Unless epsilon is super small, SARSA will avoid the cliff

#figure(
  image("img/reinforcement-learning-lecture-15.png", width: 80%),
  caption: [cliff example],
)

The optimal path is along the side of the cliff, but on this path, the
$epsilon.alt$ -greedy SARSA will often see large penalty (falling off
the cliff) and therefore, choose the safe path instead.

=== softmax
<softmax>
$epsilon.alt$-greedy can be replaged by softmax: chooses action a with
probability

$ frac(exp (Q (s , a) \/ T), sum_(a') exp (Q (s , a') \/ T)) $

where $T$ is temperature.

== Q-learning
<q-learning>
Update rule:

$ Q (s_t , a_t) arrow.l Q (s_t , a_t) + alpha (r_t + gamma max_(a') Q (s_(t + 1) , a') - Q (s_t , a_t)) $

Q-learning is off-policy: how we take actions have nothing to do with
our current Q-estimate (or its greedy policy). i.e.~Q-learning always
taks $max_(a') Q (s_(t + 1) , a')$ no matter what the real policy is.

e.g.~in the cliff setting, the optimal can always be found, no matter
the choice of $epsilon.alt$ .

=== Exercise: Multi-step Q-learning?
<exercise-multi-step-q-learning>
Does the target
$r_t + gamma r_(t + 1) + gamma^2 max_(a') Q (s_(t + 2) , a')$ work? If
not, why?

No.~Because it leads to

$ Q arrow.l cal(T)^pi cal(T) Q $

#quote(block: true)[
  This resulting $cal(T)^pi cal(T) dots.h.c cal(T)^pi cal(T) Q$ is also a
  optimal policy, but for another MDP, i.e.~on odd steps, follow $pi$ , on
  even steps, free to decide.
]

== Q-learning with experience replay
<q-learning-with-experience-replay>
So far most algorithms we see are “one-pass

- i.e., use each data point once and discard them

- \# updates = \# data points

- Concern 1: We need many updates for optimization to converge Can we
  separate optimization from data collection?

- Concern 2: Need to reuse data if sample size is limited

Sample (with replacement) a tuple randomly from the bag, and apply the
Q-learning update rule.

- \# updates \>\> \# data points

Each time get a new tuple, put in bag, and do updates for several times.

Not applicable for on-policy controls (e.g.~SARSA).

== A Question
<a-question>
$bb(E)_(s , r , s') [(V_theta (s) - r - gamma V_theta (s'))^2]$

We do
$V_theta (s) arrow.l V_theta (s) + alpha (r - gamma V_theta (s') - V_theta (s))$
in TD(0).

What if we minimize the square error between $V_theta (s)$ and its
target,
i.e.~$bb(E)_(s , r , s') [(V_theta (s) - r - gamma V_theta (s'))^2]$
?

No correct. It can be
#link("./reinforcement-learning-homework-0/#proof")[decomposed] as the
sum of 2 parts:

- $bb(E)_s [(V_theta (s) - (cal(T)^pi V_theta) (s))^2]$
  - good. It's L-2 norm Bellman Error.
- $gamma^2 bb(E)_s ["Var"_(s' divides s , pi (s)) [V_theta (s')]]$
  - Not good. It penalize policy with large variance.
  - OK for deterministic environment because the variance is always $0$
    in this case.

=== Solution
<solution>
If we have a simulator, for each $s$ in data, draw another independent
state transition.

Minimize objective

$ bb(E) \[(V_theta (s) - r - gamma V_theta (s'_A)) lr((V_theta (s) - r - gamma V_theta (s'_B)]) $

"Double sampling" and Baird's residual algorithm (Bellman residual
minimization).

== Convergence
<convergence>
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
<example>
#mermaid(
  "
graph LR
    1((1))
    --> 2((2))
    --> 3((3))
    --> 4((4))
    --> 5((5))
    --> 6((6))
    --> 7((7))
    --> 8((8))
    --> 9((9))
    --> 10((10))
    10 ~~~|'reward=Ber(0.5)'| 10
",
)

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

\(0.5 and #strong[0.502] have the same distance to #strong[0.501]\; 0.5
and #strong[0.504] have the same distance to #strong[0.502]\;…)

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

Say the function space is a #strong[plane], than the results of each
iteration (bellman operator) is not on the plane, instead, their
#strong[projections] are picked.

== Importance Sampling
<importance-sampling>
We can only sample $x tilde.op q$ but want to estimate
$bb(E)_(x tilde.op p) f (x)$

Importance Sampling (or importance weighted, or inverse propensity
yscore Ps estimator):

$ frac(p (x), q (x)) f (x) $

Unbiasedness:

$
  bb(E)_(x tilde.op q) [frac(p (x), q (x)) f (x)] = sum_x q (x) (frac(p (x), q (x)) f (x)) = sum_x p (x) f (x) = bb(E)_(x tilde.op p) \[ f (x) \]
$

== Application in contextual bandit (CB)
<application-in-contextual-bandit-cb>
- The data point is a tuple $(x , a , r)$
- The function of interest is $(x , a , r) mapsto r$
- The distribution of interest is
  $x tilde.op d_0 , a tilde.op pi , r tilde.op R (x , a)$
  - Let the joint density be $p (x , a , r)$
- The data distribution is
  $x tilde.op d_0 , a tilde.op pi_b , r tilde.op R (x , a)$
  - Let the joint density be $q (x , a , r)$
- IS estimator:
  $frac(p (x , a , r), q (x , a , r)) dot.op r$
- Write down the densities
  - $p (x , a , r) = d_0 (x) dot.op pi (a divides x) dot.op R (r divides x , a)$
  - $q (x , a , r) = d_0 (x) dot.op pi_b (a divides x) dot.op R (r divides x , a)$
  - To compute importance weight, you don't need knowledge of $mu$ or
    $R$ ! You just need $pi_b$ (or even just $pi_b (a divides x)$ ,
    "proposal prob.")
- Let $rho$ be a shorthand for $pi (a divides x)$ , so estimator is
  $rho dot.op r$
- $pi_b$ need to "cover" $pi$
  - i.e., whenever $pi (a divides x) > 0$ , we need\$ \_b(a x)\>0\$
- A special case:
  - $pi$ is deterministic, and $pi_b$ is uniformly random
    $(pi_b (a divides x) equiv 1 \/ \| A \|)$
  - $frac(bb(I) \[ a = pi (x) \], 1 \/ \| A \|) r$
    - only look at actions that match what $pi$ wants to take and
      discard other data points
    - If match, $rho = \| A \|$ \; mismatch: $rho = 0$
  - On average: only $1 \/ \| A \|$ portion of the data is useful

=== A note about using IS
<a-note-about-using-is>
- We know that shifting rewards do not matter (for planning purposes)
  for fixed-horizon problems
- However, when you apply IS, shifting rewards do impact the variance of
  the estimator
- Special case:
  - deterministic $pi$ , uniformly random $pi_b$ ,
  - reward is deterministic and constant: regardless of $(x , a)$ ,
    reward is always 1 (without any randomness)
  - We know the value of any policy is 1
  - On-policy MC has 0 variance
  - IS still has high variance!
- Where does variance come from?

$
  & 1 / n sum_(i = 1)^n frac(bb(I) [a^((i)) = pi (x^((i)))], 1 \/ \| A \|) dot.op r^((i)) = sum_(i = 1)^n frac(bb(I) [a^((i)) = pi (x^((i)))] dot.op r^((i)), n \/ \| A \|)\
  & = frac(1, n \/ \| A \|) sum_(i : a^((i)) = pi (x^((i)))) r^((i))
$

Because $n \/ \| A \|$ is the #strong[expectaton] of \# of sampling
$a^((i))$ matches $pi$ but not the true \# of matched samples, which
causes variance.

Solution: use true \# of matched samples as denometer,

$ frac(1, \| { i : a^((i)) = pi^((i)) } \|) sum_(a^((i)) = pi^((i))) r_i $

== Multi-step IS in MDPs
<multi-step-is-in-mdps>
- Data: trajectories starting from $s_1 tilde.op d_0$ using $pi_b$
  (i.e., $a_t tilde.op pi_b (s_t)$ ) (for simplicity, assume process
  terminates in $H$ time steps)

$ {(s_1^((i)) , a_1^((i)) , r_1^((i)) , s_2^((i)) , dots.h , s_H^((i)) , a_H^((i)) , r_H^((i)))}_(i = 1)^n $

- Want to estimate $J (pi) := bb(E)_(s tilde.op d_0) [V^pi (s)]$
- Same idea as in bandit: apply IS to the entire trajectory

\===

- define $tau$ as the whole trajectory. The function of interest is
  $tau mapsto sum_(t = 1)^H gamma_partial^(t - 1) r_t$ .
- Let the distribution of trajectory induced by $pi$ be $p (tau)$
- Let the distribution of trajectory induced by $pi_b$ be $q (tau)$
- IS estimator:
  $frac(p (tau), q (tau)) dot.op sum_(t = 1)^H gamma^(t - 1) r_t$

How to compute $p (tau) \/ q (tau)$ ?

- $p (tau) = d_0 (s_1) dot.op pi (a_1 divides s_1) dot.op P (s_2 divides s_1 , a_1) dot.op pi (a_2 divides s_2) dots.h.c P (s_H divides s_(H - 1) , a_(H - 1)) dot.op pi (a_H divides s_H)$
- $q (tau) = d_0 (s_1) dot.op pi_b (a_1 divides s_1) dot.op P (s_2 divides s_1 , a_1) dot.op pi_b (a_2 divides s_2) dots.h.c P (s_H divides s_(H - 1) , a_(H - 1)) dot.op pi_b (a_H divides s_H)$

Here all $P (dot.op \| dot.op)$ terms are cancelled out.

Let $rho_t = frac(pi (d_t divides s_t), pi_b (a_t divides s_t))$ , then
$frac(p (tau), q (tau)) = product_(t = 1)^H rho_t = : rho_(1 : H)$

=== Examine the special case again
<examine-the-special-case-again>
- $pi$ is deterministic, and $pi_b$ is uniformly random
  $(pi_b (a divides x) equiv 1 \/ \| A \|)$
- $rho_t = frac(bb(I) [a_t = pi (s_t)], 1 \/ \| A \|)$
- only look at trajectories where all actions happen to match what $pi$
  wants to take
  - Only if match, $rho = \| A \|^H$ \; mismatch: $rho = 0$
- On average: only $1 \/ \| A \|^H$ portion of the data is useful

== Policy Gradient
<policy-gradient>
Given policy $pi_theta$, optimize
$J (pi_theta) := bb(E)_(s tilde.op d_0) [med V^(pi_theta) (s)]$
where $d_0$ is the initial state distribution.

- Use Gradient Ascent ($nabla_theta J (pi_theta)$)
- an unbiased estimate can be obtained from a #strong[single on-policy
    trajectory]
- no need of the knowledge of $P$ and $R$ of the MDP
- Similar to
  #link("/2024/03/23/reinforcement-learning-lecture-17/#Importance-Sampling")[IS]

{% note info %} Note that when we use $pi$, we mean $pi_theta$ here, and
$nabla$ means $nabla_theta$. {% endnote %}

About PG:

- Goal: we want to find good policy.
- Value-based RL is indirect
- PG isn't based on value function
  - It's possible a good policy don't match Bellman Equation.

=== Example of policy parametrization
<example-of-policy-parametrization>
Linear + softmax:

- Featurize state-action: $phi.alt : S times A arrow.r bb(R)^d$
- Policy (softmax):
  $pi (a \| s) prop e^(theta^top phi.alt (s , a))$

Recall in SARSA, we also used
#link("/2024/03/22/reinforcement-learning-lecture-15/#softmax")[softmax]
with temperature $T$. But in PG, we don't need it. Why?

- In SARSA, softmax policy based on $Q$ function -- $Q$ function cannot
  be arbitrary.
- In PG, $phi.alt (s , a)$ is arbitrary function -- $T$ is
  included.

=== PG Derivation
<pg-derivation>
- The trajectory inducded by $pi$:
  $tau := (s_1 , a_1 , r_1 , dots.h , s_H , a_H , r_H)$ and
  $tau tilde.op pi$.
- Let $R (tau) := sum_(t = 1)^H gamma^(t - 1) r_t$

$ J (pi) := bb(E)_pi [sum_(t = 1)^H gamma^(t - 1) r_t] = bb(E)_(tau tilde.op pi) \[ R (tau) \] $

$
    & nabla J (pi) \
  = & nabla sum_(tau in (S times A)^H) P^pi (tau) R (tau) \
  = & sum_tau (nabla P^pi (tau)) R (tau) \
  = & sum_tau frac(P^pi (tau), P^pi (tau)) nabla P^pi (tau) R (tau) \
  = & sum_tau P^pi (tau) nabla log P^pi (tau) R (tau) \
  = & bb(E)_(tau tilde.op pi) [nabla log P^pi (tau) R (tau)]
$

and

$
  & nabla_theta log P^(pi_theta) (tau) = nabla_theta log (d_0 (s_1) pi (a_1 \| s_1) P (s_2 \| s_1 , a_1) pi (a_2 \| s_2) dots.h.c)\
  & = cancel(nabla log d_0 (s_1)) + nabla log pi (a_1 \| s_1) + cancel(nabla log P (s_2 \| s_1 , a_1)) + nabla log pi (a_2 \| s_2) + dots.h.c\
  & = nabla log pi (a_1 \| s_1) + nabla log pi (a_2 \| s_2) + dots.h.c
$

Note that this form is similar to that discussed in
#link("/2024/03/24/reinforcement-learning-lecture-18/#Multi-step-IS-in-MDPs")[Importance Sampling].

Given that
$pi (a \| s) = frac(e^(theta^top phi.alt (s , a)), sum_(a') e^(theta^top phi.alt (s , a')))$
(denoted by $pi (a \| s)$).

$
  & nabla log pi (a \| s)\
  = & nabla (log (e^(theta^top phi.alt (s , a'))) - log (sum_(a') e^(theta^top phi.alt (s , a'))))\
  = & phi.alt (s , a) - frac(sum_(a') e^(theta^top phi.alt (s , a')) phi.alt (s , a'), sum_(a') e^(theta^top phi.alt (s , a')))\
  = & phi.alt (s , a) - bb(E)_(a' tilde.op pi) \[ phi.alt (s , a') \]
$

Note that the expectation of the quantity above is $0$. i.e.

$
  bb(E)_(a tilde.op pi) #scale(x: 120%, y: 120%)[\[] phi.alt (s , a) - bb(E)_(a' tilde.op pi) \[ phi.alt (s , a') \] #scale(x: 120%, y: 120%)[\]] = 0
$

#strong[Couclusion]:

So far we have

$ nabla J (pi) = bb(E)_pi [(sum_(t = 1)^H gamma^(t - 1) r_t) (sum_(t = 1)^H nabla log pi (a_t \| s_t))] $

With the relation discussed above, we say
$bb(E)_pi \[ nabla log pi (a_t \| s_t) \] = sum_(a_t) nabla pi (a_t \| s_t) = nabla 1 = 0$

So, for $t' < t$, $r_(t')$ is independent to
$nabla log pi (a_t \| s_t)$, we have

$ bb(E)_pi \[ nabla log pi (a_t \| s_t) r_(t') \] = bb(E)_pi \[ nabla log pi (a_t \| s_t) \] bb(E)_pi \[ r_(t') \] = 0 $

We can therefore rewrite the $nabla J (pi)$ as

$ nabla J (pi) = bb(E)_pi [sum_(t = 1)^H (nabla log pi (a_t \| s_t) sum_(t' = t)^H gamma^(t' - 1) r_(t'))] $

=== PG and Value-Based Method
<pg-and-value-based-method>
So far we have

$ nabla J (pi) = bb(E)_pi [sum_(t = 1)^H (nabla log pi (a_t \| s_t) sum_(t' = t)^H gamma^(t' - 1) r_(t'))] . $

add a condition on expectation:

\$\$
\$\$

=== Blend PG and Value-Based Methods
<blend-pg-and-value-based-methods>
Instead of using MC estimate $sum_(t' = t)^H gamma^(t' - 1) r_t$ for
$Q^pi (s_t , a_t)$, use an approximate value-function
$hat(Q)_(s_t , a_t)$, often trained by TD, e.g.~expected SARSA: \$
Q\(S\_t, A\_t) Q\(S\_t, A\_t)+-Q\(S\_t, A\_t)\] \$.

==== Actor-critic
<actor-critic>
The parametrized #strong[policy] is called the #emph[actor], and the
#strong[value-function] estimate is called the #emph[critic].

// #figure(image("https://www.researchgate.net/profile/Nikolas-Wilhelm/publication/344770842/figure/fig1/AS:948651484516356@1603187547778/Taxonomy-of-model-free-RL-algorithms-by-Schulman-43.png"),
//   caption: [
//     Actor-Critic
//   ]
// )

==== Baseline in PG
<baseline-in-pg>
for any $f : S arrow.r bb(R)$,

$
  nabla J (pi) = frac(1, 1 - gamma) bb(E)_(s tilde.op d^pi , a tilde.op pi (s)) [(Q^pi (s , a) - f (s)) nabla log pi (a \| s)]
$

because $f (s)$ and $nabla log pi (s \| a)$ are
#strong[independent].

Choose $f = V^pi (s)$ and

$
  nabla J (pi) = frac(1, 1 - gamma) bb(E)_(s tilde.op d^pi , a tilde.op pi (s)) [A^pi (s , a) nabla log pi (a divides s)]
$

where $A$ is the advantage function. Bseline don't change the
#strong[expectation] of Gradient but lower the #strong[variance].

== Policy Gradient
<policy-gradient-1>
Given policy $pi_theta$, optimize
$J (pi_theta) := bb(E)_(s tilde.op d_0) [med V^(pi_theta) (s)]$
where $d_0$ is the initial state distribution.

- Use Gradient Ascent ($nabla_theta J (pi_theta)$)
- an unbiased estimate can be obtained from a #strong[single on-policy
    trajectory]
- no need of the knowledge of $P$ and $R$ of the MDP
- Similar to
  #link("/2024/03/23/reinforcement-learning-lecture-17/#Importance-Sampling")[IS]

{% note info %} Note that when we use $pi$, we mean $pi_theta$ here, and
$nabla$ means $nabla_theta$. {% endnote %}

About PG:

- Goal: we want to find good policy.
- Value-based RL is indirect
- PG isn't based on value function
  - It's possible a good policy don't match Bellman Equation.
-

== Markov Decision Processes
<markov-decision-processes>
=== Infinite-horizon discounted MDPs
<infinite-horizon-discounted-mdps>
An MDP $M = (S , A , P , R , gamma)$ consists of:

- #strong[State space] $S$ .
- #strong[Action space] $A$ .
- #strong[Transition function] $P$ : $S times A arrow.r Delta (S)$ .
  $Delta (S)$ is the #strong[probability] simplex over $S$ , i.e.,
  all non-negative vectors of length $\| S \|$ that sums up to $1$ .
- #strong[Reward function] $R$ : $S times A arrow.r bb(R)$ .
  (deterministic reward function)
- #strong[Discount factor] $gamma in \[ 0 , 1 \]$

The agent

+ starts in some state $s_1$
+ takes action $a_1$
+ receives reward $r_1 = R (s_1 , a_1)$
+ transitions to $s_2 tilde.op P (s_1 , a_1)$
+ takes action $a_2$
+ … and so on so forth �?the process continues forever.

Objective: (discounted) expected total reward

- Other terms used: return, value, utility, long-term reward, etc

== Example: Gridworld
<example-gridworld>
#box(image("img/reinforcement-learning-lecture-2.png"))=300x

- State: grid x, y
- Action: N, S, E, W
- Dynamics:
  - most cases: move to adjacent grid
  - meet wall or reached goal: keep in the current state
- Reward:
  - $0$ in the goal state
  - $- 1$ everywhere else
- Discount factor $gamma$ : 0.99

== discounting
<discounting>
\$= 1 $a l l o w s s o m e s t r a t e g i e s t o o b t a i n$ -\$
expected return.

For $gamma < 1$ , the total reward is finite.

=== finite horizon vs.~infinite-horizon discounted MDP
<finite-horizon-vs.-infinite-horizon-discounted-mdp>
- For finite-horizon (finite acitons), $gamma$ can be $1$ .
- For infinite-horizon (infinite acitons), $gamma < 1$ .

== Value and policy
<value-and-policy>
Take action that maximize

$ bb(E) [sum_(t = 1)^oo gamma^(t - 1) r_t] $

assume $r_t in \[ 0 , R_max \]$ ,

$ bb(E) [sum_(t = 1)^oo gamma^(t - 1) r_t] in [0 , frac(R_max, 1 - gamma)] . $

A #strong[policy] describes how the agent acts at a state:

$ a_t = pi (s_t) $

define value funtion

$ V^pi (s) = bb(E) [sum_(t = 1)^oo gamma^(t - 1) r_t mid(bar.v) s_1 = s , pi] $

== Bellman Equation
<bellman-equation>
$
  V^pi (s) & = bb(E) [sum_(t = 1)^oo gamma^(t - 1) r_t mid(bar.v) s_1 = s , pi] \
           & = R (s , pi (s)) + gamma ⟨P (dot.op mid(bar.v) s , pi (s)) , V^pi (dot.op)⟩
$

Detailed steps
$
  V^pi (s) & = bb(E) [sum_(t = 1)^oo gamma^(t - 1) r_t mid(bar.v) s_1 = s , pi]\
  & = bb(E) [r_1 + sum_(t = 2)^oo gamma^(t - 1) r_t mid(bar.v) s_1 = s , pi]\
  & = R (s , pi (s)) + sum_(s' in cal(S)) P (s' mid(bar.v) s , pi (s)) bb(E) [gamma sum_(t = 2)^oo gamma^(t - 2) r_t mid(bar.v) s_1 = s , s_2 = s' , pi]\
  & = R (s , pi (s)) + sum_(s' in cal(S)) P (s' mid(bar.v) s , pi (s)) bb(E) [gamma sum_(t = 2)^oo gamma^(t - 2) r_t mid(bar.v) s_2 = s' , pi]\
  & = R (s , pi (s)) + gamma sum_(s' in cal(S)) P (s' mid(bar.v) s , pi (s)) bb(E) [sum_(t = 1)^oo gamma^(t - 1) r_t mid(bar.v) s_1 = s' , pi]\
  & = R (s , pi (s)) + gamma sum_(s' in cal(S)) P (s' mid(bar.v) s , pi (s)) V^pi (s')\
  & = R (s , pi (s)) + gamma ⟨P (dot.op mid(bar.v) s , pi (s)) , V^pi (dot.op)⟩
$
where $chevron.l dot.op , dot.op chevron.r$ is Dot Product.

=== Matrix form
<matrix-form>
- $V^pi$ as the $\| S \| times 1$ vector $\[ V^pi (s) \]_(s in S)$
- $R^pi$ as the vector $\[ R (s , pi (s)) \]_(s in S)$
- $P^pi$ as the matrix
  $\[ P (s' \| s , pi (s)) \]_(s in S , s' in S)$

$
  V^pi = R^pi + gamma P^pi V^pi\
  (I - gamma P^pi) V^pi = R^pi\
  V^pi = (I - gamma P^pi)^(- 1) R^pi\
$

Claim: $(I - gamma P)$ is invertible.

Proof. It suffies to prove

$ forall x eq.not arrow(0) in bb(R)^S , #h(0em) (I - gamma P^pi) x eq.not arrow(0) $

then

$
        & parallel (I - gamma P^pi) x parallel_oo \
      = & parallel x - gamma P^pi x parallel_oo \
  gt.eq & parallel x parallel_oo - gamma parallel P^pi x parallel_oo \
  gt.eq & parallel x parallel_oo - gamma parallel x parallel_oo \
      = & (1 - gamma) parallel x parallel_oo \
  gt.eq & parallel x parallel_oo \
      > & 0 square.filled.medium
$

== Generalize to stochastic policies
<generalize-to-stochastic-policies>
$
  V^pi (s) = bb(E)_(a tilde.op pi (dot.op divides s) , s' tilde.op P (dot.op divides s , a)) [R (s , a) + gamma V^pi (s')]
$

=== Matrix form
<matrix-form-1>
$ V^pi = R^pi + gamma P^pi V^pi $

still holds for

$
  & R^pi (s) = bb(E)_(a tilde.op pi (dot.op divides s)) \[ R (s , a) \] \
  & P^pi (s' divides s) = sum_(a in cal(A)) pi (a divides s) P (s' divides s , a)
$

== Optimality
<optimality>
For infinite-horizon discounted MDPs, there always exists a stationary
and deterministic policy that is optimal for all starting states
simultaneously.

Optimal policy $pi^*$ and

$ V^* := V^(pi^*) $

=== Bellman Optimality Equation
<bellman-optimality-equation>
\$\$
V^{\*}(s)=\\max\_{a\\in A}\\left(R(s,a)+\\gamma\\mathbb{E}\_{s^{\\prime}\\thicksim P(s,a)}\\left\[V^{\*}(s^{\\prime})\\right\]\\right)
\$\$

=== Q-functions
<q-functions>
\$\$
\$\$

=== Bellman equation for $Q$
<bellman-equation-for-q>
\$\$
\$\$

=== Define optimal $V$ and $pi$ by $Q$
<define-optimal-v-and-pi-by-q>
$ V^(*) (s) = max_(a in A) Q^(*) (s , a) = Q^(*) (s , pi^(*) (s)) $

\$\$
\\pi^\\star (s) = \\argmax\_{a \\in A} Q^\\star (s, a)
\$\$

== Fixed-horizon MDPs
<fixed-horizon-mdps>
Specified by $(S , A , R , P , H)$ , All trajectories end in
precisely $H$ steps

$
  V_(H + 1)^pi equiv 0\
  V_h^pi (s) = R (s , pi (s)) + bb(E)_(s' tilde.op P (s , a)) \[ V_(h + 1)^pi (s') \]
$

Recall $V^* , Q^* , V^pi , Q^pi$ .

$
  V^* (s) = max_(a in A) (underbrace(R (s , a) + gamma bb(E)_(S' tilde.op P (dot.op divides s , a)) [V^* (s')], Q^* (s , a)))
$

== Bellman Operator
<bellman-operator>
$
  forall f : S times A arrow.r bb(R) ,\
  (cal(T) f) (s , a) = (R (s , a) + gamma bb(E)_(s' tilde.op P (dot.op \| s , a)) \[ max_(a') f (s' , a') \])\
  upright("where") #h(0em) cal(T) : bb(R)^(S A) arrow.r bb(R)^(S A) .
$

then

$
  Q^* = cal(T) Q^*\
  V^* = cal(T) V^*\
$

i.e.~$Q^*$ and $V^*$ are fixpoints of $cal(T)$ .

== Value Interation Algorithm (VI)
<value-interation-algorithm-vi>
$ upright("funtion ") f_0 = arrow(0) in bb(R)^(S A) $

Interation to calculate fix points:

for $k = 1 , 2 , 3 , dots.h.c$ ,

$ f_k = cal(T) f_(k - 1) $

How to do interation

$ forall s , a : #h(0em) f_1 (s , a) arrow.l cal(T) f_0 (s , a) $

- synchronized iteration: use all functions values from old version
  during the update.
- asynchronized iteration

== Convergence of VI
<convergence-of-vi>
lemma: $cal(T)$ is a $gamma$ #strong[\-contraction] under
$parallel dot.op parallel_oo$ where
$parallel dot.op parallel_oo := max_(x in (dot.op)) x$ .

which means

$ parallel cal(T) f - cal(T) f' parallel_oo lt.eq gamma parallel f - f' parallel_oo $

Proof.

$
                           & parallel f_k - Q^* parallel_oo \
                         = & parallel cal(T) f_(k - 1) - Q^* parallel_oo \
                         = & parallel cal(T) f_(k - 1) - cal(T) Q^* parallel_oo \
  lt.eq^(upright("lemma")) & gamma parallel f_(k - 1) - Q^* parallel_oo \
$

$
  arrow.r.double parallel f_k - Q^* parallel_oo lt.eq gamma^k parallel f_(k - 1) - Q^* parallel_oo #h(0em) upright("where") #h(0em) gamma in (0 , 1) square.filled.medium
$

Proof of lemma:

It suffies to prove

$
  & lr(|(cal(T) f - cal(T) f') (s , a)|)\
  = & lr(|(R (s , a) + gamma bb(E)_(s' tilde.op P (dot.op \| s , a)) \[ max_(a') f (s' , a') \]) - (R (s , a) + gamma bb(E)_(s' tilde.op P (dot.op \| s , a)) \[ max_(a') f (s' , a') \])|)\
  = & gamma lr(|bb(E)_(s' tilde.op P (dot.op \| s , a)) \[ max_(a') f (s' , a') - max_(a') f' (s' , a') \]|)
$

\$ $a s s u m e ,$ #emph[{a'} f(s', a') \> ]{a'} f'(s', a') $a n d$ a^:
\_af(s',a)=f(s',a^)\$, then

$
        & gamma lr(|bb(E)_(s' tilde.op P (dot.op \| s , a)) \[ max_(a') f (s' , a') - max_(a') f' (s' , a') \]|) \
      = & gamma lr(|bb(E)_(s' tilde.op P (dot.op \| s , a)) \[ f (s' , a^*) - max_(a') f' (s' , a') \]|) \
  lt.eq & gamma lr(|bb(E)_(s' tilde.op P (dot.op \| s , a)) \[ f (s' , a^*) - f' (s' , a^*) \]|) \
  lt.eq & gamma lr(|f (s' , a^*) - f' (s' , a^*)|) \
  lt.eq & gamma parallel f - f' parallel_oo
$

greedy policy:

\$\$
\\pi^\\star (s) = \\argmax\_{a\\in A} Q^\\star (s, a)
\$\$

sequence of function:

$ f_0 , f_1 , f_2 , dots.h.c arrow.r Q^* $

define

\$\$
\\pi\_{f\_k}^\\star (s) = \\argmax\_{a\\in A} f\_k(s, a)
\$\$

Claim:

$ parallel V^* - V^(pi_f) parallel lt.eq frac(2 parallel f - Q^* \| \|_oo, 1 - gamma) $

define operator $cal(T)$ :

$ (cal(T) f) (s) = max_(a in A) (R (s , a) + gamma E_(s' tilde.op P (dot.op divides s , A)) [f (s')]) $

#quote(block: true)[
  Note: the $cal(T)$ in $cal(T) Q^*$ and $cal(T) V^*$ are
  #strong[not the same].
]

== $V^*$ Iteration
<vstar-iteration>
$
  f_0 = arrow(0)\
  f_k arrow.l cal(T) f_(k - 1)
$

then

$ f_k (s) = max_(upright("all possible ") pi) bb(E) [sum_(t = 1)^k gamma^(t - 1) r_t divides s_1 = s , pi] $

#quote(block: true)[
  This is derived my the definaion of operator $cal(T)$ .
]

Claim:

$ parallel f_k - V^* parallel lt.tilde gamma^k $

step 1: $f_k lt.eq V^*$

step 2:

$
  f_k gt.eq & #box(stroke: black, inset: 3pt, [$ bb(E) [sum_(t = 1)^oo gamma^(t - 1) r_t divides s_1 = s , pi^*] $]) - bb(E) [sum_(t = k + 1)^oo gamma^(t - 1) r_t divides s_1 = s , pi^*]\
  gt.eq & #box(stroke: black, inset: 3pt, [$ V^* $]) - r^k V_max square.filled.medium
$

c

#quote(block: true)[
  this means, once reached goal $pi^*$ , never leave. {: .prompt-tip
  }
]

== example



#mermaid(
  "
  graph TD;
      A([start])
      A -->|+0| B([Japanese])
      A -->|+0| C([Italian])
      B -->|+2| D([Ramen])
      B -->|+2| E([Sushi])
      C -->|+1| F([Steak])
      C -->|+3| G([Pasta])
",
)

Optimal policy is heading `Pasta`.

#quote(block: true)[
  This example is a finite horizon case. To make it infinite horizon
  discount, add a state $T$ :

  #mermaid(
    "
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
",
  )

 
]

To find $V^* (s)$ , update $V$ value from leaf upwards to root
state.

=== policy iteration (example)
<policy-iteration-example>
==== interation \#0
<interation-0>
define initial $pi_0$ :

#mermaid(
  "
graph TD;
    A([start])
    A -.-|+0| B([Japanese])
    A ==>|+0| C([Italian])
    B ==>|+2| D([Ramen])
    B -.-|+2| E([Sushi])
    C ==>|+1| F([Steak])
    C -.-|+3| G([Pasta])
",
)

then the corresponding $Q^(pi_0) (s , a)$ :

#mermaid(
  "
graph TD;
    A([start])
    A -->|+2| B([Japanese])
    A -->|+1| C([Italian])
    B -->|+2| D([Ramen])
    B -->|+2| E([Sushi])
    C -->|+1| F([Steak])
    C -->|+3| G([Pasta])
",
)

==== interation \#1
<interation-1>
\$\$
\\pi\_1(s) := \\argmax\_{a\\in A} Q^{\\pi\_0} (s, a)
\$\$

#mermaid(
  "
graph TD;
    A([start])
    A ==>|+2| B([Japanese])
    A -.-|+1| C([Italian])
    B ==>|+2| D([Ramen])
    B -.-|+2| E([Sushi])
    C -.-|+1| F([Steak])
    C ==>|+3| G([Pasta])
",
)

$Q^(pi_i)$:

#mermaid(
  "
graph TD;
    A([start])
    A -->|+2| B([Japanese])
    A -->|+3| C([Italian])
    B -->|+2| D([Ramen])
    B -->|+2| E([Sushi])
    C -->|+1| F([Steak])
    C -->|+3| G([Pasta])
",
)

==== interation \#2
<interation-2>
\$\$
\\pi\_2(s) := \\argmax\_{a\\in A} Q^{\\pi\_1} (s, a)
\$\$

#mermaid(
  "
graph TD;
    A([start])
    A -.-|+2| B([Japanese])
    A ==>|+3| C([Italian])
    B ==>|+2| D([Ramen])
    B -.-|+2| E([Sushi])
    C -.-|+1| F([Steak])
    C ==>|+3| G([Pasta])
",
)

==== Comment
<comment>
Policy $pi$ was switched to Japanese for once, and switched back to
Italian at the end.

Also, the policy updates upwards.

== Monotone Policy improvement
<monotone-policy-improvement>
$ forall k , forall s : V^(pi_k) gt.eq V^(pi_(k - 1)) $

$ upright(" if ") pi_(k - 1) eq.not pi^* , exists s : v^(pi_k) (s) > V^(pi_(k - 1)) (s) $

$ arrow.r.double upright("#iteration") lt.eq \| A \|^(\| S \|) $

#quote(block: true)[
  Monotone Policy improvement produces exact solutions, while value
  iteration produces approxmitate solutions,
]

Proof of: $Q^(pi_(k + 1)) gt.eq Q^(pi_k)$

lemma 1:

$ Q^(pi_k) = cal(T)^(pi_k) Q^(pi_k) lt.eq cal(T) Q^(pi_k) $

beacuse

$
        & (cal(T)^pi f) (s , a) = R (s , a) + gamma bb(E)_(s' tilde.op p (dot.op divides s , a)) [f (s' , pi)] \
  lt.eq & (cal(T) f) (s , a) = R (s , a) + gamma bb(E)_(s' tilde.op p (dot.op divides s , a)) [max_(a') f (s' , a')]
$

lemma 2:

$ cal(T) Q^(pi_k) = cal(T)^(pi_(k + 1)) Q^(pi_k) $

lemma 3:

$ forall f gt.eq f' , #h(0em) cal(T)^pi f gt.eq cal(T)^pi f' $

with lemma 1,2,3,

$
  Q^(pi_k) & lt.eq #box(stroke: black, inset: 3pt, [$ cal(T)^(pi_(k + 1)) Q^(pi_k) $])\
  arrow.r.double #box(stroke: black, inset: 3pt, [$ cal(T)^(pi_(k + 1)) Q^(pi_k) $]) & lt.eq cal(T)^(pi_(k + 1)) cal(T)^(pi_(k + 1)) Q^(pi_k)\
  & dots.v\
  arrow.r.double Q^(pi_k) & lt.eq (cal(T)^(pi_(k + 1)))^oo Q^(pi_k) = Q^(pi_(k + 1))
$

because $Q^(pi_(k + 1))$ is the fixed point of $cal(T)^(pi_(k + 1))$ .

== recap
<recap>
in policy iteration, appply greedy algo very time.

\#steps are finite.

== another proof
<another-proof>
=== performance-difference lemma (P-D lemma)
<performance-difference-lemma-p-d-lemma>
#quote(block: true)[
  this is a fundamental tool in RL. many deep RL models relies on this
  lemma
]

$forall pi , pi' , s$,

$ V^(pi') (s) - V^pi (s) = frac(1, 1 - gamma) E_(s' tilde.op d_s^(pi')) [Q^pi (s' , pi') - V^pi (s')] $

apply the lemma in the policy iteration steps:

$
  V^(pi_(k + 1)) (s) - V^(pi_k) (s) = frac(1, 1 - gamma) E_(s' tilde.op ?) [Q^(pi_k) (s' , pi_(k + 1)) - V^(pi_k) (s')]
$

and

$ V^(pi_k) (s') = Q^(pi_k) (s' , pi_k) $

and RHS $gt.eq 0$ is trivial. QED

=== Proof of lemma
<proof-of-lemma>
