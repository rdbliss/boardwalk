import pykov
import networkx as nx
import numpy as np

"""
Important things:
    - Pykov deals with _right_ stochastic matrices, i.e. the rows sum to one.
      "Stochastic Processes" deals with _left_ stochastic matrices, i.e. the
      columns sum to one. Practical differences:
        - For a right stochastic matrix R, we multiply a distribution _row_
          vector v with R on the right, i.e. r' = jR.

        - For a left stochastic matric L, we multiply a distribution _column_
          vector v with L on the left, i.e. c' = Lc.

        - When accessing states in a right stochastic matrix L, we will have
          L[i][j] is the probability of moving to state j from state i. This
          makes a little more sense from a "sequential indexing" perspective,
          if you ask me.

Bugs so far:

    - Chain.communication_classes() tries to insert a set into a dictionary.
      Sets are not hashable.

        c = pykov.Chain({(1, 1): 1})
        c.communication_classes()

    - Calling `Chain.walk()` without a start state throws an UnboundLocalError.
      Sometimes. Not always. I'm not sure why.

    - Passing `0` as a start node causes a random node to be selected.

Complaints:
    There's not a great way to pretty print pykov Chains.

    We can't easily convert them to numpy arrays in general, because we can use
    arbitrary state names. The arbitrary name thing is really nice, though.
"""

# Two different methods: with numpy and pykov.
# Consider the following transition matrix P for a finite Markov chain:
Ppykov = pykov.Chain({(1, 1): .5, (1, 2): .5, (2, 3): 1, (3, 1): 1})
Pnp = np.r_["0,2", [.5, .5, 0], [0, 0, 1], [1, 0, 0]].T

# Note that P^4 > 0, so P is regular. Thus, P is irreducible and aperiodic
# ((strongly) ergodic, in the language of pykov). Since P is finite, our chain
# has a unique limiting distribution.

print("pykov:", Ppykov.steady())

# Since P is regular, the Perron-Frobenius Theorems guarantee that 1 is its
# dominant eigenvalue and that the corresponding eigenvector of 1 is strictly
# positive. Normed so that its sum is one, this eigenvector is our limiting
# distribution. 

eigenvals, eigenvectors = np.linalg.eig(Pnp)

# Since 1 is the dominant eigenvalue, 1 is also the largest eigenvalue.
dominant = np.argmax(eigenvals)
steady = eigenvectors[:,dominant]
steady = steady / sum(steady) # Normalize to a probability distribution.

print("numpy:", steady)

# Run a large number of walks from random states and see what the rough
# distribution is.
walks = 10 ** 4
steps = 10 ** 3
ends = {1: 0, 2: 0, 3: 0}
for k in range(walks):
    if k % 100 == 0:
        print("walk:", k)
    start = np.random.choice([1, 2, 3])
    stop = Ppykov.walk(steps, start)[-1]
    ends[stop] += 1

print("approximate steady state:")
for k, v in ends.items():
    print("{}: {}".format(k, v / walks))

# Conclusions:
#   Numpy is more gratifying to write out, but Pykov gets things done.
