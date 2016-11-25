import pykov
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def choose(n, k):
    """Compute n choose k.

    :n: Integral number of objects.
    :k: Integral subset size, 0 <= k <= n.
    :returns: Number of ways to place n objects into a set of size k.

    """
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

def dicepdf(p, n, s):
    """
    Compute the probability of obtaining a roll of `p` with `n` `s`-sided dice.

    This probability is more complicated than it would seem.
    See formula (10) in: http://mathworld.wolfram.com/Dice.html

    :p: Integral desired roll in [ndice, 6 * ndice].
    :n: Integral number of dice.
    :s: Integral number of sides on each die.
    :returns: A probability in [0, 1].

    """
    upper_bound = math.floor((p - n)/s) + 1

    c = sum((-1)**k * choose(n, k) * choose(p - s*k - 1, n - 1)
            for k in range(upper_bound))
    return c / s**n

def draw_chain(chain, layout="circular"):
    """Draw a Markov chain using networkx.

    :chain: A Pykov chain.
    :returns: Nothing.

    """
    layouts = {"circular": nx.circular_layout, "random": nx.random_layout,
                "shell": nx.shell_layout, "spring": nx.spring_layout,
                "spectral": nx.spectral_layout}
    graph = nx.DiGraph(list(chain.keys()))
    pos = layouts[layout](graph)

    nx.draw_networkx(graph, pos)

def make_numpy_monopoly(size, ndice, jail, goto_jail):
    """
    Create a transition matrix for a finite Markov chain representing a
    simplified version of Monopoly. See make_pykov_monopoly() for an
    explanation of our rules.

    We need this method because pykov doesn't allow for easy access to the full
    transition matrix, making it difficult to verify properties such as
    regularity.

    This is a _right_ stochastic matrix, i.e. P[i, j] is the probability of
    moving to state j from state i.
    """
    if jail == goto_jail:
        raise ValueError("`jail` and `goto_jail` must be distinct")

    # Account for three extra jail spaces.
    links = np.zeros((size + 3, size + 3))

    min_advance = ndice
    max_advance = 6 * ndice

    jail_first = size
    jail_second = size + 1
    jail_third = size + 2

    # Probability of rolling one identical number on `ndice`: 6**(-ndice)
    # We have six numbers to choose from, so our probability is 6**(1 - ndice)
    escape_prob = 6**(1 - ndice)

    # Setup the jail rules.
    links[jail_first, jail] = escape_prob
    links[jail_second, jail] = escape_prob
    links[jail_third, jail] = 1

    links[jail_first, jail_second] = 1 - escape_prob
    links[jail_second, jail_third] = 1 - escape_prob

    for space in range(size):
        if space == goto_jail:
            # Immediately go to jail.
            links[space, jail_first] = 1
        else:
            # Advance according to the probability of advancing.
            for advance in range(min_advance, max_advance + 1):
                effect_space = (space + advance) % size
                links[space, effect_space] = dicepdf(advance, ndice, 6)

    return links

def make_pykov_monopoly(size, ndice, jail, goto_jail):
    """Create a Markov chain representing a simplified version of Monopoly.

    Rules:
        - We have `size` spaces, running from 0 to `size` - 1.

        - "Go" is at space 0.

        - At any particular space, we roll `ndice` fair, independent dice, and
          move forward the sum of their rolls.

        - `jail` and `goto_jail` cannot be equal. (Allowing them to be equal
          _and_ obeying the jail rules contradicts the stochastic matrix
          property. We would have P[jail, jail_first] = 1 _and P[jail, jail +
          ndice] > 0.)

        - If we land on `goto_jail`, then we move to the special jail states:

            - The special jail states are `jail_first = size`, `jail_second =
              size + 1`, and `jail_third = size + 2`.

            - After landing on `goto_jail`, we move to `jail_first`.

            - To leave jail, we have two possibilities:

                - Wait three turns. From the third turn spot, we move to `jail`
                  with probability one.

                - Roll the same thing on every dice. There is a 1/6**(ndice -
                  1) probability of this occuring. If this does not happen, we
                  advance to the next turn jail spot.

    :ndice: Number of dice we may roll.
    :size: Number of spaces on the board.
    :jail: Space on which the jail is located.
    :goto_jail: Space on which the "goto jail" is located.
    :returns: Pykov Chain object.

    """
    if jail == goto_jail:
        raise ValueError("`jail` and `goto_jail` must be distinct")

    links = dict()

    min_advance = ndice
    max_advance = 6 * ndice

    jail_first = size
    jail_second = size + 1
    jail_third = size + 2

    # Probability of rolling one identical number on `ndice`: 6**(-ndice)
    # We have six numbers to choose from, so our probability is 6**(1 - ndice)
    escape_prob = 6**(1 - ndice)

    # Setup the jail rules.
    links[(jail_first, jail)] = escape_prob
    links[(jail_second, jail)] = escape_prob
    links[(jail_third, jail)] = 1

    links[(jail_first, jail_second)] = 1 - escape_prob
    links[(jail_second, jail_third)] = 1 - escape_prob

    # Establish the rules for the rest of the board.
    for space in range(size):
        if space == goto_jail:
            # Immediately go to jail.
            links[(space, jail_first)] = 1
        else:
            # Advance according to the probability of advancing.
            for advance in range(min_advance, max_advance + 1):
                effect_space = (space + advance) % size
                links[(space, effect_space)] = dicepdf(advance, ndice, 6)

    return pykov.Chain(links)

def plot_walk_histogram(walks, length, chain, start=None):
    """Plot the histogram of an accumulated walk dictionary.

    :walks: Integral number of walks to take.
    :length: Integral number of steps that each walk should be.
    :chain: Pykov Chain to perform the walk on.
    :start: Start state in `chain` for walk.
    :returns: Nothing.

    """
    walks = walk_accumulate(walks, length, chain, start)
    centered_lefts = [left - .5 for left in walks.keys()]
    plt.bar(centered_lefts, walks.values(), width=1)

def walk_accumulate(walks, length, chain, start=None):
    """Accumulate the results of many walks into a dictionary.

    :walks: Integral number of walks to take.
    :length: Integral number of steps that each walk should be.
    :chain: Pykov Chain to perform the walk on.
    :returns: Dictionary of `{ret: end}` pairs such that `ret[end]` is the
              frequency that the state `end` occured.

    """
    res = {key: 0 for key in chain.states()}

    for walk in range(walks):
        if start:
            stop = chain.walk(length, start)[-1]
        else:
            stop = chain.walk(length)[-1]

        res[stop] += 1

    return res

monopoly_chain = make_pykov_monopoly(40, 2, 30, 10)
monopoly_matrix = make_numpy_monopoly(40, 2, 30, 10)
