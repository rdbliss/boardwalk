# Boardwalk

Contained in this repository is code to model the boardgame Monopoly as a
Markov chain. The code relies on the
[Pykov](https://github.com/riccardoscalco/Pykov) library to carry out various
computations related to the Markov chain, and also exploits
[numpy](http://www.numpy.org/) to create a transition matrix for the chain.

## Features

- Create a Monopoly board of arbitrary size and an arbitrary number of dice,
  with arbitrary jail, goto jail, and chance spaces.

- Represent the Monopoly board as a Pykov `Chain` and a numpy transition
  matrix.

- Report steady state information about the board, properly accounting for jail
  spaces.
