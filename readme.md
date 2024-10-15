## Creating environment

First install conda in your machine if not already present. Then run 

    conda create -n aia2 python=3.10 numpy tk
    
in your terminal to create the conda environment `aia2` with required packages. Do not install any other package in this environment as we will run your code with just these packages. You need to create the conda environment just once. 

Second time onwards, just activate the environment using the command
    
    conda activate aia2

and run the game command.

## Interacting with the Simulator

The following command will initiate a game of ExtendedHavannah between **agent_1** and **agent_2** with the initial state given at the **test_case_path**. The script can be invoked with several options described subsequently. The command should display the GUI showing the game. This total time available each player for the game is controlled by parameter `time`.

```python
python3 game.py {agent_1} {agent_2} --start_file {test_case_path} --time {total_time_in_seconds}
```

To create a random board specify board dimension (the number of cells in each edge) `dim` and run the following command:

```python
python3 game.py {agent_1} {agent_2} --dim 10 --time {total_time_in_seconds}
```

The following command will allow a **human agent** to play against the **AI agent.** The human agent (you) can click on a cell to play that move or type in a move as *"\<row num\> ,\<col num\>"*, where row and column numbers are 0 indexed:

```python
python3 game.py ai human --dim {board dimension} --time {total_time_in_seconds}
```

A simple **random agent** is provided in the starter code. The random agent simply picks its moves uniformly at random among the available ones. A game between an AI agent and the random agent can be initiated as follows:

```python
python3 game.py ai random --dim 5 --time 20
```

You might also see your bot play against itself:

```python
python3 game.py ai ai --start_file havannah/initial_states/size4.txt --time 20
```

Moves that are played on a blocked or out of window cell are considered **invalid moves**. If a player attempts to play an invalid move, the game simulator does not change the game state (i.e., the attempted move is skipped) and the turn switches to the next player. Note, that if at any point, if a player exhausts its total game time, it straight away loses and its opponent wins the game.


## Agent Overview

Havannah Overview:
Havannah is a two-player connection game played on a hexagonal board. The objective is to form one of three specific structures to win:
1. Fork: Connecting three edges of the board.
2. Ring: Forming a loop that surrounds at least one hexagon.
3. Bridge: Connecting two corners of the board.
The game does not involve capturing, and the first player can place their stone on any cell. 
The game is known to have a first-move advantage, where the first player may gain a strategic edge. 
Notably, Havannah has been solved for boards with a dimension of less than 4.


Branching Factor and Depth Discussion:
Branching factor and depth are equal to the no of unfilled positions at the start i.e no of hexagons in the board
For Dimension = 4:
	Branching Factor = 37, depth = 37, Total Number of board States of the order 3**(37)
For Dimension = 6:
	Branching Factor = 91, depth = 91, Total Number of board States of the order 3**(91)

Mate Checks in 1, 2, and 3 Moves:
To enhance efficiency, the AI agent includes checks for mate in 1, mate in 2, and mate in 3 based on the number of unfilled positions on the board. 
These checks are necessary as always checking for deeper mates (like mate in 2 or 3) would significantly increase the computation time, especially on larger boards. 
The algorithm strategically determines when it is feasible to check for these scenarios, balancing between speed and thoroughness. 
If fewer unfilled positions remain, deeper checks like mate in 2 or mate in 3 are more feasible.

Monte Carlo Tree Search (MCTS) Implementation:
MCTS is the core algorithm behind the AI's decision-making process. Here's a breakdown of how it works:
1. Tree Expansion: The agent starts from a root node representing the current game state and expands it by simulating possible moves. New nodes are added to the tree for each unexplored move.
2. Simulation (Rollout): For each node, the agent simulates random games from that position until the game ends. The outcome (win or loss) is recorded.
3. Backpropagation: The results from the rollout are backpropagated through the tree, updating the values of all parent nodes based on the result of the simulation.
4. Selection of Best Child: Using the exploration-exploitation tradeoff (via Upper Confidence Bounds standard formula), the best move is chosen based on the node's win rate and the number of times it was visited.
5. Final Decision: The agent picks the move associated with the node that has the highest win rate after the simulations.

Strategy for Dimension 4:
For a Havannah board of dimension 4, the AI uses an opening strategy combined with a check for mate to avoid falling into easily predictable traps.
The opening phase includes strategies based on known optimal sequences of moves, allowing the AI to start with a strong position. 
After the opening, MCTS is used to handle the midgame and endgame.

The AI checks for mate in 2 and mate in 3 when the number of filled positions reaches a certain threshold to prevent unnecessary calculations early on.

Strategy for Dimension 6:
On a dimension 6 board, the AI follows a more complex opening strategy designed to quickly build a short fork or other winning structure. 
The moves for this opening are encoded to allow for fast deployment of a strategic advantage.

After the opening moves, the agent switches to MCTS for the rest of the game. 
This approach balances speed and strategic depth, allowing the AI to efficiently handle the larger search space of a dimension 6 board.

Time Management Strategy:
The AI agent manages its computation time dynamically in a parabolic fashion:
- Opening moves use less time, as they rely on precomputed strategies.
- Midgame uses the most time, where MCTS simulations are more heavily relied upon to explore the vast decision tree.
- Endgame time usage decreases as the number of remaining moves shrinks and it becomes easier to determine the outcome.

Testing and Final Submission:
The AI agent was rigorously tested against weaker agents implemented in-house. After extensive testing and optimization, 
the best-performing version of the agent was submitted. The final agent combines the strength of MCTS with carefully crafted opening 
strategies and dynamic time management, making it a formidable competitor in Havannah.
