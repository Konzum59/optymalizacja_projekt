# Solving flow-shop scheduling with ant colony algorithm

## Flow-shop scheduling

Flow-shop scheduling problem is a variant of a optimal job scheduling.\
In the general job scheduling variant we are given $n$ jobs $J_1, J_2, \dots, J_n$ of varying **processing times**, which need to be scheduled on $m$ machines $M_1, M_2, \dots, M_m$ with varying **processing power**.\
The goal of the problem is to find a optimal **makespan** ($C_{max}$) - minimizing length of the schedule.\
The general flow-shop variant is specific in a way that each job requires exactly $m$ operations and $i$-th operation of the job must be executed on a $i$-th machine. No machine can perform more than one operations simultaneously.
In the general variant the sequence of jobs on individual machine can vary, i.e. having jobs $A, B, C$, machine $1$ can process them in order $A \rightarrow B \rightarrow C$ and machine $2$ can process them in different order $A \rightarrow C \rightarrow B$. This makes the search space significantly larger ($(n!)^m$ possible schedules)\

In our case we solve **permutation** flow-shop scheduling problem which adds additional constraint such that the order of processed jobs must be the same on each machine. In such case the search space reduces to $n!$ possible schedules.\
In our extended variant we introduced additional constraint - some jobs must be executed before some other jobs (e.g. job 4 must be processed before job 6).

Most of flow-shop scheduling variants are **NP-hard**.

## Ant colony algorithm

One of the nature-inspired algorithms.\
It is based on how ants behave when searching for the shortest path to food.\
It does not rely on rigid rules, but on improving the solution through simple behaviors and "cooperation".

**Course of the solution improvement process:**

1. Ants move randomly and leave a pheromone trail.
2. More pheromones accumulate on shorter paths.
3. More and more ants travel along paths with a stronger pheromone trail.
4. After a sufficiently long time, the shortest paths begin to dominate.

**Key Components of the algorithm:**

1. _Ants_ – agents searching the problem space and constructing a solution step by step.
2. _Pheromone trails_ – numerical values storing previous experiences; they indicate which paths are more promising.
3. _Heuristic information_ – problem-specific knowledge that helps ants make better decisions (distance, cost, time).
4. _Probability-based movement_ – ants choose paths stochastically (based on probability), aided by pheromone trail values and heuristic values, rather than relying on rigid rules.
5. _Pheromone evaporation_ – pheromone levels gradually reduce over time, which prevents premature convergence and "encourages" further exploration.

**Algorithm workflow:**

1. _Initialization_ – setting initial parameters and assigning small pheromone values to each path.
2. _Path construction by ants_ – each ant creates a solution by searching the problem space and making decisions based on pheromone levels and heuristic information.
3. _Solution evaluation_ – the quality of the generated solution is evaluated based on fitness or a cost function.
4. _Pheromone level update_ – a good solution increases pheromone levels on the path, while a bad one decreases them.
5. _Iterating until an optimum is found_ – the process is repeated until the best path is found or a stopping criterion is met.

**From nature to algorithm:**

1. _Ants as agents_ – an ant is a simple agent that constructs a solution step by step.
2. _Pheromones as memory_ – pheromones store data from previous traversals/solutions and influence the creation of subsequent solutions.
3. _Nodes and paths_ – nodes represent a state or location, while paths (edges) represent transitions between nodes.
4. _Decision-making_ – the decision is made based on probability, pheromone intensity, and path quality.

**Algorithm parameterization:**

1. _Number of ants_ – determines how many solutions are explored in each iteration; a larger number of ants improves exploration but at the expense of longer computation time.
2. _Pheromone importance_ – controls how willing the ants are to follow paths that previously proved to be good; higher importance causes more frequent traversal of already known solutions (poorer exploration?).
3. _Heuristic information importance_ – defines how much the ants rely on problem-specific information.
4. _Pheromone evaporation rate_ – a higher evaporation rate encourages greater exploration; a lower rate stabilizes the search process.
