# Solving flow-shop scheduling with ant colony algorithm

## Flow-shop scheduling

Flow-shop scheduling problem is a variant of a optimal job scheduling.\
In the general job scheduling variant we are given $n$ jobs $J_1, J_2, \dots, J_n$ of varying **processing times**, which need to be scheduled on $m$ machines $M_1, M_2, \dots, M_m$ with varying **processing power**. The goal of the problem is to find a optimal **makespan** - minimizing length of the schedule.\
The flow-shop variant is specific in a way that each job requires exactly $m$ operations and $i$-th operation of the job must be executed on a $i$-th machine. No machine can perform more than one operations simultaneously.

In our extended variant we introduced additional constraint - some jobs must be executed before some other jobs.

Most of flow-shop scheduling variants are **NP-hard**.

## Ant colony algorithm

TODO
