
## Recap of your Computer Architecture Class

(it’s ok even if u didn’t take a computer architecture course because we are reviewing this only to understand fundamental concepts in gpu perf)

**Clock**

Clock is the fundamental unit that directly affects to the speed of program. If we think of a computer as a state-changing machine, clock is what determines a state and the faster the clock cycle changes (or clock _ticks_), it means the faster state changes.

**CCT(Clock Cycle Time) and CR(Clock Rate)**

There are two ways to measure how fast a clock cycle is. Clock Cycle Time means how much time it takes for a single clock cycle. Clock Rate means how much clock cycles can happen in one second. As you can see they have a reciprocal relationship and ultimately measures the same thing — how fast a clock cycle is!

For example, let’s say a CPU chip has a frequency of 3GHz. That means it runs 3,000,000,000 cycles per second. So the Clock Rate of this CPU is 3GHz($3 \times 10^9$) cycles/second. Clock Cycle of this CPU is $\frac{1}{3,000,000,000}$second per cycle which is $\approx$ 0.33ns/cycle.

{{< note >}}

GHz

GHZ(”gigahertz”) is a unit of frequency where 1Hz means 1 cycle per second. So 1GHz means 1,000,000,000 cycles per second.

{{< /note >}}


**Instructions and CPI(Cycles Per Instruction)**

We must remember that just because the clock rate is fast, it doesn’t mean the program will run fast. This is because (in a Von Neumann computer) program is basically a set of instructions. If the clock rate is fast but it takes a lot of cycles per instruction, that program will run slow. This is why CPI is also important to keep in mind.

CPI is the weighted average of instructions and Cycles per those instructions.

For example, say we have a simple program that only has three instructions: add, multiply, and load.

| Instruction | Fraction of Program | Cycles |
| ----------- | ------------------- | ------ |
| `ADD`       | 50%                 | 1      |
| `MUL`       | 30%                 | 3      |
| `LOAD`      | 20%                 | 10     |

the CPI (weighted average of this program) is 0.5 x 1 + 0.3 x 3 + 0.2 x 10 = 3.4

High CPI usually means the core is frequently waiting on latency sources such as cache misses, branch mispredictions, or other bottlenecks (but usually cache miss is critical)


## Moving From CPU to GPU

**Index vs Dimension**

An _index_ identifies a specific execution instance, while a _dimension_ defines the total size of an execution space.

You can directly convert ~Idx ⇒ index of ~ / ~Dim ⇒ Size of ~

Inside a CUDA thread:
- `blockIdx.x` is the block index that uniquely identifies a block within the grid.
- `blockDim.x` is a constant giving the number of threads per block.
- `threadIdx.x` is the thread index that uniquely identifies a thread within its block.

