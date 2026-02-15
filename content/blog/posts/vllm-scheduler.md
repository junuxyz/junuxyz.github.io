+++
title = "vLLM V1 Internals (1): How Scheduler Works"
date = 2026-01-22T22:23:10+09:00
draft = true
categories = ['ML']
tags = ['Transformer', 'inference', 'vLLM']
+++

_This post is part of the LLM System Deep Dive Series. This post is based on tag [v0.13.0](https://github.com/vllm-project/vllm/releases/tag/v0.13.0)(released in 2025.12.19)_

## What is vLLM? 

According to the [official repository](https://github.com/vllm-project/vllm), **vLLM** is a fast and easy-to-use library for LLM inference and serving. As the LLM field continues its explosive growth, the industry's focus has shifted toward a critical challenge: minimizing operational costs while maximizing inference speed. The fundamental bottleneck in throughput often lies not in raw computation, but in memory management—specifically, the **KV Cache**. Among various engineering efforts to address this, the most transformative was the introduction of PagedAttention ([Kwon et al., _Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)).

This post serves as a technical autopsy of how these theoretical breakthroughs are realized in a production-grade system. To get the most out of this deep dive, I recommend reviewing the following resources which provided the conceptual foundation for my analysis:
- [Inside vLLM by Aleksa Gordić](https://www.aleksagordic.com/blog/vllm): An excellent high-level walkthrough of the vLLM V1 architecture.
- I have an upcoming survey-ish article that captures the literature of what happened during the recent 3~4 years on LLM Inference. Hopefully this gives broader big picture and insight of the inference landsacpe. I will update this part as soon as it is published :)

This article is the first in a series on vLLM internals, where I dissect the v1 codebase (specifically v0.13.0) to explain how these complex components orchestrate generation in real-time. This specific deep dive focuses on the architecture of the vLLM V1 Scheduler.

{{< note >}}
Please feel free to reach out via my email (in my [bio](https://junupark.xyz)) if there's any misinformation or feedbacks you want to give!
{{< /note >}}

## Purpose & Core Concept of Scheduler

Maximizing throughput in an inference engine is largely an optimization problem: given a stream of user prompts, how do we pack them efficiently onto a GPU? This is the job of the Scheduler.

When multiple requests arrive, the scheduler must organize and batch them based on a specific policy (e.g., FCFS or Priority). Crucially it must decide which requests receive the scarce GPU memory (KV Cache) for the upcoming step. This decision logic is critical; a poor scheduler can lead to significant latency spikes or _starvation_, where long requests monopolize resources while short requests languish. Therefore, understanding the Scheduler is key to understanding the engine's performance characteristics.

## Structure & Lifecycle of a Request

Before diving into the algorithmic details, let's establish the data structure of a `Request`. The `Request` object is the atomic unit of the scheduler—it is a state machine that holds the lifecycle of a generation sequence.
While `Request` contains many attributes (see `vllm/v1/request.py` for details), we will focus on the attributes that drive the Scheduler's decision making:

- **`request_id`**: The unique identifier for the request.
- **`status`**: The current lifecycle state (defined in `RequestStatus`). The four main states are `WAITING`, `RUNNING`, `PREEMPTED`, and `FINISHED`.
    - _Note:_ The Scheduler manages `WAITING` requests in a RequestQueue (sorted by priority), while `RUNNING` requests are tracked in a separate active set.
- **`priority` & `arrival_time`**: These determine the processing order, but their usage depends on the configured Scheduling Policy.
	- In FCFS queue: The `priority` attribute is ignored. The queue uses a standard `deque` (FIFO), so order is strictly determined by when the request was added (insertion order).
	- In Priority queue: These attributes are critical. The queue uses a `heap` structure that sorts requests based on `priority` (lower is better), using `arrival_time` as a tie-breaker.
- **`num_prompt_tokens`**: The static length of the user provided input.
- **`num_tokens`**: The total current length (`prompt + generated output`). The scheduler uses this to calculate memory requirements (`allocate_slots`) and check against the model's context limit.
- **`num_computed_tokens`**: This tracks how many tokens (prompt + generated) have already been processed and cached in the GPU memory.
    - _Crucial for Scheduling:_ If we recall what `num_tokens` is, `num_tokens - num_computed_tokens` tells the scheduler how much work is left to do in the current step.
- **`max_tokens`**: The generation limit. If `num_output_tokens` reaches this limit, the scheduler transitions the status to `FINISHED`.

{{< note >}}
Attributes such as `spec_token_ids` and `block_hashes` will be explained in advanced section. These can be treated more as an add-on to the core scheduler logic.
{{< /note >}}

Now let's look at the lifecycle of a request. 

We can visualize the request lifecycle as a state machine:

![[Pasted image 20260126233935.png]]
_Figure 1. Simplified Lifecycle of Request_
_If you are familiar with Operating Systems, you can see this flow mirrors the process schedulers found in OS._

After a user's initial prompt is converted into an `EngineCoreRequest`, it enters the Scheduler's ecosystem.

**Waiting**
Right after a request is initialized, it's state is set to Waiting. In the perspective of the Scheduler, the request gets pushed into the Waiting queue. It has no physical resources (memory blocks) allocated yet.

**Running**
The request has successfully allocated KV cache slots on the GPU and is actively generating tokens.

**Preempted**
The request was previously `Running` but was paused by the scheduler. This usually occurs due to memory starvation, which means the system needed to reclaim its KV blocks to prevent an Out-of-Memory (OOM) error or to unblock a higher-priority request. Preempted requests are prioritized at the front of the Waiting Queue. This ensures that as soon as sufficient memory becomes available, these "paused" requests are the very first to resume execution, minimizing the latency penalty of the interruption (small detail of the arrow pointing to the tip of the waiting queue :))

**Finished**
The request is done, either because it generated an EOS token or hit the `max_tokens` limit.

In every step, the scheduler assesses the available memory, updates these statuses, and decides which "Running" requests survive and which "Waiting" requests get promoted.

![[Pasted image 20260127001111.png]]
_Figure 2. Simplified Lifecycle of Request with explanation_

### Request Queues

The scheduler maintains two queues which are waiting and running.  
```python
# Priority queues for requests.  
self.waiting = create_request_queue(self.policy) # RequestQueue  
self.running: list[Request] = []  
```


There are also two scheduling policies currently supported: FCFS(First-Come-First-Served) queue and Priority queue.  

```python
class SchedulingPolicy(Enum):  
FCFS = "fcfs" # First-come-first-served (deque)  
PRIORITY = "priority" # Priority-based (heap)
```

For FCFS, the queue is a simple deque:  
```python
class FCFSRequestQueue(deque[Request], RequestQueue):  
def add_request(self, request: Request) -> None:  
self.append(request)  
  
def pop_request(self) -> Request:  
return self.popleft()  
```
  
For priority scheduling, requests are ordered by (priority, arrival_time, request_id):  
```python
def __lt__(self, other: "Request") -> bool:  
if self.priority != other.priority:  
return self.priority < other.priority  
if self.arrival_time != other.arrival_time:  
return self.arrival_time < other.arrival_time  
return self.request_id < other.request_id  
```


### Scheduling Budget

Scheduling Budget is critical to understand how scheduling works. This is because while Scheduler follows depending on its policy, it also requires to fit the total budget that can be used.

There are two constraints govern scheduling (scheduler.py:97-98):  
  
```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs  
self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens  
```
where `max_num_runnning_reqs` means the maximum concurrent sequences. This is bound to the batch size limit set by the programmer.
`max_num_batched_tokens` means maximum tokens per step which is bound to compute limitation.

## The Scheduling Algorithm

The `schedule()` method is the heart of the scheduler. It runs every step and produces a `SchedulerOutput` that tells the model runner what to compute.

The Scheduling Algorithm is consisted of two phases. As noted in Aleksa's blog, Scheduler serves the requests in the Running state first and then admit new request from Waiting queue. This behavior ensures generation doesn't stall while new prefills may need to wait.

### Phase 1. Schedule RUNNING requests

For each running requests, the scheduler calculates how many tokens to process:
```python
# scheduler.py:292-306  
num_new_tokens = (  
request.num_tokens_with_spec # Total tokens (prompt + output + spec)  
+ request.num_output_placeholders # Async scheduling placeholders  
- request.num_computed_tokens # Already processed  
)
```

{{< note >}}
Here, `spec` means speculative tokens. This will be 
{{< /note >}}


- Phase 1: Serve the Running requests first  
- Phase 2: Admit new requests from Waiting  
- Budget system: tokens, sequences, encoder  
- Step-by-step example (MAKE THIS DETAILED - diagrams!)  


Option A: Read the code directly  
Jump to vllm/v1/core/sched/scheduler.py:216 where schedule() starts - it's ~500 lines  
but I can walk you through the key sections.  
  
Option B: Add prints and run the test  
Add 5 print statements, run the test, see the actual flow with real data.

## Preemption: When Memory Runs Out
- Why preempt? (KV cache exhaustion)  
- Victim selection (FCFS vs Priority)  
- The recompute strategy and its tradeoffs  
  
## Integration with KV Cache  
- Block allocation flow
- Prefix caching integration
- Why num_computed_tokens matters  

Token Budget

## Async Scheduling (not "advanced" - it's default now)  
- Overlapping schedule() with GPU execution  
- State update timing  
- Potential race conditions  
  
## Performance Characteristics  
- Time complexity analysis  
- Known bottlenecks (with code line references)  
- Your benchmark results  
  
## Conclusion: What Can Go Wrong  
- Common issues traced to scheduler behavior  
- What to look for when debugging



2. Data Structures  
  
2.1 The Request Object  
  


  
2.2 Request Queues  
  

  
2.3 Scheduling Budget  
  
Two constraints govern scheduling (scheduler.py:97-98):  
  
self.max_num_running_reqs = self.scheduler_config.max_num_seqs  
self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens  
  
- max_num_seqs: Maximum concurrent sequences (limits batch size)  
- max_num_batched_tokens: Maximum tokens per step (limits compute)  
  
---  
3. The Schedule Algorithm  
  
The schedule() method is the heart of the scheduler. It runs every step and produces a SchedulerOutput that tells the model runner what to compute.  
  
3.1 High-Level Flow  
  
┌─────────────────────────────────────────────────────────────────┐  
│ schedule() │  
├─────────────────────────────────────────────────────────────────┤  
│ 1. Initialize budgets and output containers │  
│ │  
│ 2. PHASE 1: Schedule RUNNING requests │  
│ └── For each running request: │  
│ ├── Calculate tokens needed │  
│ ├── Allocate KV blocks │  
│ ├── If allocation fails: PREEMPT lowest-priority request │  
│ └── Update token budget │  
│ │  
│ 3. PHASE 2: Schedule WAITING requests (if no preemption) │  
│ └── While budget available and waiting queue not empty: │  
│ ├── Check request readiness (FSM, remote KV) │  
│ ├── Lookup prefix cache hits │  
│ ├── Calculate tokens to schedule │  
│ ├── Apply chunked prefill limits │  
│ ├── Allocate KV blocks │  
│ ├── Move request to RUNNING │  
│ └── Update budgets │  
│ │  
│ 4. Build SchedulerOutput with block tables │  
│ │  
│ 5. Advance num_computed_tokens for all scheduled requests │  
└─────────────────────────────────────────────────────────────────┘  
  
3.2 Phase 1: Scheduling Running Requests  
  
Running requests (decode phase or continued prefill) are scheduled first. This ensures generation doesn't stall while new prefills wait.  
  
# scheduler.py:261-412  
req_index = 0  
while req_index < len(self.running) and token_budget > 0:  
request = self.running[req_index]  
  
3.2.1 Calculating Tokens Needed  
  
For each running request, the scheduler calculates how many tokens to process:  
  
# scheduler.py:292-306  
num_new_tokens = (  
request.num_tokens_with_spec # Total tokens (prompt + output + spec)  
+ request.num_output_placeholders # Async scheduling placeholders  
- request.num_computed_tokens # Already processed  
)  
  
# Apply long prefill threshold (chunking)  
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:  
num_new_tokens = self.scheduler_config.long_prefill_token_threshold  
  
# Respect token budget  
num_new_tokens = min(num_new_tokens, token_budget)  
  
# Don't exceed max model length  
num_new_tokens = min(  
num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens  
)  
  
The formula captures several scenarios:  
- Normal decode: 1 token (or more with speculative decoding)  
- Continued prefill: Remaining prompt tokens  
- Async scheduling: Accounts for in-flight tokens  
  
3.2.2 KV Block Allocation  
  
The scheduler must allocate memory for the new tokens:  
  
# scheduler.py:344-399  
while True:  
new_blocks = self.kv_cache_manager.allocate_slots(  
request,  
num_new_tokens,  
num_lookahead_tokens=self.num_lookahead_tokens,  
)  
  
if new_blocks is not None:  
break # Success!  
  
# Allocation failed - preempt another request  
if self.policy == SchedulingPolicy.PRIORITY:  
preempted_req = max(  
self.running,  
key=lambda r: (r.priority, r.arrival_time),  
)  
else: # FCFS  
preempted_req = self.running.pop() # Last in queue  
  
self._preempt_request(preempted_req, scheduled_timestamp)  
preempted_reqs.append(preempted_req)  
  
if preempted_req == request:  
break # Can't even fit this request  
  
This loop implements dynamic preemption: if memory is insufficient, the scheduler evicts requests until allocation succeeds.  
  
3.2.3 Tracking Scheduled Tokens  
  
On successful allocation:  
  
# scheduler.py:407-412  
scheduled_running_reqs.append(request)  
req_to_new_blocks[request.request_id] = new_blocks  
num_scheduled_tokens[request.request_id] = num_new_tokens  
token_budget -= num_new_tokens  
req_index += 1  
  
3.3 Phase 2: Scheduling Waiting Requests  
  
After running requests, the scheduler admits new requests from the waiting queue—but only if no preemption occurred:  
  
# scheduler.py:473-474  
if not preempted_reqs:  
while self.waiting and token_budget > 0:  
  
Why skip new requests after preemption? Preemption indicates memory pressure. Admitting new requests would likely cause more preemption, wasting work.  
  
3.3.1 Request Readiness Checks  
  
Before scheduling, several conditions are verified:  
  
# Check sequence limit  
if len(self.running) == self.max_num_running_reqs:  
break  
  
# Check structured output FSM compilation  
if request.status == RequestStatus.WAITING_FOR_FSM:  
if structured_output_req and structured_output_req.grammar:  
request.status = RequestStatus.WAITING  
else:  
skipped_waiting_requests.prepend_request(request)  
continue  
  
# Check LoRA constraint  
if (self.lora_config and request.lora_request and  
len(scheduled_loras) == self.lora_config.max_loras and  
request.lora_request.lora_int_id not in scheduled_loras):  
skipped_waiting_requests.prepend_request(request)  
continue  
  
3.3.2 Prefix Cache Lookup  
  
For new requests, the scheduler checks for prefix cache hits:  
  
# scheduler.py:528-536  
if request.num_computed_tokens == 0:  
new_computed_blocks, num_new_local_computed_tokens = (  
self.kv_cache_manager.get_computed_blocks(request)  
)  
  
The get_computed_blocks method (kv_cache_manager.py:164-204) searches for cached KV blocks:  
  
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:  
if not self.enable_caching or request.skip_reading_prefix_cache:  
return self.empty_kv_cache_blocks, 0  
  
# Don't cache the last token (need logits)  
max_cache_hit_length = request.num_tokens - 1  
  
computed_blocks, num_new_computed_tokens = (  
self.coordinator.find_longest_cache_hit(  
request.block_hashes, max_cache_hit_length  
)  
)  
return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens  
  
3.3.3 Chunked Prefill Decision  
  
Long prompts are chunked to prevent head-of-line blocking:  
  
# scheduler.py:577-601  
num_new_tokens = request.num_tokens - num_computed_tokens  
  
threshold = self.scheduler_config.long_prefill_token_threshold  
if 0 < threshold < num_new_tokens:  
num_new_tokens = threshold  
  
# If chunked prefill disabled, can't split  
if (not self.scheduler_config.enable_chunked_prefill  
and num_new_tokens > token_budget):  
break # Wait for more budget  
  
num_new_tokens = min(num_new_tokens, token_budget)  
  
3.3.4 Block Allocation for New Requests  
  
# scheduler.py:645-658  
new_blocks = self.kv_cache_manager.allocate_slots(  
request,  
num_new_tokens + num_external_computed_tokens,  
num_new_local_computed_tokens,  
new_computed_blocks,  
num_lookahead_tokens=effective_lookahead_tokens,  
delay_cache_blocks=load_kv_async,  
num_encoder_tokens=num_encoder_tokens,  
)  
  
if new_blocks is None:  
break # Out of memory, stop admitting  
  
Note: Unlike running requests, waiting requests don't trigger preemption—they simply wait.  
  
3.3.5 State Transition  
  
On successful scheduling:  
  
# scheduler.py:676-716  
request = self.waiting.pop_request()  
self.running.append(request)  
  
if request.status == RequestStatus.WAITING:  
scheduled_new_reqs.append(request)  
elif request.status == RequestStatus.PREEMPTED:  
scheduled_resumed_reqs.append(request)  
  
req_to_new_blocks[request.request_id] = (  
self.kv_cache_manager.get_blocks(request.request_id)  
)  
num_scheduled_tokens[request.request_id] = num_new_tokens  
token_budget -= num_new_tokens  
request.status = RequestStatus.RUNNING  
request.num_computed_tokens = num_computed_tokens  
  
---  
4. Memory Management & KV Cache  
  
4.1 The Block Pool  
  
The BlockPool (block_pool.py:128-180) manages GPU memory as fixed-size blocks:  
  
class BlockPool:  
def __init__(self, num_gpu_blocks: int, enable_caching: bool, ...):  
# All kv-cache blocks  
self.blocks: list[KVCacheBlock] = [  
KVCacheBlock(idx) for idx in range(num_gpu_blocks)  
]  
  
# Free block queue (doubly linked list for O(1) operations)  
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)  
  
# Hash → Block mapping for prefix caching  
self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()  
  
4.2 Block Allocation Flow  
  
The allocate_slots method (kv_cache_manager.py:206-324) handles allocation:  
  
┌────────────────────────────────────────────────────────────────────┐  
│ allocate_slots() │  
├────────────────────────────────────────────────────────────────────┤  
│ Input: request, num_new_tokens, new_computed_blocks │  
│ │  
│ 1. Remove blocks outside sliding window (free stale memory) │  
│ │  
│ 2. Calculate total slots needed: │  
│ num_tokens_need_slot = num_computed + num_new + num_lookahead │  
│ │  
│ 3. Calculate blocks to allocate: │  
│ num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate│  
│ │  
│ 4. Check free blocks available: │  
│ if num_blocks_to_allocate > free_blocks: │  
│ return None # Allocation fails │  
│ │  
│ 5. Touch cached blocks (prevent eviction) │  
│ │  
│ 6. Save new computed blocks │  
│ │  
│ 7. Allocate new blocks from pool │  
│ │  
│ 8. Cache full blocks for prefix caching │  
│ │  
│ Output: KVCacheBlocks (new blocks allocated) │  
└────────────────────────────────────────────────────────────────────┘  
  
The blocks layout is documented in the code:  
  
# kv_cache_manager.py:237-248  
"""  
Blocks layout:  
-----------------------------------------------------------------------  
| < computed > | < new computed > | < new > | < pre-allocated > |  
-----------------------------------------------------------------------  
| < required > |  
--------------------------------------------------  
| < full > |  
------------------------------------------------  
| <new full> |  
--------------  
"""  
  
4.3 Prefix Caching  
  
Prefix caching uses content-addressable hashing. Each block's hash is computed from its tokens:  
  
# From request.py, block hashes are computed on token append  
def append_output_token_ids(self, token_ids: int | list[int]) -> None:  
self._output_token_ids.extend(token_ids)  
self._all_token_ids.extend(token_ids)  
  
if self.get_hash_new_full_blocks is not None:  
self.block_hashes.extend(self.get_hash_new_full_blocks())  
  
The cache lookup uses a hash map (block_pool.py:182-207):  
  
def get_cached_block(  
self, block_hash: BlockHash, kv_cache_group_ids: list[int]  
) -> list[KVCacheBlock] | None:  
cached_blocks = []  
for group_id in kv_cache_group_ids:  
block_hash_with_group_id = make_block_hash_with_group_id(  
block_hash, group_id  
)  
block = self.cached_block_hash_to_block.get_one_block(  
block_hash_with_group_id  
)  
if not block:  
return None # Cache miss  
cached_blocks.append(block)  
return cached_blocks  
  
---  
5. Preemption & Recovery  
  
5.1 Preemption Trigger  
  
Preemption occurs when a running request cannot allocate blocks (scheduler.py:359-399):  
  
# The request cannot be scheduled.  
# Preempt the lowest-priority request.  
if self.policy == SchedulingPolicy.PRIORITY:  
preempted_req = max(  
self.running,  
key=lambda r: (r.priority, r.arrival_time),  
)  
self.running.remove(preempted_req)  
else:  
preempted_req = self.running.pop()  
  
self._preempt_request(preempted_req, scheduled_timestamp)  
  
5.2 Preemption Implementation  
  
The _preempt_request method (scheduler.py:846-868) handles the state cleanup:  
  
def _preempt_request(self, request: Request, timestamp: float) -> None:  
assert request.status == RequestStatus.RUNNING  
  
# Free all allocated resources  
self.kv_cache_manager.free(request)  
self.encoder_cache_manager.free(request)  
  
# Update state  
request.status = RequestStatus.PREEMPTED  
request.num_computed_tokens = 0 # Must recompute from scratch  
request.num_preemptions += 1  
  
# Re-queue at front (high priority for retry)  
self.waiting.prepend_request(request)  
  
Key insight: Preempted requests lose all computed state (num_computed_tokens = 0). This is a design choice for simplicity—partial state recovery would require complex block  
management.  
  
5.3 Recovery  
  
Preempted requests are scheduled like new requests but with status == PREEMPTED:  
  
# scheduler.py:698-702  
if request.status == RequestStatus.WAITING:  
scheduled_new_reqs.append(request)  
elif request.status == RequestStatus.PREEMPTED:  
scheduled_resumed_reqs.append(request)  
  
---  
6. Output Processing  
  
After the model runner executes, update_from_output processes results.  
  
6.1 Token Processing  
  
# scheduler.py:1142-1398  
def update_from_output(  
self,  
scheduler_output: SchedulerOutput,  
model_runner_output: ModelRunnerOutput,  
) -> dict[int, EngineCoreOutputs]:  
  
For each scheduled request:  
  
# scheduler.py:1191-1243  
for req_id, num_tokens_scheduled in num_scheduled_tokens.items():  
request = self.requests.get(req_id)  
req_index = model_runner_output.req_id_to_index[req_id]  
generated_token_ids = sampled_token_ids[req_index]  
  
# Handle speculative decoding rejections  
if scheduled_spec_token_ids:  
num_draft_tokens = len(scheduled_spec_token_ids)  
num_accepted = len(generated_token_ids) - 1  
num_rejected = num_draft_tokens - num_accepted  
  
# Adjust for rejected tokens  
request.num_computed_tokens -= num_rejected  
  
6.2 Stop Condition Checking  
  
The _update_request_with_output method (scheduler.py:1380-1398) appends tokens and checks for stop:  
  
def _update_request_with_output(  
self, request: Request, new_token_ids: list[int]  
) -> tuple[list[int], bool]:  
stopped = False  
for num_new, output_token_id in enumerate(new_token_ids, 1):  
request.append_output_token_ids(output_token_id)  
  
stopped = check_stop(request, self.max_model_len)  
if stopped:  
del new_token_ids[num_new:] # Trim  
break  
return new_token_ids, stopped  
  
6.3 Request Completion  
  
Finished requests are cleaned up:  
  
# scheduler.py:1249-1256  
if stopped:  
kv_transfer_params = self._free_request(request)  
if status_before_stop == RequestStatus.RUNNING:  
stopped_running_reqs.add(request)  
  
The _free_request method (scheduler.py:1502-1515):  
  
def _free_request(self, request: Request) -> dict[str, Any] | None:  
assert request.is_finished()  
  
self.encoder_cache_manager.free(request)  
self.finished_req_ids.add(request.request_id)  
self.kv_cache_manager.free(request)  
del self.requests[request.request_id]  
  
---  
7. Advanced Features  
  
7.1 Speculative Decoding Integration  
  
Speculative tokens are tracked separately:  
  
# scheduler.py:414-430  
if request.spec_token_ids:  
num_scheduled_spec_tokens = (  
num_new_tokens  
+ request.num_computed_tokens  
- request.num_tokens  
- request.num_output_placeholders  
)  
if num_scheduled_spec_tokens > 0:  
del request.spec_token_ids[num_scheduled_spec_tokens:]  
scheduled_spec_decode_tokens[request.request_id] = (  
request.spec_token_ids  
)  
request.spec_token_ids = []  
  
Draft tokens are updated before the next step via update_draft_token_ids:  
  
# scheduler.py:1424-1444  
def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:  
for req_id, spec_token_ids in zip(  
draft_token_ids.req_ids,  
draft_token_ids.draft_token_ids,  
):  
request = self.requests.get(req_id)  
if request is None or request.is_finished():  
continue  
  
if self.structured_output_manager.should_advance(request):  
request.spec_token_ids = metadata.grammar.validate_tokens(  
spec_token_ids  
)  
else:  
request.spec_token_ids = spec_token_ids  
  
7.2 Encoder/Multimodal Scheduling  
  
For multimodal models, encoder inputs are scheduled alongside decoder tokens:  
  
# scheduler.py:958-1115  
def _try_schedule_encoder_inputs(  
self,  
request: Request,  
num_computed_tokens: int,  
num_new_tokens: int,  
encoder_compute_budget: int,  
shift_computed_tokens: int = 0,  
) -> tuple[list[int], int, int, list[int]]:  
  
The scheduler checks:  
1. Whether encoder output is needed for this token range  
2. Whether it's already cached  
3. Whether there's budget to compute it  
4. Whether the cache has space to store it  
  
7.3 P/D Disaggregation (KV Connectors)  
  
For disaggregated prefill/decode, the scheduler coordinates with KV connectors:  
  
# scheduler.py:538-554  
if self.connector is not None:  
ext_tokens, load_kv_async = (  
self.connector.get_num_new_matched_tokens(  
request, num_new_local_computed_tokens  
)  
)  
request.num_external_computed_tokens = ext_tokens  
num_external_computed_tokens = ext_tokens  
  
Requests waiting for remote KV transfers enter WAITING_FOR_REMOTE_KVS state:  
  
# scheduler.py:677-682  
if load_kv_async:  
skipped_waiting_requests.prepend_request(request)  
request.status = RequestStatus.WAITING_FOR_REMOTE_KVS  
continue  
  
---  
Summary  
  
The vLLM V1 scheduler achieves high throughput through:  
  
1. Unified token model: No prefill/decode distinction simplifies logic  
2. Continuous batching: New requests join running batches seamlessly  
3. Dynamic preemption: Memory pressure triggers automatic eviction  
4. Prefix caching: Content-addressable KV reuse across requests  
5. Chunked prefill: Long prompts don't block other requests  
6. Budget-based scheduling: Token and sequence limits prevent overload  
  
The algorithm prioritizes running requests (avoid generation stalls), uses greedy admission for waiting requests, and gracefully handles memory pressure through preemption. This  
design enables vLLM to achieve state-of-the-art throughput on production workloads.
