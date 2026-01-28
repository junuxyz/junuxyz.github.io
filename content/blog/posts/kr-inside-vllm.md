+++
title = "vLLM V1 Internals (1): Engine Core Deep Dive"
date = 2026-01-14T21:49:26+09:00
draft = true
categories = ['ML']
tags = ['Transformer', 'inference', 'vLLM']
+++

_본 글은 [vLLM v0.13.0 코드 베이스](https://github.com/vllm-project/vllm/releases/tag/v0.13.0)를 직접 분석한 내용입니다. 글의 구조는 [Aleksa Gordic의 vLLM 분석](https://www.aleksagordic.com/blog/vllm)을 참고했으나, 원문에서 생략됐으나 이해에 도움이 될만한 구현 세부 사항, 코드 레벨 설명, 그리고 Prefix Caching·Guided Decoding 등 추가 feature 분석은 직접 코드를 따라가며 작성했습니다. 또 [commit 42172ad]([https://github.com/vllm-project/vllm/tree/42172ad)(2025.08.10) 기준으로 작성된 원문을 [v0.13.0](https://github.com/vllm-project/vllm/releases/tag/v0.13.0)(2025.12.19)으로 업데이트하며 변경된 내용들을 반영했습니다._

vLLM의 PagedAttention 개념은 유명하다. 하지만 실제 코드로 들어가면 방대한 양에 압도된다. 이 글은 vLLM v0.13.0(V1 Engine)의 코드를 해부하여, 추상적인 개념이 구체적인 파이썬 객체로 어떻게 매핑되는지 분석하고, NIPA A100 환경에서 그 성능을 검증한다.

{{< note >}}
지엽적이라고 판단되는 부분들은 
# Part 1. LLM Engine & Engine Core

우선 가장 단순한 설정인 offline inference, single GPU, no parallelism 기준으로 어떻게 실행이 되는지 확인해보도록 하겠다.

vllm의 [basic.py](https://github.com/vllm-project/vllm/tree/v0.13.0/examples/offline_inference/basic/basic.py)를 살펴보면:

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    # Create an LLM.
    llm = LLM(model="facebook/opt-125m")
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
	# ...

if __name__ == "__main__":
    main()
```

두 단계를 거쳐 process되는 것을 확인할 수 있다.

```python
	# Step 1. LLM class 에 vLLM config를 전달한다.
    llm = LLM(model="facebook/opt-125m")
    # Step 2. LLM의 `generate` method를 통해 prompt 들을 전달한다.
    outputs = llm.generate(prompts, sampling_params)
```


해당 section에서는 두 단계가 내부적으로 어떤 식으로 작동하는지 자세하게 살펴볼 것이다.

## Step 1. LLM Engine Initialization

```python
    llm = LLM(model="facebook/opt-125m")
```
LLM의 호출은 가장 먼저 engine args를 바탕으로 LLMEngine을 인스턴스화한다.

### LLM Engine
vLLM의 [**LLM Engine class**](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/engine/llm_engine.py#L46))를 살펴보면 크게 다음과 같은 component들이 존재한다:
1. [**vllm config**](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/engine/llm_engine.py#L61-L89)
	- model, cache, parallel, scheduler 등 vLLM과 관련된 모든 것들을 설정하는 인스턴스
2. [**Input Processor**](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/engine/llm_engine.py#L91)
	- raw input(Ex. plain text)을 token으로 변환한 후 request(`EngineCoreRequest`)로 wrap한다. 이 과정에서 validation, tokenization, processing 등을 담당한다.
3. [**Output Processor**](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/engine/llm_engine.py#L98-L102)
	- `EngineCoreOutput`을 `RequestOutput`으로 변환
	- (바로 아래에서 설명할) EngineCore는 EngineCoreRequest를 입력으로 받고 EngineCoreOutputs를 출력한다. Input Processor는 prompt를 Engine Core 가 받을 수 있는 입력값을 만들고 Output Processor는 `EngineCoreOutputs`를 `RequestOutput`의 형태로 변환한다.
4. [**Engine Core Client**](https://github.com/vllm-project/blob/v0.13.0/vllm/v1/engine/llm_engine.py#L108-L115)
	- LLMEngine과 Engine Core(`engine_core`)를 연결한다.
	- Client의 종류는 다섯 가지가 존재한다:
		1. `InprocClient`
		2. `SyncMPClient`
		3. `AsyncMPClient`
		4. `DPAsyncMPClient`
		5. `DPLBAsyncMPClient`
		- 각각에 대한 설명은 아래 Multiprocess 부분에서 자세히 설명하도록 하겠다.

### Engine Core Client
LLM Engine의 [Engine Core Client가 만들어지는 부분](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/engine/llm_engine.py#L108-L115)을 살펴보면,
```python
self.engine_core = EngineCoreClient.make_client(
	multiprocess_mode=multiprocess_mode,
	asyncio_mode=False,
	vllm_config=vllm_config,
	executor_class=executor_class,
	log_stats=self.log_stats,
)
```

Engine Core Client 의 `make_client` 를 통해 Engine Core Client를 인스턴스로 생성하는 것을 확인할 수 있다. (코드에서는 `self.engine_core`라고 써있지만 사실은 Engine Core Client이다.$^*$) Engine Core Client는 Engine Core가 communicate하는 방식을 설정하는 역할을 한다. 코드에서 확인할 수 있듯이 process mode(`multiprocess_mode`)와 통신 방식(`asyncio_mode`)에 따라 정해진다. Engine Core Client의 종류와 그 차이에 대해서는 뒤에서 다루도록 하겠다.

$^*$ 사실 이름을 헷갈리게 `engine_core`로 설정하는 이유가 납득이 되지 않아 이 부분에 대해서는 따로 [PR](https://github.com/vllm-project/vllm/pull/32513)로 수정을 요청하였다.

해당 섹션에서는 synchronous, not parallel인 경우를 다루기로 했으므로 Engine Core Client는 Inprocess Cleint(`InprocClient`)가 된다.

`InprocClient` class를 살펴보면 

```python
class InprocClient(EngineCoreClient):
	# ...
    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)
```

이런 식으로 `EngineCore`를 인스턴스화하는 것을 확인할 수 있다. 이를 통해 `InprocClient`의 method는 `self.engine_core`을 호출하도록 돼 있다.

즉 정리하자면 LLM Engine이 인스턴스화되는 과정은 다음과 같다:
1. `LLMEngine.__init__`이 `EngineCoreClient.make_client(multiprocess_mode=False, ...)`을 호출
2. `InprocClient`이 반환됨
3. `InprocClient.__init__`을 통해 EngineCore가 생성됨 `self.engine_core = EngineCore(...)`

 `InprocClient`의 경우 사실 Engine Core Client wrapper가 굳이 필요 없고 뒤에서 살펴볼 다른 Client 종류들에 더 필요한 Class이다. 즉 Engine Core는 모든 종류의 client가 공유하는 핵심 엔진이라면 Engince Core Client는 그 Engine Core가 communicate하는 방식에 대해서만 정의한 부분이다.

{{< note >}}
**Engine Core vs Engine Core Client**

EngineCore는 inference를 실행하는 실제 engine을 의미하고 EngineCoreClient는 EngineCore와 frontend(`LLM`) 사이를 연결하는 창구 역할을 한다.
{{< /note >}}

## Engine Core

Engine Core Client로 감싸진 Engine Core의 핵심적인 부분은 크게 세 부분으로 나뉜다:

### [Model Executor](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/engine/core.py#L102-L104)
```python
# Setup Model.
self.model_executor = executor_class(vllm_config)
if executor_fail_callback is not None:
	self.model_executor.register_failure_callback(executor_fail_callback)
```

Model Executor는 GPU에서 model forward pass (e.g. attention process)를 실행한다. 내부적으로 GPU device 하나당 Worker 하나를 생성하고 이 Worker가 연산을 수행하게 된다.
Model Executor는 GPU의 수에 따라 `UniProcExecutor`, `MultiprocExecutor`, `RayDistributedExecutor` 등이 있다. 해당 예시에서는 하나의 GPU를 사용하는 경우를 다루고 있기 때문에 Uni Process Executor (`UniProcExecutor`)를 기준으로 설명하겠다.

Model Executor가 생성될 때 내부적으로 두 단계를 거친다:
1. 먼저 [Device 초기화](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/worker/gpu_worker.py#L179-L282)가 진행된다. 이 단계에서는 Worker에 CUDA device를 할당하고 VRAM이 충분한지 확인하고 `model_runner`를 생성한다. 이 시점에서 모델 가중치(weight)은 아직 GPU에 없다. 이는 memory profiling(메모리 프로파일링)을 먼저 해야 하기 때문이다.
2. 그 다음 [model 로딩](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/worker/gpu_worker.py#L286-L289)
	- model 아키텍쳐와 weight을 GPU에 로드
	- eval mode(추론 전용)로 설정
	- (선택적으로) `torch.compile()` 을 통해서 최적화할 수도 있음

Executor 초기화가 완료된 후에는 Engine Core에서 KV Cache를 할당 및 초기화한다.:
1. Layer별 KV Cache spec을 구함.
	- 원래는 `FullAttentionSpec`만 사용했으나 현재는 [Jamba](https://arxiv.org/abs/2403.19887)와 같이 Transformer의 variation이 많아 각각의 모델 layer에 필요한 KV Cache들이 다름
2. dummy forward pass를 실행해 GPU memory snapshot을 찍고 얼마나 메모리가 필요한지 측정(프로파일링)함
3. KV Cache 할당 및 초기화


### [Scheduler](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/engine/core.py#L120-L150)
Scheduler는 CPU에서 request scheduling, continuous batching, KV cache allocation 등을 담당한다. Step 2. `llm.generate()` 에서 더 자세히 설명하도록 한다.

5.  **KV Cache System**
	PagedAttention 메커니즘을 활용하여 CPU 및 GPU의 KV cache 메모리를 효율적으로 관리하는 역할을 한다. 구체적인 구성 요소는 다음과 같다:
	- **KV Cache config**: 블록 크기, 블록 수, cache group 등에 대한 설정 정보입니다.
	- **Block Allocation**: 앞서 수행한 메모리 프로파일링 결과를 바탕으로 실제로 할당 가능한 GPU/CPU 블록의 수를 판단함함

	KV Cache System은 다음과 같은 방식으로 초기화되고 작동한다:
	1. Model Executor로부터 KV cache spec 정보를 전달받음
	2. GPU VRAM에 있는 KV 블록 메모리의 물리적 할당/반납 역할을 수행하고 BlockPool 클래스의 `free_block_queue`를 통해 관리됨
	3. KV cache manager는 이 `free_block_queue`를 유지 관리함. 이를 위해 전체 가용한 VRAM을 설정된 블록 크기(기본 크기 = 16 token) 단위로 잘라냄.
		- - KV Cache manager의 `free_block_queue`는 이중 연결 리스트(doubly linked list/queue)로 구현되어 있음
	4. Request가 들어오면 필요한 만큼의 블록 개수를 할당하고 요청 처리가 끝나면 해당 블록을 해제하여 메모리를 다시 확보(free)하는 작업을 담당

{{< note >}}>
Waiting Queue랑 free block queue는 완전히 독립적인 다른 것. waiting queue는 CPU에서 어떤 request를 처리할지에 대한 scheduling 정보를 담은 queue이고 kv cache manager 의 free block queue는 GPU에서 각 kv cache의 정보가 담긴 물리적 주소들을 관리한다.

- Scheduler Queue
    - request, can be either WAITING/RUNNING
- Free block pool
    - KV block, can be either FREE/USED 
{{< /note >}}

### 정리
정보의 밀도가 매우 높았으므로 위의 내용이 전부 이해가 되지 않았을 가능성이 높다. 큰 그림을 다시 그려보자면,

![[Pasted image 20260122153532.png]]
Figure 1. Engine Initialization components (Credit: Aleksa Gordic)

우리는 vLLM의 엔진 중 가장 단순한 형태인 Offline Inference, single GPU, no parallelism 기준으로 `llm = LLM(model="facebook/opt-125m")` 한 줄이 내부적으로 어떻게 실행되고 있는지 보고자 하였다. LLM() wrapper는 LLMEngine을 인스턴스화한다. LLM Engine class는 vLLM config에서 전달된 값을 바탕으로 vLLM과 관련된 대부분 설정을 초기화하고 request로서 들어간 prompt를 엔진 친화적인 EngineCoreRequest 만드는 Input Processor, EngineCore에서의 처리를 마친 후 RequestOutput의 형태로 변환하는 Output Processor를 생성한다. 그리고 Engine Core Client를 initialize한다.

EngineCoreClient는 `make_client()` 함수에서 어떤 설정인지에 따라 client의 종류가 달라진다. 우리는 가장 단순한 형태를 확인하기로 하였으므로 이 경우는 `InprocClient`에 해당한다. `InprocClient`는 초기화되면서 enginecore 인스턴스를 생성 및 초기화한다.

Engine Core는 모델을 실행하고 메모리를 관리하기 위한 굉장히 많은 component들이 존재한다. 대표적으로는 model forward pass(step)를 담당하는 Model Executor, 다음 step에 들어갈 request를 관리하는 Scheduler, 메모리를 블록으로 관리하는 PagedAttention을 위한 KV Cache System이 있다. 물론 이 이외에도 Aborts Queue, Multi-Modal Components, Request Block Hasher 등등의 component가 존재하지만 모든 것을 다루기에는 너무 방대해지기 때문에 핵심만 살펴보기로 한다.

## Step 2. `llm.generate()`

```python
    outputs = llm.generate(prompts, sampling_params)
```

다음으로 Engine이 initialize된 이후 `generate` function을 호출할 때 다음과 같은 과정이 발생한다.

내부적으로는 각각의 prompt에 대해
1. [unique한 request ID를 생성](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/entrypoints/llm.py#L1611-L1626)하고 (log의 목적으로) TTFT를 저장한다.
2. Input Processor를 호출하여 토큰화 과정을 진행하고 `prompt`, `prompt_token_ids`, `type`으로 이루어진 dictionary를 return 값으로 반환한다.
3. EngineCoreRequest에 2에서 생성된 info를 넣고 기타 meta data(priority policy, sampling params etc.)도 추가한다
4. request(`EngineCoreRequest`)를 Engine Core에 넘겨준다. Engine Core는 `Request` 라는 객체로 이 정보를 감싸고 Reqeust 객체의 status를 `WAITING`으로 설정한다. `WAITING` 으로 설정된 request는 Scheduler의 waiting queue에 추가된다. (FCFS 인 경우 append, priority인 경우 heap-push)

`LLM.generate(…)` 과 같이 synchronous하게 호출할 경우 이 처리 과정이 끝나기 전까지 호출자 thread는 차단되기 때문에 새로운 호출이 개입할 수 없다. 그러나 FastAPI 등을 통한 Asynchronous serving을 한다면 이미 처리 중인 요청이 있는 동안에도 새 HTTP 요청을 받아 Engine에 추가로 prompt를 넣을 수 있다.

{{< note >}}
**Prompt vs Request**
쉽게 말해 prompt는 아직 새로운 token이 추가되지 않은 첫 input 상태이고 request는 prompt가 LLM 에 보내진 이후 scheduler로 관리되는 이름이다. request는 prefill/decode 과정을 통해 추가적으로 token이 붙었을 수 있다.
LLM entrypoint 코드를 [살펴보면](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/entrypoints/llm.py#L1611-L1626) prompt가 `request_id`로 바뀌는 것을 확인할 수 있다.
{{< /note >}}

continuous batching이란 매 step마다 batch의 멤버가 변할 수 있다는 것을 말한다.

Asynchronous engine인 경우 새로운 request를 실행/처리 중간에 밀어넣을 수 있다는 점에서 continuous batching이다. Synchronous batch 는 새로운 request를 추가하는 것은 불가능하지만 전체 request batch를 하나의 sequence로 flatten한 후 이를 효율적으로 처리(Ex. chunked prefill)한다는 면에서 continous batching의 성격을 보인다고 할 수 있다.

지금까지 request들을 LLM class entrypoint를 통해 전달했을 때 어떻게 Scheduler의 Waiting Queue에 들어갔는지에 대해 살펴보았다. 이 이후부터는 처리할 request가 남아있는 한 Engine은 Model Executor의 `step()` function을 계속 부르게 된다. 각 step을 호출할 때는 다음과 같은 일이 벌어진다:

1. schedule: 어떤 request를 run할지 결정한다 (이건 바로 아래 파트에서 자세하게 다룸)
2. forward pass: 각 request에 대해 Engine Core의 Model Executor를 통해 Model 을 실행하고 다음 token을 sample함.
3. postprocess: 각 Request 객체에 sampled token IDs 를 더함 (autoregressive). 만약 request를 마친 경우 KV cache blocks 를 free block queue 에 다시 넣어준다. Write-on 방식을 사용하기 때문에 free block queue에 들어갔다고 해서 그 내용이 바로 사라지지는 않는다. 그러나 queue의 순서를 따라 나중에 다른 request의 KV cache로 덮어 씌워질 수 있다.

![image.png](attachment:e95aa919-ae41-4aee-957c-4ded758db50a:image.png)

### Scheduler
가장 중요한 Request Scheduling에 대해서만 간단히 설명하면 다음 step에 어떤 request들을 처리할지에 대해 스케줄링하는 것을 말한다. Request 정책(policy)으로는 두 가지가 존재하는데 FCFS(first come first served) queue 또는 Priority queue 를 사용할 수 있다. 하나의 request는 waiting 이나 running 두 가지 상태일 수 있다. 아직 prefill 이 진행되지 않은 request들을 waiting queue에, decode가 진행중인 request는 running queue에 속하게 된다. 
Scheduler의 `schedule()` 로직에 대한 더 자세한 deep dive는 추후 별도의 post로 다루어보겠다.

inference engine이 다루는 workload 는 크게 두 개가 있다.

1. **Prefill Request**: forward pass over all prompt tokens. 즉 한 request에 대해 prompt 길이 만큼의 token을 한꺼번에 처리하고 이 때문에 GEMM, attention 연산량이 폭증한다. 따라서 comput bound (= compute 성능에 큰 영향/제약을 받음) workload이다. 이후 prompt에 대해 next token을 sampling하는 것으로 마무리된다.
2. **Decode Request**: 이전의 token들은 KV Cache를 통해 재사용하고 이를 이용해 next token sampling을 진행하는 것이다. 이는 memory-bandwidth-bound workload이다. KV cache 재사용으로 연산 자체는 prefill request보다 덜 부담되지만 kv cache + LLM weight 을 전부 로드해야 계산이 가능하기 때문에 (특히 KV 가 커질수록) VRAM에 영향을 받는다.

LLM Inference에 한정하면 Prefill은 user 의 input prompt에 대한 forward pass, Decode는 LLM의 autorgressive forward pass를 말함. V1 scheduler는 이 둘을 굳이 구분하지 않음. 즉 한 번의 `step`에 어떤 부분은 prefill, decode 이 혼재돼 있어도 동시에 처리가 가능하게 설계돼 있다. (Prefill와 Decode 에 대해 보다 자세한 설명은 https://www.junupark.xyz/blog/posts/prefill-vs-decode/ 참고)

Scheduler는 running queue에 들어와 있는 decode request 처리를 우선시한다 (Running queue에 들어왔다는 것은 이미 Prefill 과정을 마쳤음을 의미한다)

Scheduler는 각각의 decode request에 대해

1. next token을 sample하고 (대부분의 경우 하나이지만 항상 그렇지는 않음. 예외 케이스들: speculative decoding, async scheduling)
2. KV Cache Manager의 `allocate_slots` 함수를 호출하여 각 token을 어느 block에 할당할지 계산한다.
3. token budget을 업데이트한다. $\text{token budget} = \text{token budget} - \text{token used for decode (usually 1)}$. token budget의 경우 KV cache manager가 아닌 Scheduler가 step 마다 관리하는 변수로 현재 step에서 처리가 가능한 token을 의미한다.

그 다음으로 (token budget이 남아있다면) waiting queue의 prefill request를 처리한다.

1. (Prefix caching을 사용하는 경우) 이미 계산된 block(다른 request와 공유하는 prefix block)의 수를 구한다. 해당 block들은 중복 계산 없이 KV cache를 그대로 재사용한다.
2. 나머지 token들에 대하여 KV cache manager의 `allocate_slots` 함수를 호출한다.
3. 해당 request를 Waiting Queue에서 pop 하고 Running Queue로 push. status는 RUNNING으로 설정
4. token budget update

<aside> 💡

**What if?**

running queue안에 있는 decode 를 다 처리하니까 token budget이 3 token 정도 남았는데 waiting queue의 맨 앞에 있는 (가장 높은 priority) request가 4 token이어서 한번에 prefill이 안 되는 경우? 우선순위가 더 낮지만 fit하는 request을 실행

```python
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
```

</aside>

이제 2단계에서 계속 나온 KV Cache Manager의 `allocate_slots`함수를 살펴보면

1. 새로 할당돼야 할 블록의 수를 계산. Ex. 한 block이 16 token을 담을 수 있다면 전체 필요한 token / 16 을 통해 나온 수를 올림 e.g. 17/16 = 1.xx → 2 blocks needed
2. 블록 할당이 가용한지 확인
3. 블록 할당: KV cache manager의 coordinator를 이용해 block pool (free block queue)에서 처음 n block 을 가져옴. `req_to_blocks` 라는 dictionary를 통해 각 request id를 fetch 해온 KV cache blocks랑 mapping해줌

![image.png](attachment:a16641e4-0028-424b-903d-81587c9a5d01:image.png)

그리고 allocate slots를 통해 block_id, ref_cnt, block_hash를 각 block meta data로 저장한다.

- `ref_cnt`: 그 블록이 몇 군데에서 참조되는지(prefix caching / copy-on-write 같은 공유 최적화에 필요)
- `block_hash`: prefix cache에서 “이 KV 블록 내용이 이전과 같나?” 같은 식별/재사용에 쓰기 위한 키

지금까지의 내용을 요약하면 Scheduler가 inference에 주요한 두 가지 workload인 Prefill와 Decode 각각에 대해 어떤 식으로 블록을 할당해주는지에 대해 살펴보았다.

### Forward Pass

Model Executor의 `execute_model` 함수는 Worker를 거쳐 model runner에게 pass.

이때 다음과 같은 과정이 일어난다

1. update states: input_batch로부터 끝난 request들을 prune함
    
2. prepare inputs: CPU의 buffer를 GPU에 복사, slot_mapping을 build하고 attention metadata를 구축
    
    - slot_mapping은 각 sequence의 token들의 KV가 KV Cache paged memory 어디에 들어가는지를 알려주는 것? 이라고 함 (맞나?)
    
    ![image.png](attachment:5d498630-4c5d-4d47-87c6-4000b36ff9c2:image.png)
    
    그니까 내가 이 이미지 (및 slot_mapping)를 제대로 이해한 거라면 slot mapping은 continuous batching에서의 input id와 block으로 나누었을 때의 KV block 위치를 mapping한 array인건가? 그래서 block_id가 1부터 시작하니까 1은 4랑 매치가 되는거고 근데 첫 prompt는 5 token 이어서 5/4 → 2 blocks 할당되니까 다음 prompt는 다음 block에 새로 할당한다고 했을때 12부터 시작하는 그런 개념으로. 결국 바뀌는 건 새 prompt 마다 block 이 새로 바뀌고 slot_mapping을 통해 continuous batching의 각 token 에 해당하는 KV Cache Block indice가 mapping되는거? (뭐 난 대충 이렇게 이해한듯 이걸 하는 이유는 결국 forward pass를 위해 token을 이동시킨건가?)
    
3. Forward pass: custom paged attention kernels를 통해 모델을 실행. 모든 sequence 는 거대한 하나의 super sequence 로 평탄화 / 연결됨.
    
    Position indices and attention masks ensure each sequence only attends to its own tokens, which enables continuous batching without right-padding.
    
    ![image.png](attachment:2e7c3ef6-1f01-4cb4-94c6-f595f84007f0:image.png)
    
    블록으로 구획된 것을 continuous batching을 통해 flatten할 수 있게 함. 이 하나의 거대한 sequence가 “super sequence”인 것임
    

![image.png](attachment:9ae5b4fe-5c39-4d67-ab76-1ed86f9bcdd7:image.png)

prefill을 통해서 각각의 KV cache를 구하고 (사실 prefill 과정 자체는 일반 transformer calculation이랑 크게 다를 게 없음)

![image.png](attachment:17b08e76-2915-40fc-9eeb-ba68a0164c9c:image.png)

결국 paged attention 과 kv cache의 utilization은 decode 단계에서 kv cache의 재사용이고 이런 식으로 sequence 별로 next token을 predict할 때 다음 token 하나를 위한 연산만 하면 되고 나머지는 kv cache를 재사용하면 됨.

1. last token state들을 모음 (뭔소리)
2. Sample: sampling config 에 따라 다음 token을 sample함

forward pass 는 두 가지 실행 모드를 가지는데

1. eager mode: 일반적인 PyTorch forward pass
2. Captured mode: pre-captured CUDA Graph 을 실행/재생 - 이 cuda graph는 KV cache procedure init 때 생성됨 (아직은 잘 모르겠음)

지금까지는 가장 단순한 설정에서 vLLM의 내부 Engine이 어떤 과정을 거쳐 request를 처리하는지에 대해 살펴보았다. 아래 두 번째 section에서는 Prefill과 Decode의 처리량(throughput)을 높여주는 여러 feature들에 대해 설명하고 코드에서 어떤 식으로 구현돼 있는지 살펴보도록 하겠다.

# Part 2. Advanced Features

## 1. Chunked Prefill

긴 프롬프트가 있을 때 prefill step을 더 작은 chunk로 나누는 방법

prefill은 결국 prompt의 모든 token 에 대한 연산을 수행해야 되기 때문에 이를 쪼개서 접근

이렇게 하지 않으면 매우 긴 request 하나가 다른 prefill request들을 허용하지 않은 채 독점할 우려가 있음 → adding latency to all other requests

Executing the full prefill for `P` would then take ≥ 3 engine steps (> can happen if it's not scheduled for execution in one of the steps)

어쨌든 chunk 를 하나의 mini prompt sequence라고 생각하는 것

![image.png](attachment:36b7263c-aa81-4c7a-93e7-c711bef77047:image.png)

실제 vLLM의 구현 방법:

number of new tokens per step 의 상한선을 `long_prefill_token_threshold` parameter로 정함

이 parameter가 사실상 chunk의 길이가 됨. 만약 prefill request로 인해 새로 처리해야 할 토큰이 임계값인 `long_prefill_token_threshold`을 넘어설 경우 `long_prefill_token_threshold` 만큼만 잘라서 처리

vLLM V1에서는 `long_prefill_token_threshold`를 positive integer로 세팅하는 방식으로 enable할 수 있음

## 2. Prefix Caching

여러 프롬프트가 공통된 부분을 공유한다면 굳이 계산하기 보다는 최대한 재사용하기 위한 방법

실제 vLLM의 구현 방법:

`long_prefix` 를 통해 가능: KV-cache block 보다 긴 prefix를 말함 (그럼 long prefix는 실제 코드 단위라기 보다는 개념적인 그런건가 ⇒ 실제로 vLLM에 `long_prefix`라는 파라미터는 없음. 그냥 하나의 block size를 넘어서는데 다른 prompt 와 공유하는 부분을 설명하기 위한 개념적인 정의임)

incomplete block은 cache가 불가능하므로 “몫” 부분에 해당하는 block 만 cache 가 가능하고 그말인즉슨 prefix가 적어도 1 KV-cache block size 보다는 커야 prefix caching이 가능함

Ex. 공유하는 token 이 17개인데 block size가 16 token인 경우, 17 % 16 = 1 이므로 하나의 block만 prefix cache에 사용될 수 있음.

prefix caching을 사용하면 prefill 을 할 때 prefix를 제외한 prompt token 만 process할 수 있기 때문에 연산량과 시간을 단축시킴. 물론 chunked prefill과 마찬가지로 decode에는 별 도움이 안 됨. 왜냐하면 decode는 애초에 나머지 prompt가 전부 kv cache에 load 돼 있고 sampling만 진행하는 방식이기 때문

첫 `generate` 이 호출됐을 때 engine은 `hash_request_tokens`를 evoke함

hash request tokens 함수는

1. long_prefix + prompts[0] 을 16 token chunk로 나눔 (왜 하필 prompts[0]임 ..? ⇒ 그냥 첫 번째 prompt 인듯? 즉 long_prefix를 공유하는 prompt 중 첫 번째)
    
2. 그리고 각 complete chunk 마다 hash를 계산함. Hash는 previous block hash, current token, optional metadata를 합친 형태임.
    
    <aside> 💡
    
    여기서의 “chunk” 는 chunked prefill와는 전혀 다른 맥락에서 사용됨. 그냥 쉽게 말해 generate 함수가 호출되고 block size 만큼 token을 쪼갠 각 덩어리를 말하는 거임. block size만큼 쪼갠 것이지만 KV-cache block 은 아니고 해싱을 위한 논리적 단위로 생각하면 됨.
    
    </aside>
    
    (갑자기 왜..? 그리고 왜하필 hash의 형태로?? 뭔소린지 이해 x vLLM이 block에 대해 hash 값을 계산하여 관리하는 이유는 문맥 식별, 빠른 검색, meta data 반영을 위해서임.
    
    1. 문맥 식별 단순히 16개의 token 값만 가지고 hash를 만들면 서로 다른 문장에서 우연히 겹치는 16개 토큰을 동일한 것으로 착각할 위험이 있음. 즉 prompt A 에서는 block A만 겹치는데 prompt B는 block A, block B 두 개가 겹친다고 했을때 prompt C는 block A, B … 로 구성돼 있다고 하자. 이때 previous block hash 이 없다면 prompt A, prompt B 중 어디를 prefix로 가져올지에 대한 기준이 모호해지는 문제가 생김. 따라서 앞에 어떤 hash가 나타났는지의 정보 또한 반영하는 것이 prefix의 일관성을 완벽하게 보장할 수 있게 됨
    2. efficiency hash map 방법은 일반적으로 latency가 낮고 빠른 매칭을 가능하게 하는 효율적인 방법
    3. 선택적 메타데이터 포함 LoRA ID나 Cache Salt 등의 세부 설정에 따라 같은 토큰을 공유한다고 하더라도 다른 결과가 나타날 수 있기 때문에 단순히 토큰이 같은지의 여부 뿐만 아니라 모든 조건들이 완벽하게 일치할 때만 캐시를 재사용하게 함으로써 완결성 보장 )
3. 각 결과는 hash와 token ID로 구성된 BlockHash object의 형태로 저장됨. 마지막 return은 BlockHash object들의 list 형태로 반환됨
    

이는 `self.req_to_block_hashes[request_id]` 에 저장됨.

(근데 cache가 정확히 어떻게 구현돼 있는거야? 그러니까 예를 들어 4개의 prompt가 있는데 2개 , 2개씩 prompts가 prefix cache를 공유할 수 있다고 했을 때 cache에 두 prefix 모두 저장이 돼, 아니면 하나만 저장이 가능한거야

vLLM의 prefix caching 구현 방식을 살펴보면, 질문하신 시나리오처럼 여러 개의 서로 다른 prefix를 동시에 캐시에 저장하는 것이 당연히 가능합니다.,

소스 코드와 본문의 내용을 바탕으로 구체적인 구현 메커니즘을 설명해 드리면 다음과 같습니다.

1. 해시 맵(Hash Map) 기반의 관리

vLLM 엔진은 `cached_block_hash_to_block`이라는 일종의 **해시 맵(Hash Map)**을 사용하여 캐시를 관리합니다.

- 키(Key): 16개 토큰으로 이루어진 블록의 해시값입니다. 이 해시값은 해당 블록의 토큰들뿐만 아니라 이전 블록의 해시값까지 포함하여 계산되므로, 문맥적 순서가 보장됩니다.
- 값(Value): 실제 KV 캐시 데이터가 저장되어 있는 GPU 메모리의 물리적 블록 주소입니다.

따라서 질문하신 예시에서 prefix A와 prefix B가 서로 다르다면 각각 고유한 해시값을 가지게 되며, 이 해시 맵 안에 별도의 엔트리로 공존할 수 있습니다.

2. 블록 풀(Block Pool)과 가용 메모리

캐시에 저장할 수 있는 prefix의 개수는 GPU의 VRAM 용량에 의해 결정됩니다.,

- 엔진은 초기화 단계에서 사용 가능한 VRAM을 계산하여 수십만 개의 **KV 캐시 블록(free_block_queue)**을 미리 확보해 둡니다.,
- 새로운 prefix가 들어오면 이 풀에서 비어있는 블록을 할당받아 해시값과 매핑하여 저장합니다.,
- 메모리가 부족해지기 전까지는 이전에 계산된 여러 prefix들이 캐시 내에 계속 유지됩니다.

3. 참조 횟수(Reference Counting)와 재사용

- 사용 중인 경우: 특정 프롬프트가 실행 중일 때는 해당 블록의 참조 횟수(reference count)가 증가하여 보호받습니다.
- 사용이 끝난 경우: 요청이 완료되면 블록은 `free_block_queue`로 돌아가지만, 그 안에 담긴 데이터와 해시 매핑 정보는 즉시 삭제되지 않고 그대로 남겨둡니다.,
- 재사용: 나중에 동일한 prefix를 가진 요청이 오면, 엔진은 해시 맵에서 해당 블록이 여전히 유효한지 확인하고 `free_block_queue`에서 다시 꺼내 즉시 재사용합니다.

요약하자면

질문하신 4개의 프롬프트 사례에서, Prefix A를 공유하는 2개의 프롬프트와 Prefix B를 공유하는 2개의 프롬프트 모두 각각의 캐시 블록을 할당받아 동시에 저장될 수 있습니다., 캐시는 단 하나만 저장할 수 있는 공간이 아니라, 메모리가 허용하는 한 수많은 '지식 조각(블록)'들을 보관할 수 있는 거대한 창고와 같기 때문입니다.,

이 방식은 마치 여러 명의 요리사가 각자 자주 쓰는 '비법 소스(prefix)'들을 주방 냉장고(캐시)에 각각 따로 통에 담아 보관해 두는 것과 같습니다. A 요리사의 소스가 있다고 해서 B 요리사의 소스를 보관하지 못하는 것이 아니라, 냉장고 선반에 자리가 있는 한 얼마든지 여러 종류의 소스를 동시에 보관하고 필요할 때마다 꺼내 쓸 수 있는 것과 원리가 같습니다.

)

그 다음으로 Engine은 find_longest_cache_hit 을 호출해 이 hash가 이미 cached_block_hash_to_block에 이미 존재하는지 확인함. 아래 예시 기준에서는 아직 prompt 중 하나만 확인했기 때문에 cache hit이 일어나지 않음. = miss

![image.png](attachment:66696e44-99ca-4ac4-b304-4f6dab0935f2:image.png)

[1,2,3,4,5,6,7,8,9,10,2] 는 총 11 token 이므로 11 % 4 = 2 이고 나머지 마지막 3개의 token은 block을 채우지 못하므로 hashing에 포함되지 못한다. 따라서 두 개의 block만 hash_request_tokens 를 통해 2개의 BlockHash object로 반환된다. 이때 각 BlockHash object는 token id array와 hash value를 지닌다.

(여기서 hash에서 다음 hash 로 가는것에 왜 화살표 표시가 있었는지 이해를 못했었는데 이는 이전 block의 hash 정보를 다음 hash에 포함시킴으로써 연쇄적 해싱이 가능하도록 한 것을 표현한 것이다)

![image.png](attachment:f8f31dcf-35c4-407b-bc0f-e7a3b8e3c40b:image.png)

Afterwards, the forward pass will populate KVs in paged KV cache memory corresponding to KV cache blocks that we allocated above. →

CPU에서 scheduling을 할 때에는 아직 실제 GPU 연산이 일어나지는 않고 allocation 에 대한 계획만 세움. 구체적으로 `cached_block_hash_to_block` 함수를 통해 해시 맵을 뒤진 후 현재 prompt의 token chunk가 이전에 계산된 적이 있는지 확인함.

만약 cache hit인 경우 이미 데이터가 들어가 있는 GPU의 Paged Memory 의 block ID 를 가져옴.

cache miss인 경우 `free_block_queue` 에서 비어있는 block의 block ID를 가져옴.

이 과정을 통해 CPU는 request의 각 chunk (chunk의 사이즈는 block과 같음) 가 GPU 메모리의 몇 번째 block ID 를 사용하는지에 대한 slot_mapping 정보를 완성함.

이때 KV Cache hit일 경우 이미 데이터가 있으므로 forward pass를 진행하는 대신 CPU의 scheduling의 결과인 slot_mapping을 참고함. PagedAttention kernel은 slot_mapping 주소에서 이미 계산된 KV Cache Block의 value를 읽음. 반면 새로 할당된 블록일 때에만(즉 KV Cache miss) forward pass를 진행한 후 slot_mapping을 참고해 KV 값을 지정된 주소에 저장하게 됨.

여기까지 하면 첫 prompt(long_prefix + prompt[0]) 에 대한 `generate` 처리가 완료됨

해당 이미지에서는 3개의 block에 새로운 value가 채워질 뿐 kv cache hit가 나타나지 않음을 볼 수 있는데 이는 아직은 공유하는 prefix를 cpu에서 찾지 못했기 때문에 나타나는 것이다. 이어서 두 번째를 살펴보면

두 번째 prompt(prompt[1])에 대해 generate call을 할 때 이번에는 `find_longest_cache_hit` 가 `long_prefix` 만큼이 KV Cache에 존재하는 것을 포착하고 그 n blocks에 대해 재사용을 함.

다시 말하지만 prefix cache는 무조건 block 단위로만 확인하고 공유하는 부분 중 완전히 fill되지 않은 block(나머지 token)은 포함시키지 않는다.

![image.png](attachment:baf6cd8f-cad0-460f-be31-6d02fe40fcfc:image.png)

이 부분에 대해서는 부연설명이 좀 더 필요한데 KV Cache에 저장된 메모리는 (CPU memory cache의 write-on 방식과 유사하게) lazy eviction/standby 방식과 매우 유사한 논리로 작동함.

해당 예시에서 두 번째 prompt는 첫 번째 prompt에 대한 요청이 완료된 후 시작을 하는데 이 경우 reference counter는 0으로 초기화되고 block들은 다시 `free_block_queue`로 돌아간다. 그러나 `free_block_queue`로 돌아간다고 그 block의 data가 다 지워지지는 않는다. 그냥 “다시 덮어씌워질 수 있는 block”인 상태가 되는 것임. 따라서 cached_block_hash_to_block 에는 여전히 해당 block이 어떤 데이터를 갖고 있는지에 대한 hash 정보가 여전히 남아있음

이 상태라고 할 때 두 번째 prompt에서 (이미 완료된) 첫 번째 prompt와 일치하는 해시가 있다면 해당 block은 `free_block_queue`로부터 삭제되고 다시 새로운 요청에 할당됨.

block 안의 data가 진짜로 삭제되는 순간은 새로운 데이터를 저장할 빈 공간이 부족하여 해당 block을 덮어쓸 때에만 발생하는 것임. 그러나 Queue의 특성상 FIFO(First In First Out) 정책을 따르므로 free_block_queue에 들어간 block이 다시 덮어써지기 전에 다른 block들이 씌워져야 하고 이는 자연스럽게 LRU(Least Recently Used) 정책을 따를 수 있게 한다.

- 보충
    
    작성해주신 설명은 소스의 내용과 vLLM의 실제 작동 메커니즘을 **매우 정확하게 파악하고 있으며, 오개념이 없습니다.** 특히 '공간의 해제'와 '데이터의 삭제'를 구분하여 이해하신 점이 핵심을 관통하고 있습니다.
    
    소스의 내용을 바탕으로 작성하신 내용의 타당성을 하나씩 확인해 드리겠습니다.
    
    ### 1. Lazy Eviction 및 데이터 유지 (정확함)
    
    설명하신 대로, 요청이 완료되면 블록의 참조 횟수(reference count)는 0이 되고 `free_block_queue`로 돌아가지만, 그 안의 **데이터는 즉시 지워지지 않습니다.**
    
    - **소스의 근거:** 소스에서는 요청이 완료되어 블록이 풀(pool)로 반환되더라도, `cached_block_hash_to_block`을 통해 해당 블록을 다시 찾아낼 수 있다면 그 데이터가 여전히 **유효(valid)**한 것으로 간주합니다.
    - **작동 방식:** 시스템은 블록을 `free_block_queue`에서 다시 **꺼내기만(remove)** 함으로써 별도의 계산 없이 기존 데이터를 즉시 재사용합니다.
    
    ### 2. 해시 정보의 유지 (정확함)
    
    블록이 가용 상태(`ref_cnt`=0)가 되어도 CPU의 해시 맵(`cached_block_hash_to_block`)에는 해당 블록과 데이터 해시의 연결 정보가 남아 있습니다.
    
    - 이 덕분에 두 번째 요청이 들어왔을 때, 비록 해당 블록이 '사용 가능한 상태'로 분류되어 있더라도 **동일한 해시를 가지고 있다면 즉시 복구**하여 사용할 수 있는 것입니다.
    
    ### 3. 실제 삭제 시점과 LRU 정책 (정확함)
    
    가장 중요한 부분인 **"언제 데이터가 진짜로 사라지는가"**에 대한 설명도 완벽합니다.
    
    - **삭제 시점:** 블록의 데이터와 해시 정보가 실제로 무효화(Invalid)되는 순간은, 새로운 요청을 위해 `free_block_queue`에서 블록을 **꺼내어 재할당(reallocate)하기 직전**입니다.
    - **LRU 메커니즘:** 소스에 따르면 `free_block_queue`는 왼쪽(왼쪽 끝, 즉 FIFO 방식)에서 블록을 꺼냅니다.
        - 가장 오래전에 해제된 블록(가장 먼저 큐에 들어간 블록)이 가장 먼저 재할당 대상이 됩니다.
        - 결과적으로, 자주 혹은 최근에 사용된 데이터는 큐의 뒤쪽에 머물며 생존 가능성이 높아지고, 오랫동안 사용되지 않은 데이터는 큐의 앞쪽으로 밀려나 결국 덮어씌워지게 됩니다. 이는 말씀하신 대로 **자연스럽게 LRU(Least Recently Used) 정책**으로 이어집니다.
    
    ### 요약 및 결론
    
    질문자님이 정리하신 내용은 vLLM이 **메모리 할당의 오버헤드를 줄이면서도 캐시 효율을 극대화**하는 방식을 완벽히 설명하고 있습니다.
    
    - **참조 횟수 0:** "누구나 이 자리를 써도 좋다"는 허가 (데이터는 남음)
    - **해시 맵 유지:** "이 자리에 무엇이 있었는지 기억함"
    - **재할당 시 무효화:** "새로운 손님이 올 때만 기존 짐을 치움"
    - **FIFO 큐 활용:** "오래된 짐부터 순서대로 치움 (LRU)"
    
    이 흐름은 소스에서 설명하는 **Prefix Caching의 기술적 실체**를 오류 없이 담고 있습니다.
    

V1 기준 Prefix caching은 기본적으로 활성화돼 있고 `enable_prefix_caching = False`로 설정하면 방지할 수 있다.

## 3. Guided Decoding (FSM)

Guided Decoding은 LLM이 생성하는 결과물이 사용자가 원하는 특정한 형식이나 규칙을 반드시 따르도록 강제하는 기술을 말함. 이로써 답변의 완성도/일관성을 더 높일 수 있는 방법임.

각 Decoding step에서 logits가 grammar-based FSM(Finite state machine)에 제한된 것을 말함

FSM은 흐름도라고 생각하면 편함. 즉 $s_1$ 이 있을 때 다음 상태로 $s_2$ 에 해당하는 상태들로만 갈 수 있도록 강제함.

아무튼 규칙에 해당하는 token만 sample될 수 있도록 한다는 뜻임

코드 예시로 보면

```bash
guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
```

여기서 가능한 output을 Positive 와 Negative로 강제한 경우, P가 처음으로 sample된 next token이면 `o` 만 allow 할 수 있게 하는 것임. 왜냐하면 P 인데 갑자기 다음으로 o가 아닌 것은 선택지에 없기 때문에 불가

![image.png](attachment:d717d529-b845-4eba-a0d1-b0efabb7e522:image.png)

실제 코드 구현사항은 다음과 같음:

1. Engine 생성 시 StructuredOutputManager 초기화
    
    LLM engine이 생성될 때 `StructuredOutputManager`라는 component도 함께 만들어진다. 이 매니저는 guided decoding 전체를 총괄하는 역할을 하는데, 두 가지 핵심 자원에 접근할 수 있다.
    
    첫째는 tokenizer다. tokenizer를 통해 어떤 token ID가 어떤 문자열에 대응되는지 알 수 있어서 나중에 문법 규칙을 토큰 단위로 적용할 수 있게 된다.
    
    둘째는 `_grammar_bitmask`라는 tensor인데, 이게 실제로 "어떤 토큰을 허용하고 어떤 토큰을 막을지"를 저장하는 핵심 자료구조다. 이 bitmask는 매 decoding step마다 업데이트되면서 모델의 출력을 제약하는 역할을 한다.
    
2. Request 추가 시 상태 설정과 비동기 컴파일
    
    사용자가 JSON schema나 regex 같은 structured output 제약 조건을 걸어서 요청을 보내면 시스템은 이 요청을 곧바로 추론 대기열에 넣지 않는다. 대신 요청 상태를 `WAITING_FOR_FSM`으로 설정하고, 문법을 FSM(Finite State Machine)으로 변환하는 작업을 시작한다. 이 변환 작업은 xgrammar 같은 외부 백엔드 컴파일러가 담당하는데 복잡한 문법일수록 컴파일 시간이 오래 걸릴 수 있다. 그래서 이 작업을 비동기로 처리(asynchronous compile)한다. 만약 동기로 처리했다면 컴파일이 끝날 때까지 엔진 전체가 blocking되어 다른 요청들도 처리하지 못했을 것이다. 비동기 처리 덕분에 한 요청의 FSM이 컴파일되는 동안에도 엔진은 다른 요청들을 정상적으로 처리할 수 있다.
    
3. 스케줄링 단계에서의 상태 전환
    
    Scheduler가 다음 step에서 어떤 요청들을 처리할지 결정할 때 FSM 컴파일 상태도 함께 확인한다. 비동기 컴파일이 완료된 요청은 `WAITING` 상태로 전환되고 해당 `request_id`가 `structured_output_request_ids` 목록에 추가된다. 이제 이 요청은 실제 추론을 받을 준비가 된 것이다. 반면 아직 컴파일이 진행 중인 요청은 `skipped_waiting_requests`에 담겨서 이번 step에서는 건너뛰고 다음 step에서 다시 확인하게 된다. 이런 방식으로 FSM 준비가 안 된 요청이 추론 파이프라인에 들어가는 것을 방지한다.
    
4. 스케줄링 루프 종료 후 Bitmask 준비
    
    스케줄링 루프가 끝나고 이번 step에서 실행할 요청 배치가 확정되면 아직 모델을 돌리기 전에 한 가지 작업이 더 있다. `StructuredOutputManager`가 백엔드(xgrammar)에게 현재 각 요청의 FSM 상태를 보내고 그 상태에서 다음에 허용되는 토큰들이 무엇인지 `_grammar_bitmask` 형태로 받아온다. 예를 들어 `choice=["Positive", "Negative"]` 제약이 걸린 요청이 아직 아무 토큰도 생성하지 않은 초기 상태라면 FSM은 "P"와 "N"으로 시작하는 토큰만 허용한다고 알려줄 것이다. 이 정보가 bitmask에 인코딩되어 다음 단계에서 사용된다.
    
5. Forward Pass 후 Logit Masking
    
    모델의 forward pass가 끝나면 각 위치에 대해 vocabulary 크기만큼의 logit 값들이 출력된다. 여기서 `_grammar_bitmask`를 적용해서 불허 토큰들을 걸러내는데 bitmask는 메모리 효율을 위해 비트 단위로 압축되어 있다. 32비트 정수 하나가 32개 토큰의 허용 여부를 담고 있는 식이다. `xgr_torch_compile` 같은 함수가 이 압축된 bitmask를 실제 vocabulary 크기와 동일한 길이의 배열로 확장한다. vocabulary가 32,000개라면 32비트 정수 1,000개가 32,000개의 boolean 값으로 풀리는 셈이다. 확장된 mask에서 값이 0인(불허) 위치의 logit을 $-\infty$로 덮어쓴다. logit이 $-\infty$가 되면 softmax를 거친 후 해당 토큰의 확률이 정확히 0이 되므로, sampling 단계에서 그 토큰은 절대 선택되지 않는다.
    
    좀 더 풀어서 설명하자면 vocab은 모든 token을 말하고 각 token 은 token id를 가지고 있고 이 id는 위치로 표현이 된다고 했을 때 예를 들어 한 tokenizer는 32000개의 token을 사용한다고 할 때 각 token 에 대해서 softmax 를 통해 확률로 표현을 할 수 있음 (e.g. [0.2, 0.5, 0.001, 0.001, 0.02, .... ,0.002] (32000 size array)) 근데 logit masking이란 문법적으로 허용하지 않는 token들을 -infty 로 설정해서 절대 나올 수 없게 만든다는거
    
    - bitmask 원리
        
        Bitmask는 여러 개의 boolean 값(허용/불허)을 하나의 정수에 압축해서 저장하는 방식임. 즉 정수를 숫자가 아니라 비트들의 나열로 보는 것임.
        
        간단한 예시: 8개 토큰, 8비트 정수
        
        vocab이 8개라고 가정하면:
        
        ```
        token_id:  0   1   2   3   4   5   6   7
        허용여부:  허용 불허 허용 불허 불허 불허 불허 허용
        boolean:   1   0   1   0   0   0   0   1
        
        ```
        
        이걸 8개의 boolean 배열 `[1, 0, 1, 0, 0, 0, 0, 1]`로 저장할 수도 있지만 하나의 8비트 정수 `10100001` (2진수) = `161` (10진수)로 압축할 수 있음
        
        확장할 때는 역순으로 정수 `161`을 받음 → 2진수로 풀면 `10100001` → 각 비트를 분리해서 `[1, 0, 1, 0, 0, 0, 0, 1]` 배열로 만듦
        
        만약 vocab_size가 32,000이면 boolean 배열로 저장하면 32,000개 원소 필요함.
        
        그런데 32비트 정수를 쓰면 각 정수가 32개 토큰 정보를 담으니까 `32,000 / 32 = 1,000`개 정수만 있으면 됨
        
        ```
        int32[0]  → token 0~31의 허용 여부 (32비트)
        int32[1]  → token 32~63의 허용 여부 (32비트)
        int32[2]  → token 64~95의 허용 여부 (32비트)
        ...
        int32[999] → token 31968~31999의 허용 여부 (32비트)
        
        ```
        
        확장할 때 1,000개 정수 각각을 32개 비트로 풀어서 이어붙이면 32,000개 boolean 배열이 됨.
        
        따라서 한 비트에 한 토큰 정보만 담을 수 있는 boolean에 비해 메모리 효율이 32배 좋아짐. GPU에서 대량의 요청을 동시에 처리할 때 bitmask를 배치 단위로 관리해야 하는데, 압축하지 않으면 메모리 사용량이 급격히 늘어남. 또한 비트 연산은 GPU에서 매우 빠르게 처리할 수 있어서 연산 효율도 좋아짐.
        
6. Sampling 후 FSM 상태 전이
    

masking 덕분에 문법에 맞는 토큰만 샘플링된다. 샘플링된 토큰은 `accept_tokens` 함수를 통해 FSM에 입력으로 전달되고, FSM은 이를 받아서 다음 상태로 전이한다. 예를 들어 초기 상태에서 "P"가 샘플링되었다면, FSM은 "Positive" 경로의 다음 상태로 이동하고, 이제 다음 step에서는 "o"만 허용하는 새로운 bitmask가 생성될 것이다. 이 과정이 EOS 토큰이 생성되거나 다른 종료 조건을 만족할 때까지 반복된다. 결과적으로 모델은 자유롭게 생성하는 것처럼 보이지만, 실제로는 매 step마다 FSM이 정의한 문법 규칙 안에서만 토큰을 선택하게 된다.

vLLM에서는 원하는 `guided_decoding` config (및 백엔드) 를 바탕으로 설정이 가능

## 4. Speculative Decoding

Speculative Decoding에 대한 기존 블로그의 설명은 약간 부족하다고 느껴 간단한 high level 설명은 [해당 자료](https://clova.ai/tech-blog/%EC%86%8D%EB%8F%84%EC%9D%98-%ED%95%9C%EA%B3%84%EB%A5%BC-%EB%9B%B0%EC%96%B4%EB%84%98%EB%8B%A4-hyperclova-x%EC%97%90-%EC%A0%81%EC%9A%A9%ED%95%9C-speculative-decoding-%EC%9D%B4%EC%95%BC%EA%B8%B0)를 살펴보면 좋을 것 같다. 원 논문: [https://arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318)

내가 이해한 대로 설명하면 Speculative Decoding은 비싼 연산을 최소화하면서도 품질을 유지하려고 하는 방법이다. 항상 large model을 호출하여 autoregressive하게 token을 생성하는 decode 과정은 연산량이 많고 생성 속도가 느리다는 단점이 있다. Speculative Decode 방법은 다음과 같은 방식으로 이를 해결한다:

1. k개의 autoregressive한 decode 과정은 작은 LM이 생성하도록 한다. 이때 각 자리는 전체 vocab size의 벡터에 각 토큰이 나올 확률로 구성돼 있고 가장 확률이 높은 토큰이 candidate token이 된다. 따라서 k개의 candidate token이 생성된다.
2. k 번의 작은 모델의 decode과정이 끝나면 큰 모델이 병렬적으로 작은 모델의 결과를 검증하고 추가로 마지막 token에 대해서도 sample한다 (총 k+1) (참고: Clova tech blog에서는 설명의 단순화를 위해 이미지가 헷갈리게 그려져 있어, 개념상 정확하게 다시 그렸다. 다른 점 1/ Draft 과정에서는 k개, Verify 과정에서는 총합 k+1개의 candidate가 생성되는 것이 맞다 2/ 검증 시, Decode된 output 뿐만 아니라 context(prefill 또는 prompt)도 모두 LM 에 forward pass됨)
3. 검증 결과 Rejection Sampling이라는 확률 기반 샘플링 방식을 통해 각 자리의 token을 accept 혹은 reject한다. Accept / Reject의 기준이 되는 Rejection Sampling을 이해하는 것이 중요한데 이게 원래 모델의 생성 분포를 보존하는 핵심적인 장치가 되기 때문이다. 1/ `k+1`개의 token에 대하여 Large Model로 계산된 각 자리의 candidate token의 확률이 Small Model로 계산된 확률보다 클 경우 무조건 채택한다. 2/ 그렇지 않은 경우 $\frac{\text{p\_large(token)}}{\text{p\_draft(token)}}$의 확률로 accept한다. 3/ 만약 모든 `k` 개의 draft token이 accept된 경우 `(k+1)` 번째 token을 sample한다 (왜 이게 free라는거지?) 4/ 만약 rejection이 발생했다면 발생한 그 지점에서 새로운 distribution을 만들고 마지막 token을 sample한다.

Rejection Sampling은 large model에서 sample하는 것과 통계적으로 일치하게 하면서도 더 빠르고 싸게 forward pass하는 것을 돕는다. Speculative Decoding에 대한 가장 간단한 implementation은 [https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L103-L151에서](https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L103-L151%EC%97%90%EC%84%9C) 확인 가능

Speculative Decoding 과 vLLM에서 구현된 방식에 대해서는 그 자체로 내용이 방대하고 다룰 가치가 있어보여 추후 새로운 blog post에서 더 깊이 다뤄보도록 하겠습니다.

[https://untitledtblog.tistory.com/134](https://untitledtblog.tistory.com/134)

[https://www.linkedin.com/pulse/what-speculative-decoding-jin-hyung-park-d1o5c/](https://www.linkedin.com/pulse/what-speculative-decoding-jin-hyung-park-d1o5c/)

vLLM V1은 LLM draft model 방법은 지원하지 않지만 더 빠르고 덜 정확한 방법들 — n-gram, EAGLE, Medusa 을 사용한다.

ngram의 경우 다음과 같은 방식으로 사용할 수 있다:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

speculative_config={
    "method": "ngram",
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", speculative_config=speculative_config)

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

1. Prefill의 경우 target(large) model에서 run
2. 이후 decode 단계부터 `propose_draft_token_ids(k)`를 통해 k 길이의 candidate token을 draft(small) model에서 sample
3. 해당 candidate token들을 `request.spec_token_ids`에 저장
4. 다음 engine step에서 request가 running queue에 있을 때 request의 `len(spec_token_ids)` (=k) new tokens count에 더한다.
5. draft token들을 context와 concatenate 하기 위해 `spec_token_ids`를 `input_batch.token_ids_cpu`로 복사한다
6. draft token들에 대해 large model forward pass 수행
7. `rejection_sampler`를 통해 왼쪽부터 차례로 reject / accept하고 reject할 경우 output_token_id를 생성
8. 2~7 과정을 반복

## 5. Disaggregated P/D (Prefill/Decode)

[https://www.junupark.xyz/blog/posts/prefill-vs-decode/에서도](https://www.junupark.xyz/blog/posts/prefill-vs-decode/%EC%97%90%EC%84%9C%EB%8F%84) 설명했듯 Prefill와 Decode는 다른 성격의 task이고 두 task가 혼재된 상황에서 maximum throughput을 얻기 위해서는 신중하게 설계해야 합니다.

현재 request의 p/d 비율에 따라 `N` prefill instances와 `M` decode instances를 autoscaling하는 식으로 구현합니다. Prefill worker는 Key, Value를 KV-cache service에 적고 decode worker는 KV-cache service로부터 읽는 방식으로 구현됩니다.

vLLM에서는 다음과 같은 방식으로 구현이 된다:

1. Instantiation — engine을 구축할 때 connector는 두 곳에서 생성된다.
    1. worker의 init device procedure
    2. scheduler constructor
2. Cache lookup: Scheduler 가 Prefill requests 를 waiting queue에서 process한 후 connector의 `get_num_new_matched_tokens`를 호출한다. 이 함수는 다른 GPU의 KV-cache server에 해당 token의 KV cache가 저장돼 있는지 확인한다. Prefill의 경우 항상 0이 출력되고 decode의 경우 cache hit가 발생할 수 있다.

# How to Benchmark in vLLM





1. InprocClient: 같은 프로세스에서 EngineCore 실행

- multiprocess_mode=False일 때 사용

- V0 스타일 LLMEngine과 호환

1. SyncMPClient: 별도 프로세스에서 EngineCore 실행, 동기 통신

- multiprocess_mode=True, asyncio_mode=False일 때 사용

- ZMQ로 통신, 동기식 API

1. AsyncMPClient: 별도 프로세스에서 EngineCore 실행, 비동기 통신 (Data Parallel 없음)

- multiprocess_mode=True, asyncio_mode=True, data_parallel_size=1일 때 사용

- AsyncLLM에서 사용

1. DPAsyncMPClient: 별도 프로세스에서 EngineCore 실행, 비동기 통신, Data Parallel with External Load Balancer

- data_parallel_size > 1, data_parallel_external_lb=True일 때 사용

- 외부 로드밸런서 사용, 각 DP rank마다 클라이언트

1. DPLBAsyncMPClient: 별도 프로세스에서 EngineCore 실행, 비동기 통신, Data Parallel with Internal Load Balancer

- data_parallel_size > 1, data_parallel_external_lb=False일 때 사용

- 내부 로드밸런서 사용, 클라이언트가 모든 DP rank에 요청 분산 사용, 클라이언트가 모든 DP rank에 요청 분산