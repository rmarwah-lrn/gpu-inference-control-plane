# Class 10

Class 9b stack plus three new files:

- `kernels/engineering.py` — attention kernel HBM / tile / paged gather
- `router/kv/kv_eviction.py` — which prefixes die when a worker is over KV budget
- `router/warmup.py` — faster replica warmup (fewer graphs, compile cache, prefix seed)
- `router/nccl.py` / `router/nixl.py` — empty KV hop backends (`KV_BACKEND=nccl|nixl`). Lab default is mooncake.

Spin up a Lambda instance and add it to your `.env` file.

Lambda's cloud firewall only allows SSH. Leave the Step 2 SSH session open — it tunnels the lab ports. On the Mac, open only `127.0.0.1` URLs, never the public Lambda IP.

- Open WebUI — http://127.0.0.1:30030 — after Step 4. No login.
- orch-serve — http://127.0.0.1:8080 — gateway. Locust `--host` and the REPL use this.
- Grafana — http://127.0.0.1:31495 — after Step 20. User `admin`. Password from Step 21.
- Locust UI — http://127.0.0.1:8089 — after Step 24. Locust's own UI, not the gateway.

---

## Laptop — kernels + eviction + warmup

Mac terminal.

```
cd class-code/class10
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
pytest tests/
python -m kernels.engineering --q 128 --kv 2048
python -m router.kv.kv_eviction --policy prefix_protect --need 1024
python -m router.warmup --budget 1500 --prefix 256
```

Expect green tests. Kernel print: `naive/flash` is large on long prefill, small on decode (`--q 1`). Eviction print: `prefix_protect` drops `old-unique` and keeps `shared`. Warmup print: `speedup` vs capturing every CUDA-graph bucket.

---

## Lambda cluster

### Step 1

Mac terminal, already in `class-code/class10`.

```
bash setup/sync_to_lambda.sh
```

### Step 2

Same Mac terminal. This is the only SSH. Leave it open for the rest of the lab.

```
bash setup/ssh.sh
```

### Step 3

Same terminal — prompt is now `ubuntu@…:~/class10$`. Do not open another terminal.

```
bash setup/lambda_setup.sh
```

### Step 4

Same SSH terminal.

```
bash setup/lambda_cluster.sh
```

### Step 5

Same SSH terminal.

```
kubectl get deploy,svc,scaledobject
```

### Step 6

Same SSH terminal.

```
bash setup/smoke_sliced.sh
```

Expect `SLICED SMOKE PASS`.

---

## Chat in the browser

### Step 7

New Mac terminal (local). Do not SSH. Leave the Step 2 SSH terminal open.

```
open http://127.0.0.1:30030
```

Open WebUI — no username or password. If the page does not load, Step 2 is not up.

### Step 8

Same browser tab. In the model picker choose **text** (or **vision**). Send one message and wait for a reply. The UI talks to orch-serve on the cluster (`OPENAI_API_BASE_URL=http://orch-serve:8080/v1`), not to the public Lambda IP.

---

## Send traffic from the REPL

### Step 9

Same Mac terminal as Step 7.

```
cd class-code/class10
```

### Step 10

Same Mac terminal as Step 7.

```
source .venv/bin/activate
```

### Step 11

Same Mac terminal as Step 7.

```
set -a && source .env && set +a
```

### Step 12

Same Mac terminal as Step 7. This starts the REPL — stay here through Step 20.

```
python -m gateway.repl
```

Wait for the prompt. Type **one line**, then wait for `ROUTE` and `TEXT`.

### Step 13

Same REPL. Do not open another terminal.

```
text Write one sentence about a GPU.
```

### Step 14

Same REPL.

```
vision What color is this?
```

### Step 15

Same REPL.

```
audio Transcribe: hello from class 10.
```

### Step 16

Same REPL. This is a prefill → decode KV hop. Wait for `HANDOFF` — `kv_hop` must not be `None`. Grafana Mooncake hops / blocks should increment.

```
hop Write one sentence about a GPU.
```

### Step 17

Same REPL. Drops the hopped prefix so the router cannot stick to a ghost cache.

```
evict
```

Same REPL. Kernel cost model, then eviction under pressure.

```
kernel 128 2048
pressure 1024 prefix_protect
warmup 1500
warmup seed
```

### Step 18

Same REPL.

```
metrics
```

### Step 19

Same REPL.

```
profile
```

### Step 20

Same REPL. After this you are back at the shell.

```
quit
```

### Step 21

Same Mac terminal as Step 7, now a normal shell again.

```
tail -n 5 traces/requests.jsonl
```

---

## Day 2 — Grafana

### Step 22

Step 2 SSH terminal (already on the GPU box). Do not SSH again. Cluster from Step 4 must be up.

```
bash setup/day2_observability.sh
```

Wait until it finishes. The last lines print the Grafana NodePort and the admin password. Copy the password now.

### Step 23

Same SSH terminal as Step 22. Run this if you missed the password.

```
kubectl -n monitoring get secret grafana -o jsonpath='{.data.admin-password}' | base64 -d; echo
```

Username is always `admin`. Password is the string that just printed — it is different every install. Do not commit it.

### Step 24

New Mac terminal (local), or the Step 7 Mac terminal. Do not SSH.

```
open http://127.0.0.1:31495
```

Log in as `admin` with the password from Step 23. Dashboards are under **Dashboards** — names start with `Class 10 /`. They stay mostly flat until Locust is running.

---

## Locust

### Step 25

Same Mac terminal as Step 24. Do not SSH.

```
pip install -r requirements-load.txt
```

### Step 26

Same Mac terminal as Step 25. Locust holds this terminal until you stop it with Ctrl-C.

```
locust -f app/locustfile.py --host http://127.0.0.1:8080
```

Wait until the terminal says the web UI is on port 8089.

### Step 27

New Mac terminal (local). Do not SSH. Leave Locust running.

```
open http://127.0.0.1:8089
```

### Step 28

Same Locust browser tab. Do not change the host — it must stay `http://127.0.0.1:8080`.

Set **Number of users** to `4` and **Ramp up** (spawn rate) to `2`. Click **Start**. Let it run at least one minute so Grafana `rate()` panels have samples. Then click **Stop**. A burst of `429` / `tenant_tokens` is admission doing its job.

---

## Crew flood + Grafana walk

### Step 29

Same Mac terminal as Step 27 (the one that is not blocked by Locust). Do not SSH.

```
ORCH_URL=http://127.0.0.1:8080/v1 python -m app.crew_flood --rounds 48 --workers 8
```

### Step 30

Mac browser, Grafana already open at http://127.0.0.1:31495. Open the Class 10 dashboards in this order:

1. Class 10 / Cluster
2. Class 10 / Success and failures
3. Class 10 / Overview (metrics.py)
4. Class 10 / Gateway + admission
5. Class 10 / Router
6. Class 10 / KEDA
7. Class 10 / HAMi slices
8. Class 10 / Mooncake KV
9. Class 10 / Pods and replicas
10. Class 10 / vLLM

Mooncake hops stay `0` until Step 16 `hop`. Evicts stay `0` until Step 17 `evict`. Overflow / sticky stay `0` unless those paths fire.

### Step 31

Step 2 SSH terminal. Already on the box — do not SSH again.

```
curl -sf http://127.0.0.1:8080/metrics | head
```

### Step 32

Same SSH terminal as Step 31.

```
curl -sf http://127.0.0.1:50051/metrics
```

### Step 33

Terminate your Lambda instance hahaha!!!
