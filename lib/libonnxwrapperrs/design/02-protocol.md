# step 1 — JSONL wire protocol

## depends on

- none (foundation; everything else references this).

## 1.1 framing

how bytes on stdin/stdout/stderr are split into messages.

- line format: **one JSON object per line**, UTF-8, newline (`\n`)
  terminated. a JSON object MUST NOT contain literal newline
  characters; strings use `\n` escapes.
- request delimiter: `\n` on stdin.
- response delimiter: `\n` on stdout.
- stderr usage: **diagnostic only** — free-text logs, startup banner,
  panic/stack traces. stderr is NOT part of the protocol and callers
  MUST NOT parse it as JSONL. (correlated errors go on stdout; see 1.5.)

## 1.2 message envelope

every protocol message on **stdin (requests)** and **stdout
(responses/errors)** shares this shape:

```json
{
  "id": "<string, request correlation id>",
  "type": "request | response | error",
  "payload": { }
}
```

- `id`: **required on requests**, **echoed verbatim on the matching
  response or error**. callers use it to correlate a response with the
  request they sent. if a request cannot be parsed, the error uses
  `id: null` (see 1.5).
- `type`: one of `request`, `response`, `error`.
- `payload`: command-specific object. for a request it contains a
  `cmd` field (see `02-interface.md`); for a response/error it contains
  the command result or an error object.
  **`payload` is a nested JSON object, never a JSON-encoded string**
  (no double-encoding). large embedding arrays are the only case that
  makes a line long; that is expected.

`ts` is intentionally **omitted** — correlation is by `id`, not time.

## 1.3 request stream (stdin)

- who writes: the caller only. modelcaller never writes to stdin.
- pipelining: a caller MAY write several request lines before reading
  any response (pipelining is allowed).
- processing order: modelcaller processes requests **strictly
  sequentially** (one at a time, FIFO). it emits exactly one
  `response` or `error` line per received request, in the order
  requests were received. `id` is for correlation, not an ordering
  guarantee beyond this FIFO.
- EOF semantics: **EOF on stdin = graceful shutdown signal**. after
  EOF, modelcaller finishes any in-flight request, flushes stdout,
  then exits 0. the caller must not send more requests after closing
  stdin.

## 1.4 response stream (stdout)

- who writes: modelcaller only.
- ordering: responses are emitted in request-received order, each
  echoing the request `id`.
- content: only `type: "response"` and `type: "error"` lines ever
  appear on stdout. no logs, no banners.
- backpressure / deadlock warning: OS pipe buffers are finite. a
  caller that writes a large request while modelcaller writes a large
  response (e.g. big embedding batches) can deadlock if both block on
  a full pipe. **callers MUST read stdout on a dedicated thread**
  (or via async I/O) rather than write-then-read.

## 1.5 diagnostic/error stream (stderr)

- structured errors: correlated errors are sent on **stdout** as
  `type: "error"` (not stderr) so the caller can match them by `id`.
- log lines: free text (may optionally be JSONL for tooling, but
  callers MUST NOT depend on this). used for startup banner, model
  load progress, and panic traces.
- on unparseable input: modelcaller writes one `error` to stdout with
  `id: null` and `code: "bad_req"`, then continues reading (it does
  not abort the stream).

## 1.6 example

input (caller -> modelcaller, stdin):
```jsonl
{"id":"1","type":"request","payload":{"cmd":"embed","model":"nomic-embed-text-v1","texts":["hello world"]}}
{"id":"2","type":"request","payload":{"cmd":"ping"}}
```

output (modelcaller -> caller, stdout):
```jsonl
{"id":"1","type":"response","payload":{"embeddings":[[0.01,-0.02,0.03]]}}
{"id":"2","type":"response","payload":{"ok":true}}
```

stderr (modelcaller, diagnostic — not parsed by caller):
```
[info] modelcaller 0.1.0 starting (stdio mode)
[info] loaded model nomic-embed-text-v1
```

malformed input example:
```jsonl
{"id":null,"type":"error","payload":{"code":"bad_req","msg":"invalid JSON on line 3: ..."}}
```

## 1.7 multi-vector embed variant

for `multi_vector` registry models (late interaction / ColBERT, see
bench/bench1/design/17-late-interaction.md) modelcaller sends
`opts.multi_vector: true` plus `opts.projection_files` (the model's
`extra_files`, resolved in the model dir; may be empty). onnxwrapperrs
then skips mean-pooling and returns one vector per token as
base64-packed f32 little-endian rows instead of JSON float arrays:

```jsonl
{"id":"1","type":"response","payload":{"multi_vector":true,"dim":128,"counts":[12,7],"embeddings_b64":["<12*128 f32 LE, b64>","<7*128 f32 LE, b64>"]}}
```

- with `projection_files`: the PyLate Dense chain (loaded once per
  wrapper process, cached on the session) is applied to each token
  state, then per-token L2 normalization.
- without `projection_files`: the model is expected to already emit
  projected token states (mLateOn's fused exports do), so the token
  vectors are only L2-normalized.

`counts[i]` is the token count for text i (padded positions dropped);
row j of text i is `embeddings_b64[i][j*dim*4 .. (j+1)*dim*4]` bytes.




