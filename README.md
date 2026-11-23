# PocketTTS ONNX Export

This repo exports gated [Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts) checkpoints to ONNX and writes bundle outputs directly into the sibling `pocket-tts-onnx` runtime repo.

Published exported weights and inference code:

- [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx)

It currently targets:

- `english_18m_2026-10`
- `english_2026-04`
- `english_2026-09_2`
- `french_24l`
- `german`
- `german_24l`
- `italian`
- `italian_24l`
- `portuguese`
- `portuguese_24l`
- `spanish`
- `spanish_24l`

## What It Produces

For each language bundle, the exporter writes:

- `bundle.json`
- `tokenizer.model`
- `bos_before_voice.npy`
- `flow_lm_main.onnx`
- `flow_lm_flow.onnx`
- `mimi_decoder.onnx`
- `mimi_encoder.onnx`
- `text_conditioner.onnx`

If quantization is enabled, it also writes:

- `flow_lm_main_int8.onnx`
- `flow_lm_flow_int8.onnx`
- `mimi_decoder_int8.onnx`
- `mimi_encoder_int8.onnx`
- `text_conditioner_int8.onnx`

Output goes to:

```text
pocket-tts-onnx/onnx/<language>/
```

## Usage

1. Install dependencies:

```bash
sfw pip install -r requirements.txt
```

2. Export one bundle:

```bash
python export.py --language english_2026-04
```

3. Export one bundle and quantize in place:

```bash
python export.py --language english_2026-04 --quantize
```

4. Export from a local config file instead of a named language:

```bash
python export.py --config /path/to/model.yaml
```

5. Run strict verification during export:

```bash
python export.py --language english_2026-04 --exact
```

## Export Flow

`export.py` is the entry point. It runs:

1. `scripts/export_mimi_and_conditioner.py`
2. `scripts/export_flow_lm.py`
3. `scripts/quantize.py` if `--quantize` is enabled

The exporter also writes bundle metadata used by the `pocket-tts-onnx` runtime:

- tokenizer filename
- state manifests for FlowLM and Mimi
- sample/frame metadata
- BOS-before-voice tensor path
- sampler metadata: `sampler_type`, `sampler_conditioning_dim` and `eos_threshold`
- preprocessing flags such as `remove_semicolons`

## Implementation Notes

### Split FlowLM

FlowLM is exported as two graphs:

- `flow_lm_main`: transformer backbone plus state updates
- `flow_lm_flow`: stateless flow-matching step

This keeps the LSD loop in the runtime and allows dynamic step counts and temperature control.

The inputs and output of `flow_lm_flow` depend on the model's `flow_lm.flow.type`, which is written to `bundle.json` as `sampler_type`:

- `lsd`: inputs `c`, `s`, `t` and `x`, output `flow_dir`. The runtime runs the decode loop.
- `drifting` (used by `english_18m_2026-10`): inputs `c` and `x`, output `latent`. The head returns the sample in one step, so there are no time inputs and no loop.

When a config sets `flow_lm.z_dim`, the backbone output is projected to that width before the sampler and EOS head use it. `bundle.json` records this width as `sampler_conditioning_dim`. `conditioning_dim` stays the transformer width of the `text_embeddings` input.

### Explicit Stateful I/O

Pocket TTS uses stateful streaming modules internally. During export, those modules are patched so their caches and counters become explicit ONNX inputs and outputs.

### Voice Cloning Support

The bundle format preserves the v2 voice-cloning path:

- `mimi_encoder.onnx` encodes reference audio
- `bos_before_voice.npy` is exported for models that prepend a learned BOS-before-voice embedding
- bundle metadata records the state layout needed by the runtime

### Quantization

Quantization uses `onnxruntime.quantization.quantize_dynamic` and targets `MatMul` operators. This is the safe CPU-oriented path used by this repo.

### Decoder Config and Activation

A config can give the decoder a different architecture from the encoder with `mimi.decoder_seanet` and `mimi.decoder_transformer`. Without them the decoder reuses `mimi.seanet` and `mimi.transformer`.

Feed-forward blocks use the tanh approximation of GELU, matching current upstream `pocket-tts`.

## Repo Layout

```text
pocket-tts-onnx-export/
├── export.py
├── onnx_export/
│   ├── bundle_metadata.py
│   ├── export_utils.py
│   └── wrappers.py
├── pocket_tts/
├── scripts/
│   ├── export_flow_lm.py
│   ├── export_mimi_and_conditioner.py
│   └── quantize.py
├── pocket-tts-onnx/
│   └── onnx/
└── requirements.txt
```

## Requirements

Install from:

```bash
sfw pip install -r requirements.txt
```

Main packages:

- `torch`
- `onnx`
- `onnxruntime`
- `huggingface_hub`
- `safetensors`
- `sentencepiece`
- `scipy`

## Source Model

This exporter is wired for the gated voice-cloning repo:

- `kyutai/pocket-tts`

It does not target the older no-voice-cloning release.

## License

This repo includes modified code derived from [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts).

- Original Pocket TTS code: MIT
- Export/runtime bundle artifacts: subject to the upstream model and dataset terms from Kyutai / Hugging Face
