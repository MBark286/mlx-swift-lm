The two `.mov` resources are for testing the MediaProcessing pipeline for correctness and validation.
`Resources/ModernBert` holds the ModernBERT parity fixtures (see the end of this file).

The video file was created via FFMPEG via

```
ffmpeg \
-f lavfi \
-i smptebars=size=1920x1080:rate=30:duration=5.0 \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-vf "drawtext=text='Frame\\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:alpha=0.5:box=1:boxborderw=4,drawtext=text='TC':x=(w-tw)/2:y=(lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:fontcolor=white:timecode='01\\:00\\:00\\:00':timecode_rate=(30)" \
-c:v libx264 \
-c:a aac \
-crf 23 \
-preset medium \
-pix_fmt yuv420p \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y 1080p_30.mov
```

and the audio only file 

```
ffmpeg \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-c:a aac \
-crf 23 \
-preset medium \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y audio_only.mov
```

## ModernBERT fixtures

`Resources/ModernBert/tiny-silu` and `Resources/ModernBert/tiny-gelu` are two tiny ModernBERT
checkpoints with random weights (4 layers, hidden size 16, 2 heads, `local_attention` 8) and the
states transformers 5.17.0 computes for them, in `expected.safetensors`: the embedding output, every
layer, the final hidden states and the normalized CLS vector, for a 24-token input, a 5-token input
and a right-padded batch of two. The reference ran in float32 on the CPU with eager attention
(torch 2.14.0).

- `tiny-silu` uses the key layout and the transformers 4 `config.json` keys of
  `ibm-granite/granite-embedding-97m-multilingual-r2`.
- `tiny-gelu` is saved as `ModernBertForMaskedLM` (keys under `model.` plus the masked-LM head), with
  the `config.json` that transformers 5 writes, like `answerdotai/ModernBERT-base`.
