# Model Shapes

## Vision Model

`vision_encoder.onnx`

### Inputs

| Name | Shape | Type |
|------|-------|------|
| pixel_values | [batch_size, 3, height, width] | tensor(float) |

### Outputs
| Name | Shape | Type |
|------|-------|------|
| image_features | [batch_size, sequence_length, 768] | tensor(float) |

> sequence_length = floor(Reshape_2813_o0__d0*Reshape_2813_o0__d1*Reshape_2813_o0__d2/batch_size) + 1

## Embed Model

`embed_tokens.onnx`

### Inputs

| Name | Shape | Type |
|------|-------|------|
| input_ids | [batch_size, sequence_length] | tensor(int64) |

### Outputs

| Name | Shape | Type |
|------|-------|------|
| inputs_embeds | [batch_size, sequence_length, 768] | tensor(float) |

## Encoder Model

`encoder_model.onnx`

### Inputs

| Name | Shape | Type |
|------|-------|------|
| attention_mask | [batch_size, encoder_sequence_length] | tensor(int64) |
| inputs_embeds | [batch_size, encoder_sequence_length, 768] | tensor(float) |

### Outputs

| Name | Shape | Type |
|------|-------|------|
| last_hidden_state | [batch_size, encoder_sequence_length, 768] | tensor(float) |

## Decoder Model

`decoder_model.onnx`

### Inputs
| Name | Shape | Type |
|------|-------|------|
| encoder_attention_mask | [batch_size, encoder_sequence_length] | tensor(int64) |
| encoder_hidden_states | [batch_size, encoder_sequence_length, 768] | tensor(float) |
| inputs_embeds | [batch_size, decoder_sequence_length, 768] | tensor(float) |

### Outputs

| Name | Shape | Type |
|------|-------|------|
| logits | [batch_size, decoder_sequence_length, 51289] | tensor(float) |
| present.*.decoder.key | [batch_size, 12, decoder_sequence_length, 64] | tensor(float) |
| present.*.decoder.value | [batch_size, 12, decoder_sequence_length, 64] | tensor(float) |
| present.*.encoder.key | [batch_size, 12, encoder_sequence_length, 64] | tensor(float) |
| present.*.encoder.value | [batch_size, 12, encoder_sequence_length, 64] | tensor(float) |

> Note: The present.* outputs are repeated for indices 0-5, representing attention layers.
> (There is a total of 20 present.* tensors in the output)
