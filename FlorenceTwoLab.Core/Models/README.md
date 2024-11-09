# Models

These are the shapes of the inputs and outputs of the "base" model. 

### decoder_model.onnx:
_Inputs:_
```yaml
- name: encoder_attention_mask
  value: Int64[batch_size, encoder_sequence_length]
- name: encoder_hidden_states
  value: Float[batch_size, encoder_sequence_length, 768]
- name: inputs_embeds
  value: Float[batch_size, decoder_sequence_length, 768]
```

_Outputs:_
```yaml
- name: logits
  value: Float[batch_size, decoder_sequence_length, 51289]
- name: present.0.decoder.key
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.0.decoder.value
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.0.encoder.key
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.0.encoder.value
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.1.decoder.key
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.1.decoder.value
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.1.encoder.key
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.1.encoder.value
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.2.decoder.key
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.2.decoder.value
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.2.encoder.key
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.2.encoder.value
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.3.decoder.key
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.3.decoder.value
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.3.encoder.key
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.3.encoder.value
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.4.decoder.key
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.4.decoder.value
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.4.encoder.key
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.4.encoder.value
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.5.decoder.key
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.5.decoder.value
  value: Float[batch_size, 12, decoder_sequence_length, 64]
- name: present.5.encoder.key
  value: Float[batch_size, 12, encoder_sequence_length, 64]
- name: present.5.encoder.value
  value: Float[batch_size, 12, encoder_sequence_length, 64]
```

### embed_tokens.onnx:
_Inputs:_
```yaml
- name: input_ids
  value: Int64[batch_size, sequence_length]
```

_Outputs:_
```yaml
- name: inputs_embeds
  value: Float[batch_size, sequence_length, 768]
```

### encoder_model.onnx:
_Inputs:_
```yaml
- name: attention_mask
  value: Int64[batch_size, encoder_sequence_length]
- name: inputs_embeds
  value: Float[batch_size, encoder_sequence_length, 768]
```

_Outputs:_
```yaml
- name: last_hidden_state
  value: Float[batch_size, encoder_sequence_length, 768]
```

### vision_encoder.onnx:
_Inputs:_
```yaml
- name: pixel_values
  value: Float[batch_size, 3, height, width]
```

_Outputs:_
```yaml
- name: image_features
  value: Float[batch_size, floor(Reshape_2813_o0__d0*Reshape_2813_o0__d1*Reshape_2813_o0__d2/batch_size) + 1, 768]
```


