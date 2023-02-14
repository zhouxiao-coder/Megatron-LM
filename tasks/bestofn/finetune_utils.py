def best_of_n_forward_step(batch, model):
    raise NotImplementedError("""参考 tasks/data_utils.py 中的_cross_entropy_forward_step
    
道理上BestOfN应该

  tokens, mask, label = get_from(batch)
  batch_size, n, seq_len = tokens.size()
  logits = model(tokens.reshape(n*batch_size, seq_len), mask.reshape(n*batch_size, seq_len))
  logits.reshape(n, batch_size)
  return logits, partial(_best_of_n_loss_func, label)
""")