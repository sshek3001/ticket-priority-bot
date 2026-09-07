# TICKET PRIORITY BOT

- The Purpose of this bot is to assess a task created for a Particular project and categorise the priority of the task based on the company's requirement.
- We will achieve this using preference tuning and DPO.


### SFT results:
The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 128009, 'pad_token_id': 128009}.
- {'eval_loss': '3.104', 'eval_runtime': '0.593', 'eval_samples_per_second': '8.431', 'eval_steps_per_second': '8.431', 'eval_entropy': '2.503', 'eval_num_tokens': '3740', 'eval_mean_token_accuracy': '0.4724', 'epoch': '1'}                                                                                                                          
- {'eval_loss': '2.606', 'eval_runtime': '0.5916', 'eval_samples_per_second': '8.452', 'eval_steps_per_second': '8.452', 'eval_entropy': '2.564', 'eval_num_tokens': '7480', 'eval_mean_token_accuracy': '0.5418', 'epoch': '2'}                                                                                                                         
- {'eval_loss': '2.395', 'eval_runtime': '0.5749', 'eval_samples_per_second': '8.698', 'eval_steps_per_second': '8.698', 'eval_entropy': '2.467', 'eval_num_tokens': '1.122e+04', 'eval_mean_token_accuracy': '0.571', 'epoch': '3'}                                                                                                                    
**Final Cross Entropy Loss: 2.395**

### DPO Results:


[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 128009, 'pad_token_id': 128009}.
{'eval_loss': '0.6464', 'eval_runtime': '1.434', 'eval_samples_per_second': '3.486', 'eval_steps_per_second': '3.486', 'eval_entropy': '3.905', 'eval_num_tokens': '6080', 'eval_logits/chosen': '1.926', 'eval_logits/rejected': '1.914', 'eval_mean_token_accuracy': '0', 'eval_rewards/chosen': '0.1017', 'eval_rewards/rejected': '0.00498', 'eval_rewards/accuracies': '1', 'eval_rewards/margins': '0.09675', 'eval_logps/chosen': '-17.12', 'eval_logps/rejected': '-18.12', 'epoch': '1'}
{'eval_loss': '0.6278', 'eval_runtime': '1.414', 'eval_samples_per_second': '3.537', 'eval_steps_per_second': '3.537', 'eval_entropy': '3.775', 'eval_num_tokens': '1.216e+04', 'eval_logits/chosen': '1.93', 'eval_logits/rejected': '1.9', 'eval_mean_token_accuracy': '0', 'eval_rewards/chosen': '0.1358', 'eval_rewards/rejected': '-3.227e-05', 'eval_rewards/accuracies': '1', 'eval_rewards/margins': '0.1358', 'eval_logps/chosen': '-16.78', 'eval_logps/rejected': '-18.17', 'epoch': '2'}
{'train_runtime': '24.56', 'train_samples_per_second': '1.628', 'train_steps_per_second': '0.244', 'train_loss': '0.6802', 'epoch': '2'}        

**final loss: 0.6278**