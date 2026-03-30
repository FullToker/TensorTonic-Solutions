def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    if step < warmup_steps:
        lr = step * initial_lr / warmup_steps
    elif total_steps == warmup_steps:   # no decay range, avoid div/0                        
          return final_lr   
    elif step<=total_steps:
        lr =  final_lr+ (total_steps-step)/(total_steps-warmup_steps) *(initial_lr-final_lr)
    else:
        lr = final_lr
    return lr
        
        