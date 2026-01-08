import math
from torch import optim

def get_warmup_cosine_scheduler(optimizer, lr_warmup_epochs, max_epochs, target_recon_weight):
    """
    Creates a scheduler that linearly warms up from 0 to base_lr,
    then uses cosine decay to 0.
    """
    def lr_lambda(current_epoch):
        if current_epoch < lr_warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, lr_warmup_epochs))
        else:
            # Cosine decay
            progress = float(current_epoch - lr_warmup_epochs) / float(max(1, max_epochs - lr_warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress)) * target_recon_weight

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)