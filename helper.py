import torch
import torch.nn as nn

def loss_constraint_fn(x, x_lens, out_prob, rel_idx):
    pen_down = (~x[:, :, -1].bool()).float()
    p_rel = out_prob[:, :, rel_idx].sum(dim=-1)

    loss_constraint = -(torch.log(torch.clamp(1 - p_rel, 1e-10, 1)) * pen_down) # (B, T)
    loss_constraints = 0
    for loss_constraint_seq, x_len in zip(loss_constraint, x_lens):
        loss_constraints += loss_constraint_seq[:x_len].sum()
    loss_constraint_mean = loss_constraints / len(x)
    return loss_constraint_mean

class LossFunction(nn.Module):
    def __init__(self, weight=10):
        super().__init__()
        self.weight = weight
        self.loss_constraint = loss_constraint_fn

    def forward(self, x_hat, x, y, x_lens, y_lens, rel_idx):
        loss_ctc = nn.CTCLoss()(x_hat.log_softmax(-1).permute(1, 0, 2), y, x_lens, y_lens)
        loss_constraint = self.loss_constraint(x, x_lens, x_hat.softmax(-1), rel_idx)

        return loss_ctc + self.weight * loss_constraint, loss_ctc, loss_constraint

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.blank = 0

    def forward(self, emission: torch.Tensor, mask_rel = None):
        """Given a sequence emission over labels, get the best path
        Args:
            emission (Tensor): Logit tensors. Shape `[seq_len, num_label]`.
            mask_rel (Tensor, optional): Mask for the relationship postions (pen_up).

        Returns:
            List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1) 

        if mask_rel != None:
            indices[torch.where(mask_rel > 0)] = self.blank
        indices = torch.unique_consecutive(indices, dim=-1).cpu().numpy()
        indices = [i for i in indices if i != self.blank]
        return self.vocab.decode(indices)

loss_fn = LossFunction(weight=0)
