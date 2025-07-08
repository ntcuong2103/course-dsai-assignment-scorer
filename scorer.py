from submission_fn import (
    Vocab,
    InkmlDataset,
    InkmlDataset_PL,
    LSTM_TemporalClassification,
    MathOnlineModel,
)

import inspect
import numpy as np
import torch
from pytorch_lightning import Trainer
from helper import GreedyCTCDecoder, loss_fn
from torchaudio.functional import edit_distance as ed

# base config
root_dir = "dataset/crohme2019"
# root_dir="dataset/crohme2019/crohme2019"
train_data = "dataset/crohme2019_train.txt"
val_data = "dataset/crohme2019_valid.txt"
test_data = "dataset/crohme2019_test.txt"

checkpoint_path = "epoch=46-val_wer=0.1327.ckpt"
hidden_size = 256
num_layers = 2


# checkpoint_path = "artifacts/model-0e7o3aep:v8/model.ckpt"
# hidden_size = 128
# num_layers = 3
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         test_loss         │    0.6167247891426086     │
# │   test_loss_constraint    │   0.0003048815415240824   │
# │         test_wer          │    0.1275891214609146     │
# │       test_wer_rel        │    0.10645334422588348    │
# │      test_wer_rel_ex      │    0.10657280683517456    │
# │       test_wer_sym        │    0.14478223025798798    │
# │      test_wer_sym_ex      │    0.14479823410511017    │
# └───────────────────────────┴───────────────────────────┘
# vocab = Vocab("vocab.json")

# char2idx = {line.strip():id for id, line in enumerate(open('crohme_vocab.txt').readlines())}
# char2idx.update({'':len(char2idx)})
# mapping = [char2idx[vocab.idx2char[id]] for id in range(len(vocab.idx2char))]



def test_task1():
    """
    Test the task1 function.
    """
    annotations = [
        "dataset/crohme2019_train.txt",
        "dataset/crohme2019_test.txt",
        "dataset/crohme2019_valid.txt",
    ]

    vocab = Vocab()
    vocab.build_vocab(annotations)
    vocab.save_vocab("vocab.json")
    print("Task 1 passed: Vocab built and saved successfully.")


def test_task2():
    # inkml_path = 'dataset/crohme2019/crohme2019/valid/18_em_0.inkml'
    vocab = Vocab("vocab.json")
    val_ds = InkmlDataset(annotation=val_data, root_dir=root_dir, vocab=vocab)
    if len(val_ds[583]) == 4:
        feature, target_tensor, input_len, label_len = val_ds[583]
    elif len(val_ds[583]) == 5:
        feature, target_tensor, input_len, label_len, _ = val_ds[583]
    else:
        raise ValueError("Unexpected number of elements in dataset item.")

    import numpy.testing as npt

    assert feature.shape == (614, 4)
    npt.assert_allclose(
        feature[:, :3].mean(axis=0),
        np.array([0.17677799, 0.29519369, 11.35540311]),
        rtol=1e-5,
    )
    npt.assert_allclose(
        feature[:, :3].var(axis=0),
        np.array([3.75390418e-01, 5.06219812e-01, 4.19017361e02]),
        rtol=1e-2,
    )
    npt.assert_array_equal(torch.where(feature[:, -1])[0], [ 81,  97, 118, 196, 278, 293, 311, 329, 357, 415, 430, 449, 505, 575, 593])
    print("Task 2 passed: InkmlDataset works correctly.")


def test_task3():
    # Check signature of InkmlDataset_PL and decide to use vocab or vocab_file

    sig = inspect.signature(InkmlDataset_PL.__init__)
    # Check if vocab is in kwargs of signature
    if "vocab" not in sig.parameters:
        kwargs = {"vocab_file": "vocab.json"}
    else:
        kwargs = {"vocab": Vocab("vocab.json")}
    
    kwargs.update({
        "root_dir": root_dir,
        "train_data": test_data,
        "val_data": test_data,
        "test_data": test_data,
        "batch_size": 4,
    })

    # custom
    kwargs.update({
        "use_constraint_loss": True,
    })
    

    dm = InkmlDataset_PL(
        **kwargs
    ) 

    dm.setup()
    test_loader = dm.test_dataloader()
    batch = next(iter(test_loader))

    if len(batch) == 4:
        features, targets, input_lengths, label_lengths = batch
    elif len(batch) == 5:
        features, targets, input_lengths, label_lengths, _ = batch
    else:
        raise ValueError("Unexpected number of elements in dataset item.")

    assert features.shape == (4, 1681, 4)
    assert targets.shape == (4, 87)
    import numpy.testing as npt

    npt.assert_array_equal(sorted(input_lengths.flatten().tolist(),reverse=True), [1681,  567,  495,  434])
    npt.assert_array_equal(sorted(label_lengths.flatten().tolist(),reverse=True), [87, 41, 35, 15])
    print("Task 3 passed: InkmlDataset_PL works correctly.")


def test_task4():
    input_size = 4
    output_size = 109
    hidden_size = 256
    num_layers = 2

    model = LSTM_TemporalClassification(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=output_size,
    )

    assert model.forward(torch.randn(2, 10, input_size)).shape == (2, 10, output_size)
    print("Task 4 passed: LSTM_TemporalClassification works correctly.")


from torchaudio.models.decoder import cuda_ctc_decoder, CUCTCDecoder


class VocabTester(Vocab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rels = ["Right", "Sub", "Sup", "Above", "Below", "Inside", "NoRel"]


class LSTM_TemporalClassificationTester(LSTM_TemporalClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):  # rewrite if needed
        for layer in self.children():
            if isinstance(layer, torch.nn.LSTM):
                x, _ = layer(x)
            elif isinstance(layer, torch.nn.Linear):
                x = layer(x)
                break
        return x
        return out[:, :, mapping]  # map to vocab size


class MathOnlineModelTester(MathOnlineModel):
    def __init__(self, *args, **kwargs):
        # Use inpsect signature
        sig = inspect.signature(MathOnlineModel.__init__)
        # Check if vocab is in kwargs of signature
        if "vocab" not in sig.parameters:
            kwargs.pop("vocab", None)
            
        super().__init__(*args, **kwargs)
        if "vocab" not in sig.parameters:
            self.vocab = VocabTester("vocab.json")

        self.loss_fn = loss_fn
        self.cuda_decoder = CUCTCDecoder(
            list(self.vocab.char2idx.keys()), blank_skip_threshold=0.95
        )  # blank must be 0
        self.greedy_decoder = GreedyCTCDecoder(self.vocab)
        self.model = LSTM_TemporalClassificationTester(
            input_size=kwargs["input_size"],
            hidden_size=kwargs["hidden_size"],
            num_layers=kwargs["num_layers"],
            num_classes=kwargs["output_size"],
        )

    def base_step(self, x, y, x_lens, y_lens):
        x_hat = self.model(x.float())
        loss, loss_ctc, loss_constraint = self.loss_fn(x_hat, x, y, x_lens, y_lens, [self.vocab.char2idx[c] for c in self.vocab.rels])

        total_edits = 0
        total_lens = 0

        decoded = self.cuda_decoder(x_hat.log_softmax(-1), x_lens.int())
        for decoded_seq, y_seq, y_len in zip(decoded, y, y_lens):
            label = y_seq[:y_len].cpu().numpy()
            edit_distance = ed(
                self.vocab.decode(label), decoded_seq[0].words
            )  # first hyp
            total_edits += edit_distance
            total_lens += y_len
        wer = total_edits / total_lens

        return loss, loss_ctc, loss_constraint, wer, decoded

    def test_step(self, batch, batch_idx):
        if len(batch) == 4:
            # batch = (x, y, x_lens, y_lens)
            x, y, x_lens, y_lens = batch
        elif len(batch) == 5:
            x, y, x_lens, y_lens, pen_ups = batch
        
        x_hat = self.model(x.float())

        loss, loss_ctc, loss_constraint, wer, decoded = self.base_step(
            x, y, x_lens, y_lens
        )

        pen_ups = x.cpu().float()[:, :, -1]


        # relation @ penup: <blank> or <rel>

        #         p_rel = x_hat.log_softmax(-1)[torch.where(pen_up > 0)]

        total_edits = 0
        total_lens = 0

        total_edits_sym = 0
        total_lens_sym = 0

        total_edits_rel = 0
        total_lens_rel = 0

        total_edits_sym_ex = 0
        total_edits_rel_ex = 0

        for seq, decoded_seq, y_seq, y_len, pen_up in zip(
            x_hat.detach().cpu(), decoded, y, y_lens, pen_ups
        ):
            label = y_seq[:y_len].cpu().numpy()

            label_sym = [
                l for l in self.vocab.decode(label) if l not in self.vocab.rels
            ]
            label_rel = [l for l in self.vocab.decode(label) if l not in label_sym]

            decoded_sym = [l for l in decoded_seq[0].words if l not in self.vocab.rels]
            decoded_rel = [l for l in decoded_seq[0].words if l not in decoded_sym]

            edit_distance = ed(label_sym, decoded_sym)
            total_edits_sym += edit_distance
            total_lens_sym += len(label_sym)

            edit_distance = ed(label_rel, decoded_rel)
            total_edits_rel += edit_distance
            total_lens_rel += len(label_rel)

            # exact decode
            # mask: rels + <blank>
            blank_idx = 0
            masked_indices = torch.tensor(
                [self.vocab.char2idx[l] for l in self.vocab.rels] + [blank_idx]
            )
            decoded_rel_ex = self.vocab.decode(
                [
                    token
                    for token in masked_indices[
                        seq[:, masked_indices].argmax(-1)[torch.where(pen_up > 0)]
                    ].numpy()
                    if token != blank_idx
                ]
            )
            decoded_sym_ex = self.greedy_decoder.forward(seq, pen_up)

            edit_distance = ed(label_sym, decoded_sym_ex)
            total_edits_sym_ex += edit_distance

            edit_distance = ed(label_rel, decoded_rel_ex)
            total_edits_rel_ex += edit_distance

            # if decoded_rel != decoded_rel_ex:
            # print (label_rel, decoded_rel, decoded_rel_ex, ed(label_rel,decoded_rel), ed(label_rel, decoded_rel_ex))

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_loss_constraint",
            loss_constraint,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self.log("test_wer", wer, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_wer_sym",
            total_edits_sym / total_lens_sym,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_wer_rel",
            total_edits_rel / total_lens_rel,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_wer_sym_ex",
            total_edits_sym_ex / total_lens_sym,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_wer_rel_ex",
            total_edits_rel_ex / total_lens_rel,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss


def test_task8(checkpoint_path):
    model = MathOnlineModelTester.load_from_checkpoint(
        checkpoint_path,
        input_size=4,
        output_size=109,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab=VocabTester("vocab.json"),
    )

    model.eval()
    dm = InkmlDataset_PL(
        root_dir=root_dir,
        train_data=test_data,
        val_data=test_data,
        test_data=test_data,
        # vocab_file="vocab.json",
        vocab=Vocab("vocab.json"),
        batch_size=32,
        use_constraint_loss=True,
    )
    trainer = Trainer(
        accelerator="auto",
        enable_progress_bar=True,
        devices=1,
        fast_dev_run=False,
    )

    trainer.test(model, datamodule=dm)


def main():
    """Main function to run the scorer."""
    test_task1()
    test_task2()
    test_task3()
    test_task4()

    # import wandb
    # run = wandb.init()
    # artifact = run.use_artifact('cuong-nt-vgu-ai-2025/math_online_2025/model-0e7o3aep:v8', type='model')
    # artifact_dir = artifact.download()
    # checkpoint_path = artifact_dir + '/model.ckpt'

    test_task8(checkpoint_path)


if __name__ == "__main__":
    main()
