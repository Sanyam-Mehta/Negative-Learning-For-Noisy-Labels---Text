from fastai.text import *
from utils import read_sst5
import pandas as pd
import torch

dataset_location = "/datasets/location"
ULMFIT_MODEL_LOCATION = "/path/to/model"


class Callback():
    "Base class for callbacks that want to record values, dynamically change learner params, etc."
    _order = 0

    def on_train_begin(self, **kwargs: Any) -> None:
        "To initialize constants in the callback."
        pass

    def on_epoch_begin(self, **kwargs: Any) -> None:
        "At the beginning of each epoch."
        pass

    def on_batch_begin(self, **kwargs: Any) -> None:
        "Set HP before the output and loss are computed."
        pass

    def on_loss_begin(self, **kwargs: Any) -> None:
        "Called after forward pass but before loss has been computed."
        pass

    def on_backward_begin(self, **kwargs: Any) -> None:
        "Called after the forward pass and the loss has been computed, but before backprop."
        pass

    def on_backward_end(self, **kwargs: Any) -> None:
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass

    def on_step_end(self, **kwargs: Any) -> None:
        "Called after the step of the optimizer but before the gradients are zeroed."
        pass

    def on_batch_end(self, **kwargs: Any) -> None:
        "Called at the end of the batch."
        pass

    def on_epoch_end(self, **kwargs: Any) -> None:
        "Called at the end of an epoch."
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        "Useful for cleaning up things and saving files/models."
        pass

    def jump_to_epoch(self, epoch) -> None:
        "To resume training at `epoch` directly."
        pass

    def get_state(self, minimal: bool = True):
        "Return the inner state of the `Callback`, `minimal` or not."
        to_remove = ['exclude', 'not_min'] + getattr(self, 'exclude', []).copy()
        if minimal: to_remove += getattr(self, 'not_min', []).copy()
        return {k: v for k, v in self.__dict__.items() if k not in to_remove}

    def __repr__(self):
        attrs = func_args(self.__init__)
        to_remove = getattr(self, 'exclude', [])
        list_repr = [self.__class__.__name__] + [f'{k}: {getattr(self, k)}' for k in attrs if
                                                 k != 'self' and k not in to_remove]
        return '\n'.join(list_repr)


class ComplementProbabilities(LearnerCallback):
    def __init__(self):
        self.original_vals = []

    def on_loss_begin(self, out: Tensor) -> Any:
        "Handle start of loss calculation with model output `out`."
        self.state_dict['last_output'] = out
        self.original_vals.append(out)
        return 1 - F.log_softmax(self.state_dict['last_output'])


data = TextClasDataBunch.from_csv(dataset_location, 'text.csv')

bs = 48  # batch_size

data_lm = (TextList.from_folder(dataset_location + "/Text")
           .split_by_rand_pct(0.2)
           .label_for_lm()
           .databunch(bs=bs))

data_lm.save(ULMFIT_MODEL_LOCATION + '/data_lm.pkl')

print("LANGUAGE MODEL WRITTEN")

data_lm = load_data(ULMFIT_MODEL_LOCATION, '/data_lm.pkl', bs=bs)

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn.load_encoder('fine_tuned_enc')

learn.lr_find()

learn.recorder.plot()

learn.unfreeze()
learn.fit_one_cycle(cyc_len=20, max_lr=1e-3, moms=(0.8, 0.7))

learn.save_encoder(ULMFIT_MODEL_LOCATION + '/ft_enc')

print("ENCODER SAVED")

learn.load(ULMFIT_MODEL_LOCATION + '/ft_enc')

learn.freeze()
