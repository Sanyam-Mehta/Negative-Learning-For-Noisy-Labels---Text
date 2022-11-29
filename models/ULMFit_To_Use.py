from fastai.text import *
from utils import read_sst5, read_trec6
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F



torch.cuda.set_device(3)


def choose_complement_targets(targets, 6):
    complement_targets = (targets + torch.LongTensor(targets.size(0)).random_(1, num_classes)) % num_classes
    return complement_targets


class ComplementProbabilities(LearnerCallback):
    def __init__(self):
        self.original_vals = []

    def on_loss_begin(self, out):
        "Handle start of loss calculation with model output `out`."
        self.state_dict['last_output'] = out
        self.original_vals.append(out)
        return 1 - F.log_softmax(self.state_dict['last_output'])



dataset_location = "/Data/trec6"
ULMFIT_MODEL_LOCATION = "/Saved/ULMFit"

datasets = read_trec6(dataset_location)

df_train = datasets["train"]
df_dev = datasets["dev"]
df_test = datasets["test"]


df = pd.concat([df_train, df_dev])
df.to_csv("/Data/sst5/Text/text.csv", index=False, header=False)
df_test.to_csv("/Data/sst5/Text_test/text.csv", index=False, header=False)


bs = 48  # batch_size

data_lm = TextLMDataBunch.from_df(dataset_location, train_df=df_train, valid_df=df_dev)
data_clas = TextClasDataBunch.from_df(dataset_location, train_df=df_train, valid_df=df_dev, test_df=df_test, vocab=data_lm.train_ds.vocab, bs=bs)



data_lm.save('data_lm_export.pkl')
data_clas.save('data_clas_export.pkl')

data_lm = load_data(dataset_location, 'data_lm_export.pkl')
data_clas = load_data(dataset_location, 'data_clas_export.pkl', bs=bs)

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit(1, 1e-2)
learn.predict()

learn.model.load_state_dict()

learn.data.train_dl.new(shuffle=False, sampler=SubsetRandomSampler())

learn.data.val


# learn = Learner(data, wrn_22(), metrics=accuracy).to_fp16()

learn.unfreeze()
learn.validate()
learn.fit_one_cycle(1, 1e-3)

learn.save_encoder('ft_enc')

Callback()

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.fit()
learn.load_encoder('ft_enc')

print("***")
learn.fit_one_cycle(10, 1e-2, callbacks=[ComplementProbabilities()])
print("***")

learn.freeze_to(-2) # When unfreezed, it is fine tuned
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

learn.unfreeze()
learn.fit_one_cycle(10, slice(2e-3/100, 2e-3))

exit()



data_clas = TextClasDataBunch.from_ids(dataset_location, train_df=df_train, valid_df=df_dev, test_df=df_test, vocab=data_lm.train_ds.vocab, bs=bs)
