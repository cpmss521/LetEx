
import pandas as pd
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer



class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids',
                     'expl_attention_mask': 'attention_mask'}).to_dict('records')

        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)

        return {
            'pred': pred_features,
            'expl': expl_features,
        }


class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale



    def compute_loss(self, model, inputs, return_outputs=False):
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])
        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)

        if self.output_rationale and self.train_dataset is not None:
            expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        else:
            expl_outputs = pred_outputs

        loss = self.alpha * pred_outputs[0] + (1 - self.alpha) * expl_outputs[0]

        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )
