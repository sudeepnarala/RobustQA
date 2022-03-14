from transformers import DistilBertForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.distilbert.modeling_distilbert import TransformerBlock
import math

class DistilBertForQuestionAnsweringScaledPenalty(DistilBertForQuestionAnswering):
    # Forward method for meta learning
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)

        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            # start_loss = loss_fct(start_logits, start_positions)
            # end_loss = loss_fct(end_logits, end_positions)
            start_loss = self.loss_fct(start_logits, start_positions)
            end_loss = self.loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    def loss_fct(self, logits, ground_truth):
        """
        logits: (bs, max_query_len)
        ground_truth: (bs)
        """
        # import pdb; pdb.set_trace()
        ground_truth_broadcast = ground_truth.unsqueeze(1)
        scaling = torch.arange(logits.shape[-1]).unsqueeze(0).repeat(ground_truth.shape[0], 1).type(torch.float).to(self.device)
        scaling -= ground_truth_broadcast
        # scaling = torch.sigmoid(scaling)
        scaling /= 4
        scaling = 2 * torch.abs(torch.sigmoid(scaling) - 0.5)
        loss = -(scaling * torch.log(1-F.softmax(logits)))
        # Special case, [CLS] token
        # Just do regular if [CLS]
        loss = loss.sum(dim=-1)
        loss = loss*(ground_truth != 0)
        loss = loss.sum(dim=0)
        loss_fct = nn.CrossEntropyLoss()
        og_loss = loss_fct(logits, ground_truth)
        return loss + og_loss

