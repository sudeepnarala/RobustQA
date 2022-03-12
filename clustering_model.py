from transformers import DistilBertForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.distilbert.modeling_distilbert import TransformerBlock
import math

class ClusterModel(DistilBertForQuestionAnswering):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        assert("num_clusters" in kwargs)
        num_clusters = kwargs["num_clusters"]
        del kwargs["num_clusters"]
        model = super(ClusterModel, cls).from_pretrained(*args, **kwargs)
        # s.parallel = torch.nn.Linear(768, 2, bias=True)
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # s.parallel = s.parallel.to(device)
        def init_weights(module):
            if isinstance(module, nn.Linear):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        # model.cluster_keys = torch.nn.Linear(model.config.dim, 1, bias=False)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.cluster_keys = torch.randn(num_clusters, model.config.dim)
        model.cluster_keys = model.cluster_keys.to(device)
        model.cluster_transformers = torch.nn.ModuleList([TransformerBlock(model.config) for _ in range(2*num_clusters)])
        model.lin_query = torch.nn.Linear(model.config.dim, model.config.dim)
        # for module in model.cluster_transformers.modules():
        #     init_weights(module)
        # init_weights(model.cluster_keys)
        model.num_clusters = num_clusters
        torch.nn.init.xavier_uniform_(model.cluster_keys)
        torch.nn.init.xavier_uniform_(model.lin_query.weight)
        model.gamma = 0.2
        # s.forward = cls.forward
        return model

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
        # final_attention = distilbert_output[1][-1]
        # Compute cluster logits based on [CLS] token
        # queries = self.lin_query(hidden_states[:,0,:])
        # queries /= math.sqrt(self.config.dim)
        # cluster_logits = self.cluster_keys(queries)    # (bs, num_clusters)
        cluster_hidden_states = []
        cluster_queries = []
        for cluster_idx in range(self.num_clusters):
            cluster_hidden_state = self.cluster_transformers[cluster_idx*2](hidden_states, attn_mask=attention_mask, head_mask=head_mask)[0]   # (bs, max_query_len, dim)
            cluster_hidden_state = self.cluster_transformers[cluster_idx*2+1](cluster_hidden_state, attn_mask=attention_mask, head_mask=head_mask)[0]   # (bs, max_query_len, dim)
            queries = self.lin_query(cluster_hidden_state[:, 0, :])     # (bs, dim)
            queries = queries.unsqueeze(dim=1)
            cluster_queries.append(queries)
            cluster_hidden_state = torch.unsqueeze(cluster_hidden_state, dim=1)    # (bs, 1, max_query_len, dim)
            cluster_hidden_states.append(cluster_hidden_state)
        cluster_hidden_states = torch.cat(cluster_hidden_states, dim=1)     # (bs, num_clusters, max_query_len, dim)
        cluster_queries = torch.cat(cluster_queries, dim=1)                 # (bs, num_clusters, dim)
        cluster_queries /= math.sqrt(self.config.dim)
        cluster_logits = torch.sum(cluster_queries*self.cluster_keys, dim=-1)   # (bs, num_clusters, dim) * (num_clusters, dim) then (bs, num_clusters)
        cluster_coefficients = F.softmax(cluster_logits, dim=-1)       # (bs, num_clusters)
        # Penalize variance? --> i.e. (0.5, 0.5) is ok but (0.33, 0.2, 0.2, 0.1,....) is not
        # max_coeffs =
        # total_loss +=
        cluster_coefficients = cluster_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, cluster_hidden_states.shape[2], cluster_hidden_states.shape[3])
        hidden_states = cluster_coefficients*cluster_hidden_states
        hidden_states = hidden_states.sum(dim=1)/self.num_clusters   # Average
        # TODO: FILL IN
        final_hidden_states = self.dropout(hidden_states)


        # logits = F.linear(input=hidden_states, weight=weights, bias=None)  # (bs, max_query_len, 2)
        logits = self.qa_outputs(final_hidden_states)
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

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
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