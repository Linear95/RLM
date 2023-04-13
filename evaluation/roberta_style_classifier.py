
from transformers import RobertaTokenizer, RobertaModel, RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

class StyleClassifier(RobertaPreTrainedModel):

    def __init__(self, config, style_num):

        super().__init__(config)

        self.style_num = style_num

        self.config = config
        self.config.num_labels = style_num

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(self.config)

    def forward(
        self,
        input_ids=None,
        input_styles=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        target_styles=None,
        target_labels=None,
        contextual_attn_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
        ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if target_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.style_num), target_labels.view(-1))

        return {
            "logits": logits,
            "loss": loss
        }
