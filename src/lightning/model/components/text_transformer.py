import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DefaultModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 1,
        gradient_checkpointing_enable: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.model_config)
        else:
            self.model = AutoModel.from_config(config=self.model_config)

        if gradient_checkpointing_enable:
            self.model.gradient_checkpointing_enable()

        self.out = nn.Linear(
            in_features=self.model_config.hidden_size,
            out_features=num_labels,
        )
        self._init_weights(self.out)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        feature = outputs[0][:, 0, :]
        return feature

    def forward(self, batch: dict) -> torch.Tensor:
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        output = self.out(feature)
        return output
