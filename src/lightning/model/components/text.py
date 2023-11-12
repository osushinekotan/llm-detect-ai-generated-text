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


class EmbeddingModel(nn.Module):
    def __init__(self, model_name: str, embedding_type: str = "cls"):
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.model_config)

        self.embedding_type = embedding_type

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.embedding_type == "mean":
            return mean_pooling(outputs, attention_mask)
        elif self.embedding_type == "max":
            return max_pooling(outputs, attention_mask)
        elif self.embedding_type == "last":
            return get_last_token_embedding(outputs)
        elif self.embedding_type == "cls":
            return get_cls_token_embedding(outputs)

        raise ValueError(f"Invalid output type: {self.embedding_type}")


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.sum(input_mask_expanded, 1)
    return sum_embeddings / sum_mask


def max_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    max_embeddings = torch.max(token_embeddings, 1)[0]
    return max_embeddings


def get_last_token_embedding(model_output: torch.Tensor) -> torch.Tensor:
    return model_output[0][:, -1, :]


def get_cls_token_embedding(model_output: torch.Tensor) -> torch.Tensor:
    return model_output[0][:, 0, :]
