def collate(batch: dict) -> dict:
    mask_len = int(batch["attention_mask"].sum(axis=1).max())
    for k, v in batch.items():
        if k not in ["labels", "targets"]:
            batch[k] = batch[k][:, :mask_len]
    return batch
