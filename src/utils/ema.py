import torch


def _canonical_name(name: str) -> str:
    # torch.compile and DDP may be nested in either order. Strip both wrappers
    # repeatedly so a rank-0 EMA can shadow the underlying model parameters.
    previous = None
    while name != previous:
        previous = name
        name = name.removeprefix("module.").removeprefix("_orig_mod.")
    return name


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {}
        for raw_name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[_canonical_name(raw_name)] = param.detach().clone()

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            pairs = [
                (self.shadow[_canonical_name(raw_name)], param.detach())
                for raw_name, param in model.named_parameters()
                if _canonical_name(raw_name) in self.shadow
            ]
            if not pairs:
                return
            shadow, current = zip(*pairs)
            torch._foreach_mul_(shadow, self.decay)
            torch._foreach_add_(shadow, current, alpha=1.0 - self.decay)

    def apply(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {}
        with torch.no_grad():
            for raw_name, param in model.named_parameters():
                name = _canonical_name(raw_name)
                if name in self.shadow:
                    backup[name] = param.detach().to(device="cpu", copy=True)
                    param.copy_(self.shadow[name])
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for raw_name, param in model.named_parameters():
                name = _canonical_name(raw_name)
                if name in backup:
                    param.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}
