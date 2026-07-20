import torch


class KV_Cache:
    """
    A class representing the KV Cache.
    """

    def __init__(
        self,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        prefill_bsz: int,
        num_gpus: int,
        model_size: int
    ) -> None:
        """ Initializes the KV Cache.
        Args:
            layer_num (int)
            batch_size (int)
            num_key_value_heads (int)
            num_heads (int)
            max_length (int)
            head_dim (int)
            dtype (torch.dtype)
            layer_mapping (dict)
            prefill_bsz (int)
            num_gpus (int)
            model_size (int)
        """

        self.layer_num = layer_num
        self.batch_size = batch_size
        self.max_length = max_length
        self.kv_head = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.layer_mapping = layer_mapping
        self.prefill_bsz = min(prefill_bsz, batch_size)
        self.context = 0

        # estimate free GPU memory when prefilling
        self.num_gpus = num_gpus
        self.model_size = model_size
        self.device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(":")[-1]))
        self.total_memory_by_device = {
            dev: torch.cuda.get_device_properties(torch.device(dev)).total_memory / 1024 / 1024 / 1024
            for dev in self.device_list
        }
        layers_by_device = {dev: 0 for dev in self.device_list}
        for ldx in range(self.layer_num):
            layers_by_device[self.layer_mapping[str(ldx)]] += 1
        # Rough per-device model weight estimate kept only for admission
        # heuristics. Actual allocation checks below should use mem_get_info().
        self.model_weight_consumption_by_device = {
            dev: (self.model_size * 2) * (layers_by_device[dev] / max(self.layer_num, 1))
            for dev in self.device_list
        }
        # prefill consumption for single GPU
        prefill_consumption = self.prefill_bsz * self.num_heads * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024 # hidden
        prefill_consumption += self.prefill_bsz * self.num_heads * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024 # residual
        prefill_consumption += self.prefill_bsz * (self.num_heads + 2*self.kv_head) * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024 # qkv
        prefill_consumption += 4 * self.prefill_bsz * self.num_heads * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024 # temp
        self.prefill_consumption_by_device = {
            dev: prefill_consumption if layers_by_device[dev] > 0 else 0.0
            for dev in self.device_list
        }
        self.free_memory_by_device = {
            dev: self.total_memory_by_device[dev]
            - self.model_weight_consumption_by_device[dev]
            - self.prefill_consumption_by_device[dev]
            for dev in self.device_list
        }
        self.free_memory = min(self.free_memory_by_device.values()) if self.free_memory_by_device else 0.0
