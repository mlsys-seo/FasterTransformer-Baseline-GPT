from dataclasses import dataclass, field
from math import ceil
import enum
from .decoder_estimator import estimate_decoder_batch
from scipy.stats import poisson

class TPType(enum.Enum):
    none = 1
    tp2 = 2
    tp4 = 3
    tp8 = 4

def get_TPType(num_tp):
    if num_tp == 1:
        return TPType.none
    elif num_tp == 2:
        return TPType.tp2
    elif num_tp == 4:
        return TPType.tp4
    elif num_tp == 8:
        return TPType.tp8

@dataclass
class GPUPart:
    encoder: int = 0
    decoder: int = 0
    
    tp_type: TPType = TPType.none
    
    def __post_init__(self) -> None:
        assert self.encoder >= 0 and self.decoder >= 0 and (self.encoder + self.decoder) > 0, \
        f"layer count must be positive and have at least 1 layer got encoder: {self.encoder} decoder: {self.decoder}"
        
        
    def is_encoder(self) -> bool:
        return self.encoder > 0
    
    def is_decoder(self) -> bool:
        return self.decoder > 0
    
    def is_encoder_decoder(self) -> bool:
        return self.is_encoder() and self.is_decoder()
    
    def is_encoder_only(self) -> bool:
        return self.is_encoder() and not self.is_decoder()
    
    def __add__(self, other):
        if isinstance(other, GPUPart):
            self.encoder += other.encoder
            self.decoder += other.decoder
        return self
    

@dataclass
class PerfEstim:
    latency: float = 0.
    throughput: float = 0.
    
    def __post_init__(self) -> None:
        self.throughput = self.throughput * 1000
        
    
    
@dataclass
class config:
    gpu_topo: list = field(default_factory=list)
    
    n_gpus: int = 0
    n_vgpus: int = field(init=False, default=0)
    n_layers: int = 0
    n_decoder_seq_len: int = 0
    encoder_batch_size: int = 0
    
    # decoder_batch_size: int = field(init=False, default=0)
    decoder_batch_size: int = 0
    decoder_mb_count: int = 1
    
    tp_gpu_num: int = 0
    
    def __post_init__(self) -> None:
        if self.decoder_batch_size == 0:
            self.decoder_batch_size = self.encoder_batch_size
        self.n_vgpus = self.n_gpus
        
        for _ in range(self.tp_gpu_num):
            self.n_vgpus -= 1
                
    def is_encoder_only(self) -> bool:
        return self.gpu_topo[0].is_encoder_only()
                
        
    def is_tp(self) -> bool:
        return self.tp_gpu_num != 0
    
    
    def is_encoder_decoder_combined(self) -> bool:
        return self.get_encoder_device_count() + self.get_decoder_device_count() > self.n_vgpus
        
        
    def get_decoder_device_count(self) -> int:
        count = 0
        
        for node in self.gpu_topo:
            if node.is_decoder():
                count += 1
        
        return count
    
    
    def get_encoder_device_count(self) -> int:
        count = 0
        
        for node in self.gpu_topo:
            if node.is_encoder():
                count += 1
        
        return count


@dataclass
class c1_config(config):    
    def __post_init__(self) -> None:
        super().__post_init__()
        # assert (self.encoder_batch_size * self.n_decoder_seq_len) % self.decoder_batch_slice_degree == 0, "Decoder Batch must deviable by decoder batch slice degree"
        
        # self.encoder_batch_list = [self.encoder_batch_size // self.decoder_mb_count for _ in range(self.decoder_mb_count)]
        # for idx in range(self.encoder_batch_size % self.decoder_mb_count):
        #     self.encoder_batch_list[idx] += 1
        
        self.encoder_sliced_batch_size = ceil(self.encoder_batch_size / self.decoder_mb_count) if self.decoder_mb_count != 0 else self.encoder_batch_size
        
        # self.decoder_batch_list = [(self.decoder_batch_size * self.n_decoder_seq_len) // self.decoder_mb_count for _ in range(self.decoder_mb_count)]
        # for idx in range((self.decoder_batch_size * self.n_decoder_seq_len) % self.decoder_mb_count):
        #     self.decoder_batch_list[idx] += 1
        # print(self.encoder_batch_size * self.n_decoder_seq_len)
        
        self.decoder_batch_size = ceil((self.encoder_batch_size * 32) / self.decoder_mb_count) if self.decoder_mb_count != 0 else self.encoder_batch_size * self.n_decoder_seq_len


@dataclass
class c2_config(config):  
    decoder_seq_lamb: int = 0
    encoder_exec_freq: int = 0
    mb: int = 1
   
    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_decoder_seq_len = int(poisson.ppf(0.99, mu=self.decoder_seq_lamb)) - 1
        
        if self.encoder_exec_freq == 0:
            self.n_decoder_step = [self.n_decoder_seq_len]
        else:
            self.n_decoder_step = [self.n_decoder_seq_len // (self.encoder_exec_freq + 1) for _ in range(self.encoder_exec_freq + 1)]
            for idx in range(self.n_decoder_seq_len % (self.encoder_exec_freq + 1)):
                self.n_decoder_step[idx] += 1
        
        self.decoder_batch_list, self.encoder_batch_size, self.embd_batch_list = estimate_decoder_batch(self.decoder_seq_lamb, self.decoder_batch_size, self.decoder_seq_lamb*2, self.n_decoder_step[0])
