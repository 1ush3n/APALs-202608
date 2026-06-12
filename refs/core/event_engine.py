import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Tuple

class EventType(Enum):
    TASK_FINISH = 1
    WORKER_RETURN = 2
    WORKER_LEAVE = 3
    STATION_BREAKDOWN = 4
    STATION_RECOVER = 5
    DURATION_PERTURB = 6
    MATERIAL_ARRIVE = 7

@dataclass
class Event:
    """
    仿真事件类
    Attributes:
        time (float): 事件发生的时间
        type (EventType): 事件类型
        data (Dict): 事件携带的数据 (如 task_id, worker_ids 等)
    """
    time: float
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)

class EventQueue:
    """
    基于 Priority Queue (heapq) 的事件管理引擎
    解决了浮点数时间碰撞与优先级不确定的隐患。
    """
    def __init__(self, max_size: int = 10000):
        # 存储元组: (safe_time, priority, seq, Event)
        self._queue: List[Tuple[float, int, int, Event]] = []
        self.max_size = max_size
        self._seq_counter = 0
        
    def push(self, event: Event):
        if len(self._queue) >= self.max_size:
            raise RuntimeError(f"EventQueue 超过最大容量限制 ({self.max_size})，可能存在死循环！")
        
        # 消除浮点数极微小误差，对齐到 5 位小数 (保证足够精度的同时避免 0.1+0.2 != 0.3)
        safe_time = round(event.time, 5)
        # 事件优先级：数值越小优先级越高 (TASK_FINISH(1) 优先于 WORKER_RETURN(2) 优先于 LEAVE(3))
        priority = event.type.value
        
        self._seq_counter += 1
        
        # 放入队列，heapq 会按顺序比较 safe_time, priority, _seq_counter
        heapq.heappush(self._queue, (safe_time, priority, self._seq_counter, event))
        
    def pop(self) -> Event:
        if not self._queue:
            raise IndexError("pop from empty EventQueue")
        return heapq.heappop(self._queue)[3]
        
    def peek(self) -> Event:
        if not self._queue:
            raise IndexError("peek from empty EventQueue")
        return self._queue[0][3]
        
    def is_empty(self) -> bool:
        return len(self._queue) == 0
        
    def clear(self):
        self._queue.clear()
        
    def __len__(self) -> int:
        return len(self._queue)

