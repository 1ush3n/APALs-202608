import time
import logging
from datetime import datetime
from pathlib import Path
from configs import configs

def init_logger(args, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = getattr(args, 'data_path', None) or getattr(configs, 'data_file_path', 'default')
    data_basename = Path(data_path).stem
    output_root = (
        getattr(args, 'output_dir', None)
        or getattr(args, 'result_dir', None)
        or getattr(configs, 'result_dir', 'results')
    )
    exp_dir = Path(output_root) / f"{experiment_name}_{data_basename}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = exp_dir / f"{experiment_name}.log"
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"========== {experiment_name} 实验开始 ==========")
    logger.info(f"实验参数: {vars(args) if hasattr(args, '__dict__') else args}")
    logger.info(f"结果将归档至: {exp_dir}")
    
    return logger, exp_dir

def record_experiment_time(logger, start_time):
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    logger.info(f"实验结束。总耗时: {hours}小时{minutes}分钟{seconds}秒")
    return elapsed
