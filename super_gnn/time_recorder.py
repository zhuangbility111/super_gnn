import torch
import numpy as np
import torch.distributed as dist
import time
import pandas as pd
from collections import defaultdict

class TimeRecorder(object):
    def __init__(self, num_layer, num_epoch) -> None:
        self.mode = None  # 当前模式，例如 'training' 或 'validation'
        self.current_epoch = None  # 当前迭代号
        self.records = defaultdict(lambda: defaultdict(list))
        TimeRecorder.ctx = self

    def set_mode(self, mode):
        """设置当前模式"""
        self.mode = mode
    
    def set_epoch(self, epoch):
        """设置当前迭代号"""
        self.current_epoch = epoch

    def time_block(self, module_name, is_record=True):
        """上下文管理器，用于测量代码段耗时"""
        class TimerContext:
            def __init__(self, outer):
                self.outer = outer
                self.module_name = module_name
                self.start_time = None
                self.is_record = is_record

            def __enter__(self):
                if self.outer.mode == "training":  # 仅在模式为 'training' 时记录
                    self.start_time = time.perf_counter()

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.outer.mode == "training" and self.start_time is not None:
                    elapsed_time = (time.perf_counter() - self.start_time) * 1000.0  # 转换为毫秒
                    current_epoch = self.outer.current_epoch
                    if self.is_record:
                        self.outer.records[self.module_name][current_epoch].append(elapsed_time)
                    print(f"Module: {self.module_name}, Time: {elapsed_time:.6f} ms")

        return TimerContext(self)

    def report(self):
        """打印耗时统计结果"""
        for module_name, epochs in self.records.items():
            print(f"Module: {module_name}")
            for epoch, times in epochs.items():
                print(f"  epoch: {epoch}")
                print(f"    Count: {len(times)}")
                print(f"    Avg Time: {sum(times) / len(times):.6f} s")
                print(f"    Max Time: {max(times):.6f} s")
                print(f"    Min Time: {min(times):.6f} s")
            print()
    
    def get_stats(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        # convert the list of times to a tensor according to the predefined order of module_name and epoch
        times = []
        for module_name, epochs in self.records.items():
            for epoch, epoch_times in epochs.items():
                times.extend(epoch_times)
        # 2d tensor with shape (num_module * num_epoch, num_times)
        num_module = len(self.records)
        num_epoch = len(self.records[list(self.records.keys())[0]])
        times = torch.tensor(times, dtype=torch.float32).reshape(num_module * num_epoch, -1)

        """获取最小、最大、平均和标准差值"""
        # for module_name, epochs in self.records.items():
        #     for epoch, times in epochs.items():
        #         if len(times) == 0:
        #             continue
        min_time = torch.clone(times)
        max_time = torch.clone(times)
        avg_time = torch.clone(times)

        # 使用 torch.distributed 汇总结果
        if dist.is_initialized():
            dist.reduce(min_time, dst=0, op=dist.ReduceOp.MIN)
            dist.reduce(max_time, dst=0, op=dist.ReduceOp.MAX)
            dist.reduce(avg_time, dst=0, op=dist.ReduceOp.SUM)
            avg_time /= dist.get_world_size()
        
        # convert the min_time, max_time, avg_time to back to dict
        min_time_record, max_time_record, avg_time_record = None, None, None
        if rank == 0:
            min_time = min_time.tolist()
            max_time = max_time.tolist()
            avg_time = avg_time.tolist()

            min_time_record = defaultdict(lambda: defaultdict(list))
            max_time_record = defaultdict(lambda: defaultdict(list))
            avg_time_record = defaultdict(lambda: defaultdict(list))

            for module_name, epochs in self.records.items():
                for epoch, times in epochs.items():
                    if len(times) == 0:
                        continue
                    min_time_record[module_name][epoch] = min_time.pop(0)
                    max_time_record[module_name][epoch] = max_time.pop(0)
                    avg_time_record[module_name][epoch] = avg_time.pop(0)
            
        return min_time_record, max_time_record, avg_time_record
    
    def save_to_excel(self, file_path):
        rank = dist.get_rank() if dist.is_initialized() else 0
        min_time_record, max_time_record, avg_time_record = self.get_stats()
        if rank == 0:
            # get minimum, maximum and average acorss different processes with torch.distributed
            for (records, name) in [(min_time_record, "min_time"), (max_time_record, "max_time"), (avg_time_record, "avg_time")]:
                """将耗时记录保存为 Excel 文件，拆分不同值到不同的 sheet"""
                file_name = file_path + "_" + name + ".xlsx"
                # 创建一个 Pandas Excel writer 对象
                with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                    # 获取所有迭代号和模块名称
                    all_iterations = sorted(
                        {iteration for module_data in records.values() for iteration in module_data.keys()}
                    )
                    all_modules = records.keys()

                    num_values = 6  # 根据实际情况调整，如果是 5 个统计值

                    # 遍历每个模块
                    for value_index in range(num_values):
                        # 对于每个值，创建一个 sheet
                        data = []

                        # 填充数据：每个 iteration 对应一个模块的某个值
                        for iteration in all_iterations:
                            row = [records[module][iteration][value_index] if len(records[module][iteration]) > value_index else None for module in all_modules]
                            data.append(row)

                        # 创建 DataFrame 并写入到 Excel 的不同 sheet
                        df = pd.DataFrame(data, columns=all_modules, index=all_iterations)
                        sheet_name = f"Layer_{value_index+1}"
                        df.to_excel(writer, sheet_name=sheet_name)

                print(f"Results saved to {file_name}")
