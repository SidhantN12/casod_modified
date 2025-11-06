# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import gc
import psutil
import threading

import torch


def byte2gb(x):
    return int(x / 2**30)


# This context manager is used to track the peak memory usage of the process
class MemoryTrace:
    def __enter__(self):
        gc.collect()
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # Guard against missing peak reset symbols on some builds
                try:
                    torch.cuda.reset_max_memory_allocated()
                except Exception:
                    pass
                try:
                    self.begin = byte2gb(torch.cuda.memory_allocated())
                except Exception:
                    self.begin = 0
            else:
                self.begin = 0
        except Exception:
            self.begin = 0
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    self.end = byte2gb(torch.cuda.memory_allocated())
                except Exception:
                    self.end = 0
                try:
                    self.peak = byte2gb(torch.cuda.max_memory_allocated())
                except Exception:
                    self.peak = 0
                try:
                    cuda_info = torch.cuda.memory_stats()
                    self.peak_active_gb = byte2gb(cuda_info.get("active_bytes.all.peak", 0))
                    self.cuda_malloc_retires = cuda_info.get("num_alloc_retries", 0)
                    self.m_cuda_ooms = cuda_info.get("num_ooms", 0)
                except Exception:
                    self.peak_active_gb = 0
                    self.cuda_malloc_retires = 0
                    self.m_cuda_ooms = 0
                self.used = byte2gb(self.end - self.begin)
                self.peaked = byte2gb(self.peak - self.begin)
                try:
                    self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())
                except Exception:
                    self.max_reserved = 0
            else:
                self.end = self.peak = self.used = self.peaked = self.max_reserved = 0
                self.peak_active_gb = 0
                self.cuda_malloc_retires = 0
                self.m_cuda_ooms = 0
        except Exception:
            self.end = self.peak = self.used = self.peaked = self.max_reserved = 0
            self.peak_active_gb = 0
            self.cuda_malloc_retires = 0
            self.m_cuda_ooms = 0

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")
