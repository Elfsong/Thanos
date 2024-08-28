# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)

import optuna
from optuna.trial import TrialState

import hashlib
import subprocess, resource

class Measure(object):
    def __init__(self) -> None:
        pass
    
    def hash_code(self, text: str, length=12) -> str:
        text_bytes = text.encode('utf-8')
        sha256_hash = hashlib.sha256()
        sha256_hash.update(text_bytes)
        return sha256_hash.hexdigest()[:length]
    
    def system_call(self, command: str) -> str:
        try:
            usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd='/home/nus_cisco_wp1/Projects/Thanos/data')
            usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_time = usage_end.ru_utime - usage_start.ru_utime
        except subprocess.CalledProcessError as e:
            result = f"An error occurred while executing the command: {e}"
            cpu_time = -1
        return result, cpu_time
    
    def create(self, code, pass_sequence):
        raise NotImplementedError("Don't call the base class directly")

    def measure(self, code, pass_sequence):
        raise NotImplementedError("Don't call the base class directly")
    
class PerfMeasure(Measure):
    def create(self, code, pass_sequence):
        # Create Hash (code + pass_sequence)
        cid = self.hash_code(code + ''.join(pass_sequence))
        # cid = self.hash_code(code)
        
        # Create source code file
        with open(f"./data/{cid}.c", "w") as source:
            source.write(code)
        
        # Call Clang
        self.system_call(f'clang-18 -S -emit-llvm {cid}.c -o {cid}.ll')
        
        # Call opt
        pass_text = ' '.join(pass_sequence)
        self.system_call(f'opt-18 {cid}.ll {pass_text} -S -o {cid}.ll')
        
        # Execution
        self.system_call(f'llc-18 -filetype=obj -o {cid}.o {cid}.ll')
        self.system_call(f'clang-18 -o {cid} {cid}.o')
        
        return cid
    
    def measure(self, cid) -> float:
        # _, cpu_time = self.system_call(f'./{cid}')
        _, cpu_time = self.system_call(f'lli-18 {cid}.ll')
        
        return cpu_time