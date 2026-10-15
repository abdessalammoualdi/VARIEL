#!/usr/bin/env python3
"""Remote-machine report. Python standard library only; no credentials collected."""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def command(args, timeout=15):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def collect(network=False, gpu_test=False):
    memory = {}
    p = Path('/proc/meminfo')
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith(('MemTotal:', 'MemAvailable:')):
                key, val, *_ = line.split()
                memory[key.rstrip(':') + '_GiB'] = round(int(val) / 1024**2, 2)
    cpu = command(['lscpu', '-J'])
    try:
        wanted = {'Architecture:', 'CPU(s):', 'Model name:', 'Core(s) per socket:', 'Socket(s):'}
        cpu = {r['field'].rstrip(':'): r['data'] for r in json.loads(cpu['stdout'])['lscpu'] if r['field'] in wanted}
    except (KeyError, ValueError):
        pass
    result = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'os': platform.platform(), 'architecture': platform.machine(),
        'python': sys.version, 'python_executable': sys.executable,
        'cwd': str(Path.cwd()), 'cpu': cpu,
        'available_cpu_threads': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
        'memory': memory, 'free_disk_GiB': round(shutil.disk_usage(Path.cwd()).free / 1024**3, 2),
        'python_candidates': {v: shutil.which(v) for v in ['python3.12', 'python3.11', 'python3.10', 'python3']},
        'gpu': command(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.free', '--format=csv,noheader']),
        'packages': {},
    }
    for name in ['pip', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'xgboost', 'xgboost-cpu', 'catboost', 'torch', 'rtdl-revisiting-models']:
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    if gpu_test:
        result['torch_kernel_test'] = command([sys.executable, '-c',
            "import torch,json; assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "x=torch.randn(256,256,device='cuda'); z=x@x; torch.cuda.synchronize(); "
            "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'device':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),"
            "'finite':bool(torch.isfinite(z).all())}))"], timeout=60)
    if network:
        result['network'] = {}
        urls = {'pypi': 'https://pypi.org/pypi/numpy/json',
                'pytorch': 'https://download.pytorch.org/whl/cu128/',
                'uci': 'https://archive.ics.uci.edu/dataset/222/bank+marketing',
                'openml': 'https://www.openml.org/api/v1/json/data/151'}
        for name, url in urls.items():
            try:
                with urlopen(Request(url, headers={'User-Agent': 'DTRA/0.1'}), timeout=12) as response:
                    response.read(64)
                    result['network'][name] = response.status
            except Exception as exc:
                result['network'][name] = str(exc)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', action='store_true')
    parser.add_argument('--gpu-test', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = collect(args.network, args.gpu_test)
    text = json.dumps(report, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n')
    print(text)
