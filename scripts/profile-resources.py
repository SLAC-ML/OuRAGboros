#!/usr/bin/env python3

"""
Resource profiling script for OuRAGboros benchmarks.
Monitors CPU, memory, thread usage during concurrent load testing.
"""

import psutil
import time
import json
import sys
import argparse
import threading
import signal
from datetime import datetime
from typing import Dict, List
import subprocess
import os

class ResourceProfiler:
    def __init__(self, interval: float = 1.0, output_file: str = None):
        self.interval = interval
        self.output_file = output_file or f"resource_profile_{int(time.time())}.json"
        self.running = False
        self.data_points = []
        self.start_time = None
        self.process_name = "uvicorn"  # FastAPI process name
        
    def find_target_processes(self) -> List[psutil.Process]:
        """Find all processes related to OuRAGboros (uvicorn, python, etc.)"""
        target_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                name = proc.info['name'] or ''
                
                # Look for OuRAGboros-related processes
                if any(pattern in cmdline.lower() for pattern in [
                    'app_api:app', 'uvicorn', 'src.app_api', 
                    'streamlit run src/main.py', 'fastapi'
                ]) or (name.startswith('python') and 'rag' in cmdline.lower()):
                    target_processes.append(proc)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return target_processes
    
    def get_system_metrics(self) -> Dict:
        """Get system-wide resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(memory.total / (1024**3), 2),
            'memory_used_gb': round(memory.used / (1024**3), 2),
            'memory_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_total_gb': round(disk.total / (1024**3), 2),
            'disk_used_gb': round(disk.used / (1024**3), 2),
            'disk_percent': round((disk.used / disk.total) * 100, 2),
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv,
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    
    def get_process_metrics(self, processes: List[psutil.Process]) -> Dict:
        """Get metrics for target processes"""
        total_cpu = 0
        total_memory = 0
        total_threads = 0
        total_open_files = 0
        process_details = []
        
        for proc in processes:
            try:
                with proc.oneshot():
                    cpu_percent = proc.cpu_percent()
                    memory_info = proc.memory_info()
                    memory_mb = round(memory_info.rss / (1024**2), 2)
                    num_threads = proc.num_threads()
                    
                    # Try to get open files count
                    try:
                        open_files = len(proc.open_files())
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        open_files = 0
                    
                    process_details.append({
                        'pid': proc.pid,
                        'name': proc.name(),
                        'cpu_percent': round(cpu_percent, 2),
                        'memory_mb': memory_mb,
                        'num_threads': num_threads,
                        'open_files': open_files,
                        'status': proc.status()
                    })
                    
                    total_cpu += cpu_percent
                    total_memory += memory_mb
                    total_threads += num_threads
                    total_open_files += open_files
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process disappeared or access denied
                continue
        
        return {
            'process_count': len(process_details),
            'total_cpu_percent': round(total_cpu, 2),
            'total_memory_mb': round(total_memory, 2),
            'total_threads': total_threads,
            'total_open_files': total_open_files,
            'process_details': process_details
        }
    
    def collect_metrics(self):
        """Main metrics collection loop"""
        print(f"🔍 Starting resource profiling (interval: {self.interval}s)")
        print(f"📊 Output file: {self.output_file}")
        
        self.start_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                timestamp = datetime.fromtimestamp(current_time).isoformat()
                elapsed = round(current_time - self.start_time, 2)
                
                # Find target processes each time (in case they restart)
                target_processes = self.find_target_processes()
                
                # Collect system metrics
                system_metrics = self.get_system_metrics()
                
                # Collect process metrics
                process_metrics = self.get_process_metrics(target_processes)
                
                # Combine all metrics
                data_point = {
                    'timestamp': timestamp,
                    'elapsed_seconds': elapsed,
                    'system': system_metrics,
                    'processes': process_metrics
                }
                
                self.data_points.append(data_point)
                
                # Print progress
                print(f"⏱️  {elapsed:6.1f}s | CPU: {system_metrics['cpu_percent']:5.1f}% | "
                      f"RAM: {system_metrics['memory_used_gb']:5.2f}GB | "
                      f"Processes: {process_metrics['process_count']} | "
                      f"Threads: {process_metrics['total_threads']}")
                
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  Error collecting metrics: {e}")
                time.sleep(self.interval)
    
    def start_profiling(self):
        """Start profiling in background thread"""
        self.running = True
        self.profile_thread = threading.Thread(target=self.collect_metrics, daemon=True)
        self.profile_thread.start()
        return self.profile_thread
    
    def stop_profiling(self):
        """Stop profiling and save results"""
        self.running = False
        if hasattr(self, 'profile_thread'):
            self.profile_thread.join(timeout=5)
        
        # Save results
        if self.data_points:
            self.save_results()
            print(f"✅ Profiling complete. Saved {len(self.data_points)} data points to {self.output_file}")
        else:
            print("⚠️  No data points collected")
    
    def save_results(self):
        """Save collected data to JSON file"""
        results = {
            'metadata': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': round(time.time() - self.start_time, 2),
                'sample_interval_seconds': self.interval,
                'total_samples': len(self.data_points)
            },
            'data_points': self.data_points
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_summary(self):
        """Generate summary statistics"""
        if not self.data_points:
            return "No data collected"
        
        cpu_values = [dp['system']['cpu_percent'] for dp in self.data_points]
        memory_values = [dp['system']['memory_used_gb'] for dp in self.data_points]
        thread_values = [dp['processes']['total_threads'] for dp in self.data_points]
        
        summary = {
            'cpu_avg': round(sum(cpu_values) / len(cpu_values), 2),
            'cpu_max': max(cpu_values),
            'memory_avg_gb': round(sum(memory_values) / len(memory_values), 2),
            'memory_max_gb': max(memory_values),
            'threads_avg': round(sum(thread_values) / len(thread_values), 2),
            'threads_max': max(thread_values),
            'samples': len(self.data_points)
        }
        
        return summary

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal, stopping profiler...")
    global profiler
    if profiler:
        profiler.stop_profiling()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Profile system resources during OuRAGboros benchmarks')
    parser.add_argument('--interval', type=float, default=1.0, 
                       help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument('--output', type=str, 
                       help='Output JSON file (default: resource_profile_<timestamp>.json)')
    parser.add_argument('--duration', type=int,
                       help='Duration to profile in seconds (default: run until interrupted)')
    parser.add_argument('--command', type=str,
                       help='Command to run and profile (e.g., benchmark script)')
    
    args = parser.parse_args()
    
    global profiler
    profiler = ResourceProfiler(
        interval=args.interval,
        output_file=args.output
    )
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.command:
            # Start profiling, run command, then stop
            print(f"🚀 Starting profiler and running command: {args.command}")
            profiler.start_profiling()
            
            # Run the command
            result = subprocess.run(args.command, shell=True)
            
            profiler.stop_profiling()
            
            # Print summary
            summary = profiler.generate_summary()
            print(f"\n📈 Performance Summary:")
            print(f"   CPU Average: {summary['cpu_avg']}% (Max: {summary['cpu_max']}%)")
            print(f"   Memory Average: {summary['memory_avg_gb']}GB (Max: {summary['memory_max_gb']}GB)")
            print(f"   Threads Average: {summary['threads_avg']} (Max: {summary['threads_max']})")
            print(f"   Total Samples: {summary['samples']}")
            
            sys.exit(result.returncode)
            
        else:
            # Interactive mode
            print("🔍 Resource profiler started. Press Ctrl+C to stop.")
            
            if args.duration:
                print(f"⏱️  Will run for {args.duration} seconds")
                
            profiler.start_profiling()
            
            # Wait for specified duration or until interrupted
            if args.duration:
                time.sleep(args.duration)
                profiler.stop_profiling()
            else:
                # Wait forever until interrupted
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()