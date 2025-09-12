#!/usr/bin/env python3
"""
Setup verification script for Long Video AI Automation System
Run this to verify your environment is properly configured
"""

import sys
import os
from pathlib import Path
import subprocess

def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_status(check, status, details=""):
    """Print status with emoji"""
    emoji = "✅" if status else "❌"
    print(f"{emoji} {check}")
    if details:
        print(f"   → {details}")

def check_python_version():
    """Check Python version"""
    print_header("PYTHON VERSION CHECK")
    
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    current = f"{version.major}.{version.minor}.{version.micro}"
    required = f"{required_major}.{required_minor}+"
    
    is_valid = version.major >= required_major and version.minor >= required_minor
    
    print_status(f"Python version: {current}", is_valid, 
                f"Required: {required}")
    
    return is_valid

def check_cuda_torch():
    """Check CUDA and PyTorch installation"""
    print_header("CUDA & PYTORCH CHECK")
    
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        print_status(f"PyTorch installed: {torch_version}", True)
        print_status(f"CUDA available", cuda_available)
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_version = torch.version.cuda
            
            print_status(f"GPU: {device_name}", True)
            print_status(f"VRAM: {memory_gb:.1f} GB", True)
            print_status(f"CUDA Version: {cuda_version}", True)
            
            # Check for RTX series
            is_rtx = "RTX" in device_name or "GeForce" in device_name
            print_status(f"RTX/GeForce GPU detected", is_rtx, 
                        "Optimized for RTX 5080" if is_rtx else "Performance may vary")
        else:
            print("⚠️  CUDA not available. GPU acceleration disabled.")
            print("   Install CUDA-enabled PyTorch:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        
        return True
        
    except ImportError:
        print_status("PyTorch installation", False, "Run: pip install torch")
        return False

def check_dependencies():
    """Check required dependencies"""
    print_header("DEPENDENCY CHECK")
    
    required_packages = [
        ('transformers', 'transformers'),
        ('diffusers', 'diffusers'), 
        ('TTS', 'TTS'),
        ('librosa', 'librosa'),
        ('PIL', 'Pillow'),
        ('rich', 'rich'),
        ('pydantic', 'pydantic'),
        ('yaml', 'PyYAML'),
        ('aiohttp', 'aiohttp'),
        ('soundfile', 'soundfile'),
        ('pydub', 'pydub')
    ]
    
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
            print_status(f"{package_name}", True)
        except ImportError:
            print_status(f"{package_name}", False, f"Run: pip install {pip_name}")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print(f"💡 Install all: pip install {' '.join(missing_packages)}")
    
    return len(missing_packages) == 0

def check_project_structure():
    """Check project directory structure"""
    print_header("PROJECT STRUCTURE CHECK")
    
    required_dirs = [
        'src/content_generation',
        'src/media_generation', 
        'src/utils',
        'configs',
        'data',
        'output',
        'temp',
        'logs'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'configs/config.yaml',
        'data/my_mythology_topics.yaml',
        'src/utils/config.py',
        'src/content_generation/content_pipeline.py',
        'src/media_generation/media_pipeline.py'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        print_status(f"Directory: {dir_path}", exists)
        if not exists:
            all_good = False
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   → Created directory: {dir_path}")
    
    # Check files
    for file_path in required_files:
        exists = Path(file_path).exists()
        print_status(f"File: {file_path}", exists)
        if not exists:
            all_good = False
    
    return all_good

def check_config_file():
    """Check configuration file"""
    print_header("CONFIGURATION CHECK")
    
    config_path = Path("configs/config.yaml")
    
    if not config_path.exists():
        print_status("Config file exists", False, "configs/config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['system', 'content', 'tts', 'image_generation', 'topics']
        
        for section in required_sections:
            exists = section in config
            print_status(f"Config section: {section}", exists)
        
        # Check if topics are defined
        topics_count = len(config.get('topics', {}))
        print_status(f"Topics configured: {topics_count}", topics_count > 0)
        
        return True
        
    except Exception as e:
        print_status("Config file validation", False, f"Error: {e}")
        return False

def check_topic_files():
    """Check topic files"""
    print_header("TOPIC FILES CHECK")
    
    topic_files = [
        'data/my_mythology_topics.yaml',
        'data/space_topics_example.yaml',
        'data/completed_topics.yaml'
    ]
    
    for file_path in topic_files:
        exists = Path(file_path).exists()
        print_status(f"Topic file: {file_path}", exists)
        
        if exists and file_path.endswith('my_mythology_topics.yaml'):
            try:
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                topics_count = len(data.get('topics', []))
                print_status(f"  → Topics defined: {topics_count}", topics_count > 0)
                
            except Exception as e:
                print_status(f"  → File validation", False, f"Error: {e}")

def run_quick_test():
    """Run a quick component test"""
    print_header("QUICK COMPONENT TEST")
    
    try:
        # Test config loading
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.utils.config import Config
        
        config = Config.load("configs/config.yaml")
        print_status("Config loading", True)
        
        # Test topic queue
        from src.content_generation.topic_queue import TopicQueue
        topic_queue = TopicQueue(config)
        
        queue_status = topic_queue.get_queue_status()
        pending_count = queue_status['pending_count']
        print_status(f"Topic queue: {pending_count} topics pending", pending_count > 0)
        
        return True
        
    except Exception as e:
        print_status("Component test", False, f"Error: {e}")
        return False

def generate_setup_summary():
    """Generate setup summary and next steps"""
    print_header("SETUP SUMMARY & NEXT STEPS")
    
    print("🎯 WHAT TO DO NEXT:")
    print()
    print("1. 📝 ADD YOUR TOPICS:")
    print("   → Edit 'data/my_mythology_topics.yaml'")
    print("   → Add 20-30 video topics you want to create")
    print()
    print("2. 🚀 TEST THE SYSTEM:")
    print("   → Run: python main.py --mode interactive")
    print("   → Choose option 4 to test components")
    print()
    print("3. 🎬 GENERATE YOUR FIRST VIDEO:")
    print("   → Run: python main.py --mode single --topic mythology --subtopic 'Zeus'")
    print("   → Watch the magic happen!")
    print()
    print("4. 🤖 SET UP AUTOMATION:")
    print("   → Fill your topic queue with 20+ topics")
    print("   → Run: python main.py --mode auto")
    print("   → Enjoy daily video generation!")
    print()
    print("📚 DOCUMENTATION:")
    print("   → README.md - Complete system overview")
    print("   → configs/config.yaml - System configuration")
    print("   → data/ - Add your topic files here")
    print()
    print("🔍 TROUBLESHOOTING:")
    print("   → Check logs/video_ai.log for errors")
    print("   → Run this script again to re-verify setup")
    print("   → Test individual components with option 4 in interactive mode")

def main():
    """Main setup check function"""
    print("🎬 Long Video AI Automation - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("CUDA & PyTorch", check_cuda_torch),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Configuration", check_config_file),
        ("Topic Files", check_topic_files),
        ("Component Test", run_quick_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print_header("VERIFICATION RESULTS")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\nOVERALL: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 SETUP COMPLETE! Your system is ready to generate videos!")
    elif passed >= total - 2:
        print("⚠️  MOSTLY READY! Fix remaining issues and you're good to go!")
    else:
        print("🔧 SETUP NEEDED! Please address the failed checks above.")
    
    generate_setup_summary()

if __name__ == "__main__":
    main()

