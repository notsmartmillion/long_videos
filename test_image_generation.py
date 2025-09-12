#!/usr/bin/env python3
"""
RTX 5080 SDXL Image Generation Test
Requires PyTorch 2.9+ nightly with CUDA 12.8 for RTX 5080 support
"""

import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path
import time

def test_image_generation():
    print("🚀 Testing SDXL Image Generation on RTX 5080...")
    
    # Strict RTX 5080 validation - NO CPU FALLBACK
    if not torch.cuda.is_available():
        print("💥 FATAL ERROR: CUDA not available!")
        print("RTX 5080 required - no CPU fallback allowed")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    if "RTX 5080" not in device_name:
        print(f"💥 FATAL ERROR: Expected RTX 5080, found {device_name}")
        print("This test is specifically for RTX 5080")
        return False
    
    device = torch.device("cuda")
    print(f"✅ RTX 5080 detected: {device_name}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA: {torch.version.cuda}")
    
    try:
        print("\n📥 Loading SDXL model (this will download ~7GB on first run)...")
        start_time = time.time()
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,  # RTX 5080 FP16 optimization
            use_safetensors=True,
            variant="fp16"
        )
        
        pipeline = pipeline.to(device)
        
        # Enable memory optimizations
        pipeline.enable_vae_slicing()
        pipeline.enable_vae_tiling()
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        # Test generation
        print("\n🎨 Generating test image...")
        prompt = "a majestic dragon flying over ancient mountains, epic fantasy art, dramatic lighting"
        
        start_time = time.time()
        image = pipeline(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024
        ).images[0]
        
        generation_time = time.time() - start_time
        print(f"✅ Image generated in {generation_time:.1f} seconds")
        
        # Save test image
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)
        
        image_path = output_dir / "test_sdxl_generation.png"
        image.save(image_path)
        
        print(f"💾 Test image saved to: {image_path}")
        print(f"📏 Image size: {image.size}")
        
        print("\n✅ SDXL Image Generation Test PASSED!")
        print("🎉 Your setup is ready for AI video generation!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("💡 This might be due to:")
        print("   - Insufficient GPU memory")
        print("   - Network issues during model download")
        print("   - Missing dependencies")
        return False

if __name__ == "__main__":
    test_image_generation()

