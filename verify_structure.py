import os

print("="*70)
print(" VANILLA GAN PROJECT - FILE STRUCTURE VERIFICATION")
print("="*70)

files = {
    'Core Configuration': [
        'config.yaml',
        'requirements.txt',
    ],
    'Main Modules': [
        'data_loader.py',
        'generator.py', 
        'discriminator.py',
        'vanilla_gan.py',
        'train.py',
        'evaluation.py',
        'inference.py',
    ],
    'User Interface': [
        'app.py',
    ],
    'Utils (created by smoke_test.py)': [
        'utils/__init__.py',
        'utils/visualizer.py',
        'utils/metrics.py',
        'utils/logger.py',
    ],
    'Documentation': [
        'README.md',
        'SETUP_STATUS.txt',
    ],
    'Testing & Setup': [
        'smoke_test.py',
    ],
    'Deployment': [
        'Dockerfile',
    ]
}

total = 0
ready = 0

for category, file_list in files.items():
    print(f"\n{category}:")
    print("-" * 70)
    for f in file_list:
        exists = os.path.exists(f)
        status = "✓" if exists else "○"
        print(f"  {status} {f}")
        total += 1
        if exists:
            ready += 1

print("\n" + "="*70)
print(f" STATUS: {ready}/{total} files present")
if ready < total:
    print(f" → Run 'python smoke_test.py' to create remaining files")
else:
    print(f" → All files ready!")
print("="*70)
