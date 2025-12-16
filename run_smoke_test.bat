@echo off
python setup_utils.py
python -c "import generator; g = generator.build_generator(); print('Generator OK')"
python -c "import discriminator; d = discriminator.build_discriminator(); print('Discriminator OK')"
python -c "import vanilla_gan; gan = vanilla_gan.VanillaGAN(); print('VanillaGAN OK')"
echo Smoke tests complete
