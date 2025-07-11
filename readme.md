# Perceptual Extreme Super Resolution Network with Receptive Field Block🚀

## <a href="https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Shang_Perceptual_Extreme_Super-Resolution_Network_With_Receptive_Field_Block_CVPRW_2020_paper.pdf" target="_blank">Paper</a>


# Runing Environment
   Packages should be install:
- python3
- pytorch
- tqdm
- opencv-python
- lmdb
- pyyaml
		
If still cannot run the scripts, please check the requirements.txt

The script run on GPU 0, please Guarantee the GPU 0 has enough memory for run the scripts. 

If you need use anathor GPU to run the scipts, please change the GPU number `gpu_ids` in `./options/test/config.yml`

# Pretrain Checkpoint
In `./pth` folder

# How to run the script?
   replace the `./NTIRE2020_testLR` as the absolute path of your test images dir.
   ```bash
   python3 test.py --test_images ./NTIRE2020_testLR
   ```

# Check Results
   the result is in dir `./result_images`, the full-resolution result images is in dir `./full_resolution_images`
