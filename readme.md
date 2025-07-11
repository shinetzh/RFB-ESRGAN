1. runing environment
   packages should be install:
		python3
		pytorch
		tqdm
		opencv-python
		lmdb
		pyyaml
		
   if still cannot run the scripts, please check the requirements.txt
   the script run on GPU 0, please Guarantee the GPU 0 has enough memory for run the scripts. 
   if you need use anathor GPU to run the scipts, please change the GPU number 'gpu_ids' in './options/test/config.yml'
   
2. how to run the script?
   before run the scripts, please replace the './NTIRE2020_testLR' as the absolute path of your test images dir.
   
   python3 test.py --test_images ./NTIRE2020_testLR


3. the result is in dir './result_images', the full-resolution result images is in dir './full_resolution_images'