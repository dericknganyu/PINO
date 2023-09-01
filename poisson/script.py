#runnung fno.py for different reolutions on different gpus

import subprocess

resolutions = [128, 256, 512]
gpu_infos = [0, 1, 2]

for res, gpu_in in zip(resolutions, gpu_infos):

    screen_name = 'inv_pino_poisson_'+str(res)
    command =  'python inv-pino.py --res %s'%(res)
    subprocess.run('screen -dmS '+screen_name, shell=True)
    
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'conda activate base'), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'export CUDA_VISIBLE_DEVICES=%s'%(gpu_in)), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, command), shell=True)