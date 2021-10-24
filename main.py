import time
import datetime
import os
import yaml
import argparse
import shutil
import psutil
import random
import sys

import utils.comm as comm



parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='./runner_config.yml')
args = parser.parse_args()


with open(args.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
sleep = config['default_info']['sleep']
log_dir = config['default_info']['log_dir']

logger = comm.get_logger(os.path.join(log_dir, 'ssh_gpu.log'))



class GpuManager():
    def __init__(self, config_file) -> None:
        '''
        machines_dict = {
            ip1: {
                port: 22,
                username: root,
                password: 123456,
                pkey: '/root/.ssh/id_rsa',
                ssh: ssh_obj,
                sftp: sftp_obj,
                memory: 256     # G
                gpu_ids: [0, 2, 4],
            },
            ip2: {
                ...
            },
            ...
        }'''
        self.machines_dict = {}
        
        '''
        waiting_jobs_dict = {
            job_dir: {
                job_gpu_num: 4,
                job_memory: 256,
                job_port: True,
                script: train.sh,
            },
            job_id2: {
                ...
            }
            ...
        }'''
        self.running_job_dict = None

        while True:
            self.init_update_info(config_file)
            self.send_jobs()
            now = datetime.datetime.now().strftime("%m/%d %H:%M:%S")

            print(f'{now}, sleep {sleep}s...')
            time.sleep(sleep)


    def init_update_info(self, config_file):
        with open(config_file, 'r') as f:
            try:
                config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                logger.info('config_file error...')
                return

        default_info = config['default_info']

        self.machines_dict = {}
        for machine_info in config['machines_info']:
            ip = machine_info['ip']
            if ip not in self.machines_dict:
                name = machine_info['name'] if 'name' in machine_info else None
                port = machine_info['port'] if 'port' in machine_info else default_info['port']
                pre_cmd = machine_info['pre_cmd'] if 'pre_cmd' in machine_info else default_info['pre_cmd']
                username = machine_info['username'] if 'username' in machine_info else default_info['username']
                password = machine_info['password'] if 'password' in machine_info else default_info['password']
                pkey_dir = machine_info['pkey_dir'] if 'pkey' in machine_info else default_info['pkey_dir']

                self.machines_dict[ip] = {}
                self.machines_dict[ip]['name'] = name
                self.machines_dict[ip]['port'] = port
                self.machines_dict[ip]['pre_cmd'] = pre_cmd
                self.machines_dict[ip]['username'] = username
                self.machines_dict[ip]['password'] = password
                self.machines_dict[ip]['pkey_dir'] = pkey_dir
                self.machines_dict[ip]['ssh'] = comm.SSH(ip, port, username, password, pkey_dir)
                self.machines_dict[ip]['memory'] = comm.query_memory(self.machines_dict[ip]['ssh'])

        # logger.info(self.machines_dict)

        date_fmt = "%y-%m-%d %H:%M:%S"
        self.waiting_jobs_dir = default_info['waiting_jobs_dir']
        self.sended_jobs_dir = default_info['sended_jobs_dir']
        self.waiting_jobs_dict = {}

        for job_dir in sorted(os.listdir(self.waiting_jobs_dir)):
            if os.path.isdir(os.path.join(self.waiting_jobs_dir, job_dir)):
                if job_dir not in self.waiting_jobs_dict:
                    start_time = datetime.datetime.now().strftime(date_fmt)
                    self.waiting_jobs_dict[job_dir] = {'start_time': start_time}
        for job_dir in self.waiting_jobs_dict:
            start_time_dt = datetime.datetime.strptime(self.waiting_jobs_dict[job_dir]['start_time'], date_fmt)
            self.waiting_jobs_dict[job_dir]['waiting_time'] = round((datetime.datetime.now() - start_time_dt).seconds / 3600, 4)
            with open(os.path.join(self.waiting_jobs_dir, job_dir, 'job_config.yml'), 'r') as f:
                try:
                    job_config = yaml.load(f, Loader=yaml.FullLoader)
                except:
                    logger.info('config_file error...')
                    return
            self.waiting_jobs_dict[job_dir]['job_gpu_num'] = job_config['job_gpu_num']
            self.waiting_jobs_dict[job_dir]['job_memory'] = job_config['job_memory']
            self.waiting_jobs_dict[job_dir]['job_per_gpu_memory'] = job_config['job_per_gpu_memory']
            self.waiting_jobs_dict[job_dir]['job_port'] = job_config['job_port']
            self.waiting_jobs_dict[job_dir]['script'] = job_config['script']
            self.waiting_jobs_dict[job_dir]['send_src'] = job_config['send_src']
            self.waiting_jobs_dict[job_dir]['rm_src'] = job_config['rm_src']
        # logger.info(self.waiting_jobs_dict)
        # exit()
        
    def send_jobs(self):
        for ip in self.machines_dict:
            try:
                for job_dir in self.waiting_jobs_dict:
                    job_gpu_num = self.waiting_jobs_dict[job_dir]['job_gpu_num']
                    job_memory = self.waiting_jobs_dict[job_dir]['job_memory']
                    job_per_gpu_memory = self.waiting_jobs_dict[job_dir]['job_per_gpu_memory']

                    machine_empty_gpu_ids = comm.query_empty_gpu_ids(self.machines_dict[ip]['ssh'], job_per_gpu_memory)
                    machine_memory = self.machines_dict[ip]['memory']

                    if not self.waiting_jobs_dict[job_dir]['send_src']:
                        continue

                    if len(machine_empty_gpu_ids) >= job_gpu_num and machine_memory >= job_memory:
                        now = datetime.datetime.now().strftime("%m%d-%H:%M:%S")
                        cmd = f"{self.machines_dict[ip]['pre_cmd']};" if self.machines_dict[ip]['pre_cmd'] else ''
                        visible_gpu_ids_str = ','.join(str(i) for i in machine_empty_gpu_ids[:job_gpu_num])
                        cmd += f'export CUDA_VISIBLE_DEVICES={visible_gpu_ids_str};'

                        if self.waiting_jobs_dict[job_dir]['job_port']:
                            while True:
                                job_port = random.randint(20000, 60000)
                                if not comm.is_port_in_use(job_port, ip):
                                    break
                            cmd += f'export JOB_PORT={job_port};'
                        else:
                            job_port = None
                        
                        src = os.path.join(self.waiting_jobs_dir, job_dir)

                        dst = os.path.join(self.sended_jobs_dir, f"{now}_{self.machines_dict[ip]['name']}_"+job_dir)
                        self.machines_dict[ip]['ssh'].sftp_put_dir(src, dst)
                        if self.waiting_jobs_dict[job_dir]['rm_src']:
                            shutil.rmtree(src)
                        script_dir = os.path.join(dst, self.waiting_jobs_dict[job_dir]['script'])
                        cmd += f"cd '{os.path.dirname(script_dir)}';"
                        cmd += f"nohup bash '{script_dir}' > nohup.log 2>&1 &"
                        stdout = self.machines_dict[ip]['ssh'].exec_cmd(cmd)
                        del self.waiting_jobs_dict[job_dir]

                        logger.info(f"Sended: ip:{ip}, name:{self.machines_dict[ip]['name']}, job_dir:{job_dir}, CUDA_VISIBLE_DEVICES:{visible_gpu_ids_str}, JOB_PORT:{job_port}")
                        return
            except:
                # print("Unexpected error:", sys.exc_info())
                logger.info(f'Login:{ip} failed, try next ip...')
                continue









gpu_manager = GpuManager(args.config_file)
