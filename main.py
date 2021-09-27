import time
import datetime
import os
import yaml
import argparse
import shutil

import utils.comm as comm



parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='./config.yml')
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
                gpu_num: 4,
                memory: 256,
                script: train.sh,
                start_time: 2021.08.05-22:30,
                waiting_time: 163:30:25,
            },
            job_id2: {
                ...
            }
            ...
        }'''
        

        '''
        running_job_dict = {
            job_id1: {
                gpu_num: 2,
                ip: 10.24.82.22,
                gpu_ids: [0, 4]
                start_time: 2021.08.05-22:30
                total_time: 163:30:25,
            },
            job_id2: {
                ...
            }
        }'''
        self.running_job_dict = None

        while True:
            self.init_update_info(config_file)
            self.send_jobs()
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
                username = machine_info['username'] if 'username' in machine_info else default_info['username']
                password = machine_info['password'] if 'password' in machine_info else default_info['password']
                pkey_dir = machine_info['pkey_dir'] if 'pkey' in machine_info else default_info['pkey_dir']

                self.machines_dict[ip] = {}
                self.machines_dict[ip]['name'] = name
                self.machines_dict[ip]['port'] = port
                self.machines_dict[ip]['username'] = username
                self.machines_dict[ip]['password'] = password
                self.machines_dict[ip]['pkey_dir'] = pkey_dir
                self.machines_dict[ip]['ssh'] = comm.SSH(ip, port, username, password, pkey_dir)
                self.machines_dict[ip]['memory'] = self.query_memory(self.machines_dict[ip]['ssh'])

            self.machines_dict[ip]['gpu_ids'] = self.query_gpu_ids(self.machines_dict[ip]['ssh'])

        logger.info(self.machines_dict)

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
            self.waiting_jobs_dict[job_dir]['gpu_num'] = job_config['gpu_num']
            self.waiting_jobs_dict[job_dir]['memory'] = job_config['memory']
            self.waiting_jobs_dict[job_dir]['script'] = job_config['script']
        logger.info(self.waiting_jobs_dict)
        # exit()
        
    def send_jobs(self):
        for ip in self.machines_dict:
            for job_dir in self.waiting_jobs_dict:
                machine_gpu_ids = self.machines_dict[ip]['gpu_ids']
                machine_gpu_num = len(self.machines_dict[ip]['gpu_ids'])
                machine_memory = self.machines_dict[ip]['memory']
                job_gpu_num = self.waiting_jobs_dict[job_dir]['gpu_num']
                job_memory = self.waiting_jobs_dict[job_dir]['memory']
                if machine_gpu_num >= job_gpu_num and machine_memory >= job_memory:
                    now = datetime.datetime.now().strftime("%m%d-%H:%M:%S")
                    cmd = ''
                    print(machine_gpu_ids)
                    visible_gpu_ids_str = ','.join(str(i) for i in machine_gpu_ids)
                    cmd += f'export CUDA_VISIBLE_DEVICES={visible_gpu_ids_str};'
                    print(cmd)




                    src = os.path.join(self.waiting_jobs_dir, job_dir)

                    if 'copy' in job_dir:
                        dst = os.path.join(self.sended_jobs_dir, 'copy', f"{now}_{self.machines_dict[ip]['name']}_"+job_dir)
                        shutil.copytree(src, dst)
                    elif 'ctn' in job_dir:
                        continue
                    else:
                        dst = os.path.join(self.sended_jobs_dir, f"{now}_{self.machines_dict[ip]['name']}_"+job_dir)
                        shutil.move(src, dst)
                    # if 'copy' in src:
                    #     shutil.copytree(src, dst)
                    # else:
                    #     shutil.move(src, dst)
                    script_dir = os.path.join(dst, self.waiting_jobs_dict[job_dir]['script'])
                    cmd += f"cd '{os.path.dirname(script_dir)}';\
                            nohup bash '{script_dir}' > nohup.log 2>&1 &;"
                    stdout = self.machines_dict[ip]['ssh'].exec_cmd(cmd)
                    del self.waiting_jobs_dict[job_dir]
                    return




    def query_gpu_ids(self, ssh) -> list:
        def parse(line, qargs):
            numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']    # 可计数的参数
            power_manage_enable=lambda v:(not 'Not Support' in v)   # lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
            to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))     # 带单位字符串去掉单位
            process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
            return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}
        qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = ssh.exec_cmd(cmd).split('\n')
        gpu_info_list = [parse(line,qargs) for line in results]
        gpu_ids = []
        for gpu_info in gpu_info_list:
            if gpu_info['memory.free'] / gpu_info['memory.total'] > 0.9:
                gpu_ids.append(gpu_info['index'])
        return gpu_ids


    def query_memory(self, ssh) -> float:
        if comm.is_docker():
            cmd = 'cat /sys/fs/cgroup/memory/memory.limit_in_bytes'
        else:
            raise NotImplementedError
        stdout= ssh.exec_cmd(cmd)
        return float(stdout) / 1024**3


gpu_manager = GpuManager(args.config_file)