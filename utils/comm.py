import os
import stat
import paramiko
import logging
import time
import psutil
import socket
import operator


def is_docker():
    return os.path.exists('/.dockerenv')

def query_empty_gpu_ids(ssh, empty_memory=5000) -> list:
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
    # gpu_ids = []
    empty_gpus = {}
    for gpu_info in gpu_info_list:
        # if gpu_info['memory.free'] / gpu_info['memory.total'] > 0.2:
        if gpu_info['memory.free'] > empty_memory:      # MB
            # gpu_ids.append(gpu_info['index'])
            empty_gpus[gpu_info['index']] = gpu_info['memory.free']
    empty_gpus = dict(sorted(empty_gpus.items(), key=operator.itemgetter(1), reverse=True))
    return list(empty_gpus.keys())


def query_memory(ssh) -> float:
    if is_docker():
        cmd = 'cat /sys/fs/cgroup/memory/memory.limit_in_bytes'
        memory = float(ssh.exec_cmd(cmd))       # byte
    else:
        memory = psutil.virtual_memory()[0]
    return float(memory) / 1024**3


def is_port_in_use(port, host='127.0.0.1'):
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, int(port)))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    if os.path.exists(file_path):
        os.remove(file_path)
    logger = logging.getLogger('SSH')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger



class SSH(object):
    def __init__(self, ip, port=22, username=None, password=None, pkey_dir=None):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.pkey_dir = pkey_dir

    def _password_connect(self):
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=self.ip, port=self.port, username=self.username, password=self.password)
        self.t.connect(username=self.username, password=self.password)

    def _key_connect(self):
        self.pkey = paramiko.RSAKey.from_private_key_file(self.pkey_dir)
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=self.ip, port=self.port, username=self.username, pkey=self.pkey)
        self.t.connect(username=self.username, pkey=self.pkey)

    def _connect(self):
        self.ssh = paramiko.SSHClient()
        self.t = paramiko.Transport(sock=(self.ip, self.port))
        self._key_connect()
        # self._password_connect()

    def _close(self):
        self.t.close()
        self.ssh.close()

    def _get_all_files_in_remote_dir(self, sftp, remote_dir):
        all_files = list()
        if remote_dir[-1] == '/':
            remote_dir = remote_dir[0:-1]
        files = sftp.listdir_attr(remote_dir)
        for file in files:
            filename = remote_dir + '/' + file.filename
            if stat.S_ISDIR(file.st_mode):
                all_files.extend(self._get_all_files_in_remote_dir(sftp, filename))
            else:
                all_files.append(filename)
        return all_files

    def _get_all_files_in_local_dir(self, local_dir):
        all_files = list()
        for root, dirs, files in os.walk(local_dir, topdown=True):
            for file in files:
                filename = os.path.join(root, file)
                all_files.append(filename)
        return all_files

    def exec_cmd(self, cmd):
        self._connect()
        _, stdout, stderr = self.ssh.exec_command(cmd)

        res, err = stdout.read(), stderr.read()
        result = res if res else err
        self._close()
        return result.decode().strip()

    def sftp_get(self, remotefile, localfile):
        self._connect()
        sftp = paramiko.SFTPClient.from_transport(self.t)
        sftp.get(remotefile, localfile)
        self._close()

    def sftp_put(self, localfile, remotefile):
        self._connect()
        sftp = paramiko.SFTPClient.from_transport(self.t)
        sftp.put(localfile, remotefile)
        self._close()

    def sftp_get_dir(self, remote_dir, local_dir):
        self._connect()
        sftp = paramiko.SFTPClient.from_transport(self.t)
        if remote_dir[-1] == "/":
            remote_dir = remote_dir[:-1]
        if local_dir[-1] == "/":
            local_dir = local_dir[:-1]
        all_files = self._get_all_files_in_remote_dir(sftp, remote_dir)
        for file in all_files:
            local_filename = file.replace(remote_dir, local_dir)
            local_filepath = os.path.dirname(local_filename)
            if not os.path.exists(local_filepath):
                os.makedirs(local_filepath)
            sftp.get(file, local_filename)
        self._close()

    def sftp_put_dir(self, local_dir, remote_dir):
        self._connect()
        sftp = paramiko.SFTPClient.from_transport(self.t)
        if remote_dir[-1] == "/":
            remote_dir = remote_dir[:-1]
        if local_dir[-1] == "/":
            local_dir = local_dir[:-1]
        all_files = self._get_all_files_in_local_dir(local_dir)
        for file in all_files:
            remote_filename = file.replace(local_dir, remote_dir)
            remote_path = os.path.dirname(remote_filename)
            self.exec_cmd(f'mkdir -p {remote_path}')
            sftp.put(file, remote_filename)
        self._close()


if __name__ == "__main__":

    ssh = SSH(ip='10.26.236.23', username='root', pkey_dir='/root/.ssh/id_rsa')  # 创建一个ssh类对象
    cmd = 'ls -lh'
    ssh.exec_cmd(cmd)  # 执行命令
    # ssh.exec_cmd(cmd)  # 执行命令

    # local_file, remotefile = '/root/ddd/test.py', '/root/test.py'
    # ssh.sftp_put(local_file, remotefile)  # 下载文件
    
    localdir, remotedir = '/root/ddd', '/root/ddd/'
    # ssh.sftp_put_dir(localdir, remotedir)
    ssh.sftp_get_dir(remotedir, localdir)
