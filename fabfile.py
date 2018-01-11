from fabric.api import *
from fabric.contrib.project import rsync_project
import os

env.hosts = ['titanx', 'gtx']
env.user = 'hiep'
env.use_ssh_config = True
ROOT_DIR = os.path.dirname(__file__)


def update():
    local_dir = ROOT_DIR
    rsync_project(local_dir=local_dir, remote_dir='/home/hiep/', exclude=['*.pyc', '.*'], delete=True)
