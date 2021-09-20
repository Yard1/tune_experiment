import ray
import subprocess
import os
import socket


def set_up_s3fs(path: str = "~/results"):
    path = os.path.expanduser(path)
    if os.path.exists(path):
        return
    subprocess.Popen(
        f"mkdir '{path}' && nohup s3fs tune-experiment-result '{path}' -o iam_role='auto' -o url='https://s3-us-west-2.amazonaws.com' &",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        shell=True,
        start_new_session=True)
    subprocess.Popen(
        f"touch '{path}/{socket.gethostname()}'",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        shell=True)


def run_on_every_ray_node(func, **kwargs):
    nodes = {k for k in ray.cluster_resources() if "node:" in k}
    remote_func = ray.remote(func)
    refs = []
    for node in nodes:
        refs.append(
            remote_func.options(num_cpus=0, resources={
                node: 1
            }).remote(**kwargs))
    return ray.get(refs)