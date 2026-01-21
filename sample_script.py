import subprocess

env_name = 'hoppercpx'
commands = [
    # 注意：这里去掉了 'nohup' 和 '&'，利用 Python 自身来阻塞等待
    f"python sample_diffusion.py --config-name=time_RoS_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_RoS_{env_name}.out 2>&1",
    f"python sample_flow.py --config-name=time_flow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_flow_{env_name}.out 2>&1",    
    f"python sample_flow.py --config-name=time_safeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_safeflow_{env_name}.out 2>&1",
    f"python sample_polyflow.py --config-name=time_polyflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_polyflow_{env_name}.out 2>&1",
    f"python sample_gaugeflow.py --config-name=time_gaugeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_gaugeflow_{env_name}.out 2>&1",
]

for cmd in commands:
    print(f"正在执行: {cmd}")
    # Python 会在这里等待，直到当前命令运行结束才继续下一行
    subprocess.run(cmd, shell=True) 
    
print(f"所有 {env_name} 任务执行完毕。")



env_name = 'hoppercpx2'
commands = [
    # 注意：这里去掉了 'nohup' 和 '&'，利用 Python 自身来阻塞等待
    f"python sample_diffusion.py --config-name=time_RoS_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_RoS_{env_name}.out 2>&1",
    f"python sample_flow.py --config-name=time_flow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_flow_{env_name}.out 2>&1",    
    f"python sample_flow.py --config-name=time_safeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_safeflow_{env_name}.out 2>&1",
    f"python sample_polyflow.py --config-name=time_polyflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_polyflow_{env_name}.out 2>&1",
    f"python sample_gaugeflow.py --config-name=time_gaugeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_gaugeflow_{env_name}.out 2>&1",
]

for cmd in commands:
    print(f"正在执行: {cmd}")
    # Python 会在这里等待，直到当前命令运行结束才继续下一行
    subprocess.run(cmd, shell=True) 
    
print(f"所有 {env_name} 任务执行完毕。")


env_name = 'walkercpx'
commands = [
    # 注意：这里去掉了 'nohup' 和 '&'，利用 Python 自身来阻塞等待
    f"python sample_diffusion.py --config-name=time_RoS_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_RoS_{env_name}.out 2>&1",
    f"python sample_flow.py --config-name=time_flow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_flow_{env_name}.out 2>&1",    
    f"python sample_flow.py --config-name=time_safeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_safeflow_{env_name}.out 2>&1",
    f"python sample_polyflow.py --config-name=time_polyflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_polyflow_{env_name}.out 2>&1",
    f"python sample_gaugeflow.py --config-name=time_gaugeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_gaugeflow_{env_name}.out 2>&1",
]

for cmd in commands:
    print(f"正在执行: {cmd}")
    # Python 会在这里等待，直到当前命令运行结束才继续下一行
    subprocess.run(cmd, shell=True) 
    
print(f"所有 {env_name} 任务执行完毕。")


env_name = 'walkercpx2'
commands = [
    # 注意：这里去掉了 'nohup' 和 '&'，利用 Python 自身来阻塞等待
    f"python sample_diffusion.py --config-name=time_RoS_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_RoS_{env_name}.out 2>&1",
    f"python sample_flow.py --config-name=time_flow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_flow_{env_name}.out 2>&1",    
    f"python sample_flow.py --config-name=time_safeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_safeflow_{env_name}.out 2>&1",
    f"python sample_polyflow.py --config-name=time_polyflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_polyflow_{env_name}.out 2>&1",
    f"python sample_gaugeflow.py --config-name=time_gaugeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_gaugeflow_{env_name}.out 2>&1",
]

for cmd in commands:
    print(f"正在执行: {cmd}")
    # Python 会在这里等待，直到当前命令运行结束才继续下一行
    subprocess.run(cmd, shell=True) 
    
print(f"所有 {env_name} 任务执行完毕。")


env_name = 'halfcheetah'
commands = [
    # 注意：这里去掉了 'nohup' 和 '&'，利用 Python 自身来阻塞等待
    f"python sample_diffusion.py --config-name=time_RoS_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_RoS_{env_name}.out 2>&1",
    f"python sample_flow.py --config-name=time_flow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_flow_{env_name}.out 2>&1",    
    f"python sample_flow.py --config-name=time_safeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_safeflow_{env_name}.out 2>&1",
    f"python sample_polyflow.py --config-name=time_polyflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_polyflow_{env_name}.out 2>&1",
    f"python sample_gaugeflow.py --config-name=time_gaugeflow_{env_name}.yaml eval.skip_rollout=false device=cuda:0 >tmp_logs/time_gaugeflow_{env_name}.out 2>&1",
]

for cmd in commands:
    print(f"正在执行: {cmd}")
    # Python 会在这里等待，直到当前命令运行结束才继续下一行
    subprocess.run(cmd, shell=True) 
    
print(f"所有 {env_name} 任务执行完毕。")