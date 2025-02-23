$current_dir = Split-Path -Path (Get-Item -Path $MyInvocation.MyCommand.Path).FullName -Parent

Set-Location -Path $current_dir
Set-Location -Path "../../../"

$env:DOCKER_WORKPLACE_NAME = "workplace"
$env:EVAL_MODE = "True"
$env:DEBUG = "True"
$env:BASE_IMAGES = "gaia-bookworm-with-git:latest"
$env:COMPLETION_MODEL = "openrouter/google/gemini-2.0-flash-001"

python evaluation/gaia/run_infer.py --container_name gaia_lite_eval --model $env:COMPLETION_MODEL --test_pull_name main --debug --eval_num_workers 1 --port 12345 --data_split validation --level 2023_all --agent_func get_system_triage_agent --git_clone
# python /Users/tangjiabin/Documents/reasoning/metachain/test_gaia_tool.py