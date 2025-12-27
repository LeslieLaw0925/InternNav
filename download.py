import os

from huggingface_hub import snapshot_download


snapshot_download(repo_id="InternRobotics/InternVLA-N1-w-NavDP", 
                  local_dir="checkpoints/InternVLA-N1-w-NavDP", 
                  repo_type="model",
                  token=os.getenv("HUGGINGFACE_TOKEN")
                  )