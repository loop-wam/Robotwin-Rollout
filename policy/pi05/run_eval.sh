# bash eval.sh beat_block_hammer demo_clean pi05_robotwin2 demo_clean 0 4
# bash eval.sh beat_block_hammer demo_randomized pi05_robotwin2 demo_clean 0 4

bash eval_save.sh beat_block_hammer demo_clean pi05_robotwin2 demo_clean 0 5 \
  --save_rollout True \
  --rollout_save_root /data/dengyy/lyp/RoboTwin/data/robotwin2_0-ours/raw_data/5debug \
  --rollout_save_only_success False \
  --rollout_resume False \
  --eval_num_episodes 100
# bash eval.sh beat_block_hammer demo_clean pi05_robotwin2 demo_clean 0 5