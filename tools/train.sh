set -ex

python train.py \
../configs/rgba/csd/lbl_completed_decomposition_htc_csd.py \
--work_dir ./work_dirs/test \
--resume_from ../checkpoints/lbl_completion_decomposition_csd_finetune_merged/latest.pth