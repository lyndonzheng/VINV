set -x
python test.py \
../configs/rgba/cocoa/lbl_completed_decomposition_htc_cocoa.py \
../checkpoints/lbl_completion_decomposition_cocoa_end_pseudo_48_fixedbatch_5/latest.pth \
--out ../results/test.pkl \
--eval bbox segm \
--out_path ../results/lbl_completion_decomposition_htc_csd_pre_mask/ \
--with_occ \
--show