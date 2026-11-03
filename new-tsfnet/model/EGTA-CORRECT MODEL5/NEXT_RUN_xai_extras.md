# NEXT RUN -- last XAI extras (in this order; paths: replace "..." with your own)

Rule: decide the seed BEFORE looking at results. Seed 42 is the main one; 999 was used for trunk_input_deletion only. Report both when both are run.

1. Channel order (CPU, minutes; no model). Decision rule is in the script header.
   python check_channel_order.py --data-root "E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback" --subjects 1-26 --out-dir "E:\...\channel_order"
   -> VERDICT CONFIRMED  : channels.txt can be passed as --channel-names to xai_member_shap.py (frontal-theta / parietal-alpha steps then run).
   -> NOT CONFIRMED      : channel/region claims stay forbidden (state it as a limitation).
   Needs mne (pip install mne) and the files {sid}_eeg_clean_raw.fif next to the .npy files.

2. GradientShap completeness with more samples (GPU, one subject first):
   python xai_classic_plots.py collect --results-dir "...\results_cf64a71127_seed42" --data-root "..." --subjects 1 --shap-samples 50 --out-dir "...\classic50"
   -> prints median |residual| and corr(sum phi, f(x)-base); compare with the 6-sample run (same line printed by the default call).
   Optional, all subjects: python xai_validated_maps.py ... --shap-samples 50 --out-dir "...\validated_shap50"  (default stays 6 = bit-exact with earlier runs)

3. Trunk input deletion on seed 42 (GPU light):
   python trunk_input_deletion.py --results-dir "...\results_cf64a71127_seed42" --data-root "..." --subjects 1-26 --stride 2 --out-dir "...\results_cf64a71127_seed42\trunk_deletion"

4. Only if time: deep EEG branch in time (GPU, same cost as xai_validated_maps):
   python xai_eeg_branch.py --results-dir "...\results_cf64a71127_seed42" --data-root "..." --subjects 1-26 --out-dir "...\eeg_branch_xai"
   (default --output eeg explains the EEG head; --output pred the fused output)
