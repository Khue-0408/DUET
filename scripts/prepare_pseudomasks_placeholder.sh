# path: scripts/prepare_pseudomasks_placeholder.sh
#!/usr/bin/env bash
# Placeholder script: generate pseudo-masks externally, then place them under:
# data/<DATASET>/pseudo_masks/<same_stem_as_image>.png
#
# The paper mentions "ensemble of existing segmentation models" for pseudomasks
# as a comparison approach (Sec. 3.1). This repo does NOT download any models/datasets.
#
# TODO: integrate your own pseudo-mask generator pipeline here.
echo "Put your pseudo-masks into data/<DATASET>/pseudo_masks/ and set meta.csv pseudo_mask column."
