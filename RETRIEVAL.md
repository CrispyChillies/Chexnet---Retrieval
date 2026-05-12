# Retrieval Finetuning Plan

This repo was built for multilabel classification. Retrieval needs a separate training path because the objective, batch construction, and validation metrics are different.

## Recommended task definition

Use image-to-image retrieval on NIH ChestX-ray14.

- Query: one chest X-ray.
- Gallery: all validation or test chest X-rays.
- Positive match: configurable. The trainer supports:
  `label_overlap` - share at least one pathology.
  `exact_label` - exactly the same multilabel vector.
  `same_patient` - same patient follow-up retrieval.
- Hard negative: visually similar image with no pathology overlap.

That definition is simple enough to implement with the existing split files and is a good first stage for backbone finetuning.

## What the setup needs

- A backbone without the sigmoid classifier head. The retrieval path now supports `resnet50` directly and has optional registry entries for `convnextv2` and `dinov2` families when the environment provides them.
- A dataset that returns image tensor, multilabel vector, patient ID, and sample index.
- A batch sampler that deliberately puts multiple examples of the same pathology into one batch; otherwise contrastive loss degenerates because many anchors have no positives.
- A retrieval loss on normalized embeddings or projections.
- Optional auxiliary multilabel classification loss to keep pathology semantics stable during finetuning.
- Retrieval validation metrics such as Recall@1 and Recall@5 instead of AUROC.

## Loss choice

The scaffold uses:

- `MultiLabelSupConLoss`: positives are any pair with label overlap.
- `AsymmetricLoss`: auxiliary head over the NIH labels.
- `HardNegativeMemoryBank`: mines difficult non-positive examples from the full training gallery representation built each epoch.

The training loss is:

`total_loss = retrieval_weight * supcon_loss + classification_weight * asl_loss + hard_negative_weight * hard_negative_loss`

Why this is a good default:

- Pure triplet loss is fragile unless mining is good.
- InfoNCE with one positive per anchor does not fit NIH well because each image may have multiple valid positives.
- Supervised contrastive loss handles many positives per anchor and works naturally with multilabel overlap.

## Dataloader design

The retrieval dataloader now uses `BalancedPathologyBatchSampler`.

- Pick several pathology groups per batch.
- Sample multiple images for each selected pathology.
- Fill the rest of the batch with random images if needed.
- Grouping can be switched to `pathology`, `exact_label`, or `patient`.

This guarantees anchors usually see positives in-batch.

## Important dataset caveats

- NIH images with all-zero labels are "normal" under the split files in this repo. Treating every normal image as mutually positive can be noisy, so the current loss does not do that by default.
- The repo now includes `SplitBuilder.py` to rebuild patient-disjoint train, val, and test files from the existing split lists.
- NIH labels are weak labels. Retrieval quality based on label overlap will not fully match radiologist similarity.

## Suggested training recipe

Stage 1:

- Freeze early DenseNet blocks for 1 to 3 epochs.
- Train embedding head, projection head, and auxiliary classifier.
- `batch_size=32`, `samples_per_class=4`, `embedding_dim=128`, `temperature=0.07`.

Stage 2:

- Unfreeze the full backbone.
- Continue with lower LR on the backbone and higher LR on the heads.

Good first hyperparameters:

- Backbone LR: `1e-5`
- Head LR: `1e-4`
- Weight decay: `1e-5`
- Retrieval weight: `1.0`
- Classification weight: `0.25`
- Hard negative weight: `0.25`

## Files added

- `RetrievalDataset.py`
- `RetrievalModels.py`
- `RetrievalTrainer.py`
- `SplitBuilder.py`

## Next improvements worth doing

Implemented in the current scaffold:

- Patient-disjoint split builder.
- Hard-negative mining against a full train-gallery memory bank.
- `mAP@10` and `nDCG@10` evaluation.
- Configurable positive definitions.

Environment note:

- `resnet50` is the safest default in the current local environment.
- `convnextv2_*` and `dinov2_*` are registered but require a newer `torchvision` build or `timm`.
