# Component 3 research plan

## Distinct research contribution

Link fabric identification to recorded physical properties and measurement-driven design suggestions. Component 2 supplies its shared measurement validation; Component 3 does not duplicate body/skin-tone inference. Component 1 image generation is outside scope.

The proposal's Mahanta (2024), Sharma (2021) and novelty/gap claims need full paper titles/DOIs and literature review before being cited as established facts. Do not invent citations or claim universal novelty.

## Milestones

1. **Foundation — implemented:** separate local app, read-only shared measurements/taxonomy, reviewed properties/provenance, provisional rules, unavailable classifier state and tests.
2. **Dataset and properties:** agree label definitions; obtain permitted close-ups across real Sri Lankan/global samples; physical-sample IDs; source permissions; documented weight/drape/stretch protocols. Create a versioned property database keyed by physical fabric specimen, not just class average. Distinguish measured values from uncertain reference ranges.
3. **Classification development — baseline and controlled partial fine-tuning implemented:** audited TextileClass-7 manifest, EfficientNet-B0 frozen-head baseline (replacing the original ResNet-18 proposal), error review and five-epoch last-stage fine-tuning comparison. Best candidate validation macro-F1 0.6881; test reserved. Provider images do not have reliable specimen IDs; current conservative image grouping is provisional, not specimen-independent. ConvNeXt-Tiny comparison remains. Train only supported heads/labels, masked unknown labels. Plan print/weave/method heads separately, never exclusive mixed cotton-vs-handloom classes.
4. **Experimental inference — integrated:** pinned weights/report/manifest checksums, CPU model caching, explicit opt-in, reviewed category separate from composition, top-three relative scores and no predictions when model unavailable. Softmax is not calibrated accuracy. Calibration and actual OOD/unknown rejection remain unimplemented; evaluate out-of-distribution photos before any validated release.
5. **Expert design engine:** replace starter rules/ease placeholders with attributed reviewed rules and measured fabric-property records. Include shrinkage, stretch/recovery, garment construction and movement needs where evidence exists. Unknown values must not silently become averages.
6. **Evaluation:** final held-out per-class precision/recall/F1 and confusion matrices, macro-F1/balanced accuracy, sample counts and uncertainty. Report Sri Lankan/global sets separately and document overlapping fabric types. Measure latency on stated hardware. Blind independent design-suitability ratings by relevant reviewers; compare properties-aware recommendations against style/measurement-only baseline, with sample/scenario grouping and reviewer agreement.
7. **Release:** frozen evaluated artifact, property/rule provenance, reproducible environment, explicit limitations and complete README/checkpoint.

## Latest implementation checkpoint

Independent-pipeline update: the researcher confirmed no new labelled photos are available. External collection schema/audit, true separate calibration and assessment, masked independent technique/categorical heads, attribute assessment and one-shot final evaluators are implemented. No actual independent calibration/OOD or real attribute-training results exist yet. No new head is activated. Reserved provider inference is guarded and not executed. Follow `INDEPENDENT_RESEARCH_PIPELINE.md`; fixtures test mechanics only.

The four-column frontend, append-only local specimen catalogue with per-property source assertions/checksums, and controlled property-ablation comparison are implemented. The comparison holds measurements/style/fit/silhouette constant and removes properties only from its baseline; it reports rule ranks/points, not suitability accuracy. Unknown properties leave rankings identical. Both modes still use illustrative, unvalidated ease rules. No real specimen measurements or expert approvals have been fabricated. Next evidence-dependent tasks remain genuine sourced specimen records, attributed expert rule review and independent design-suitability ratings. The final classifier test split remains reserved.

Later on 18 October: ConvNeXt frozen-head (59.27%, macro-F1 0.5998) and five-epoch partial fine-tuning (selected epoch 4, 69.38%, macro-F1 0.6993) completed. The latter is only three more correct validation images than EfficientNet; no statistical superiority claim. Both models have post-selection group-split development calibration reports. Temperatures and model switching are not activated. The inference code supports either pinned architecture; CPU candidate repeat/reload checks passed. Unknown-input evaluation tooling/fixtures are complete but real external supported/unknown images remain missing. See `CALIBRATION_AND_UNKNOWN.md`; independent calibration/OOD evaluation and specimen review are still required. Latest automated suite: 74 tests.

## Risks and guardrails

ConvNeXt V2 update, 18 October: pinned FCMAE/ImageNet-22K/1K V2-Tiny frozen-head plus five-epoch final-stage experiment completed at 72.47% validation accuracy / macro-F1 0.7313 (epoch 5), above V1's 69.38%. Broader pretraining and architecture-specific transforms prevent attributing the gain to architecture alone. The non-commercial candidate passed CPU repeatability, frozen-state and reload checks; 77 automated tests pass. No active release change or final-test inference. See `CONVNEXTV2_EXPERIMENT.md`; real external unknown inputs and independent calibration remain outstanding.

- Photo appearance does not certify fibre chemistry, handloom manufacture, GSM or mechanical drape.
- Batik (dye technique) and handloom (production method) can coexist with cotton/silk and with each other. Unknown or multilabel annotations must be supported by the dataset design.
- Weave construction, printed motifs and surface texture are separate dimensions.
- Garment drape_style and material drapability must never share the same semantic field.
- Multiple photos/crops of one physical specimen belong to the same split; source/photographer and near-duplicates can also leak information.
- Diverse capture conditions and real specimens are needed. Synthetic/UI fixture tests prove mechanics only.
- Supplier statements and user confirmations are provenance assertions, not independent scientific verification.
- Avoid body-shape stereotypes and skin-tone-based design restrictions; current rules use measurements and explicit preferences only.
- No fitting/cutting instructions should be presented as validated before appropriate pattern-making review.

## Sources checked for model planning

- [Official PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial): pretrained ResNet-18 with replaced classifier, frozen backbone or fine-tuning.
- [Torchvision ResNet-18 implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).
- [MIT Flickr Material Database project](https://people.csail.mit.edu/lavanya/matlrecog.html): fabric is only one of ten broad material categories, so this is not by itself a cotton/silk/batik/handloom training set. No use/download is claimed.
