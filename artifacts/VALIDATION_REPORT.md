# Validation report

## Data audit

- 150 July MP4 files were inventoried: 43 for Pen 4 and 107 for Pen 5.
- The reproducible validation subset contains three date/size-stratified videos per pen.
- Pen 6 has roster entries but no delivered video. July 20 and 21 folders are empty.
- The roster contains 48 rows across three pens. The marking folder contains 61 references.

## Existing detector comparison

Eight review images were generated from the first and last frames of one Pen 4 and one Pen 5
video, using both existing checkpoints. `train-3` finds more animals in crowded groups but also
shows duplicate sow boxes and a clear floor false positive near the feeder. `train-2` is less
aggressive and was therefore used for the identified smoke run. This is a preliminary observation,
not a precision/recall result; complete the three manual review columns in
`detection_review/detection_review.csv` before choosing a production checkpoint.

## Runtime smoke validation

- `train-3`, Pen 4, 100-frame first pass: 1,003 detections, 13 tracks, and two events. Inspection
  showed that both events came from the sow's large box intersecting the ROI.
- The pipeline was corrected to keep clean crops and exclude implausibly large boxes from piglet
  events/identity.
- Corrected `train-3`, Pen 4, 50 frames: 521 detections, 10 piglet tracks, zero feeding events.
- End-to-end `train-2` plus identity model, Pen 4, 50 frames: 442 detections, 9 piglet tracks, zero
  feeding events. The run produced a valid annotated video, clean crops, event CSV, crop manifest,
  and summary.

The draft ROI JSON was estimated from the first frame only. It must be confirmed with `mark-roi`
before event metrics are meaningful.

## Identity result

All 61 clear close-up references were assigned painted-number labels. The seed model trained on 59
images covering 12 trainable IDs; IDs 14 and 17 have only one image each and were excluded.
Held-out reference accuracy was 11.1%, and no held-out prediction passed the 0.65 rejection
threshold. Every July smoke-run crop was therefore correctly left as `unknown` rather than forcing
an unsupported ID.

This result confirms that close-up reference photos alone do not bridge the domain gap to the
overhead videos. The 44 exported Pen 4 track crops remain in `identity_labels.csv` for manual
labeling. More marked, readable crops from both pens and multiple dates are required before
identity accuracy can be claimed.

## Expansion criteria

Do not batch all 150 videos yet. First:

1. confirm one ROI for each changed camera view;
2. manually review detector counts and choose/fine-tune a checkpoint;
3. label visible track crops from both pens, leaving unreadable marks unknown;
4. annotate true feeding intervals in copies of `ground_truth_events_template.csv`;
5. require agreed event precision/recall, identity accuracy, and identity coverage on held-out
   videos before expansion.
