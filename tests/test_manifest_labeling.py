from pathlib import Path

from piglet_identity.labeling import score_manifest_labels


def test_score_manifest_compares_manual_to_classifier(tmp_path: Path) -> None:
    manifest = tmp_path / "crop_manifest.csv"
    manifest.write_text(
        "image_path,track_id,pig_id,classifier_pig_id,manual_pig_id,manual_status\n"
        "/a.jpg,1,unknown,12,12,labeled\n"
        "/b.jpg,2,12,12,14,labeled\n"
        "/c.jpg,3,unknown,unknown,,skip\n"
    )
    metrics = score_manifest_labels(manifest)
    assert metrics["manual_labeled_rows"] == 2
    assert metrics["classifier_pig_id_accuracy"] == 0.5
    assert metrics["pig_id_accuracy"] == 0.0
