from pathlib import Path

from piglet_identity.labeling import prepare_labeling_package


def test_second_prepare_keeps_unlabeled_track_crops(tmp_path: Path) -> None:
    """Empty pig_id must not be treated as skip on re-prepare (regression)."""
    from openpyxl import Workbook

    workbook = tmp_path / "roster.xlsx"
    wb = Workbook()
    wb.active.title = "7315"
    wb.active.append(["sow", "ear", "piglet", "gender"])
    wb.active.append([1, 1, 7, "F"])
    wb.save(workbook)

    marks = tmp_path / "marks"
    marks.mkdir()
    samples = tmp_path / "samples" / "video_a"
    samples.mkdir(parents=True)
    (samples / "track_42.jpg").write_bytes(b"fake")
    out = tmp_path / "labels"

    first = prepare_labeling_package(workbook, marks, out, crops_root=tmp_path / "samples")
    assert first["new_unlabeled"] == 1
    assert first["unlabeled_total"] == 1

    second = prepare_labeling_package(workbook, marks, out, crops_root=tmp_path / "samples")
    assert second["new_unlabeled"] == 0
    assert second["unlabeled_total"] == 1
    assert second["unlabeled_by_video"] == {"video_a": 1}
