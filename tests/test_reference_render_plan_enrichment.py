import json
import tempfile
import unittest
from pathlib import Path

from feverslop.application.reference_bible import enrich_render_plan_with_reference_sheets


class ReferenceRenderPlanEnrichmentTests(unittest.TestCase):
    def test_enriches_reference_ids_with_sheet_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor_dir = temp / "actors" / "singer"
            location_dir = temp / "locations" / "stage"
            actor_dir.mkdir(parents=True)
            location_dir.mkdir(parents=True)
            actor_sheet = actor_dir / "sheet.png"
            location_sheet = location_dir / "sheet.png"
            actor_sheet.write_bytes(b"actor")
            location_sheet.write_bytes(b"location")
            (actor_dir / "manifest.json").write_text(
                json.dumps({"id": "singer", "kind": "actor", "sheet_path": str(actor_sheet)}),
                encoding="utf-8",
            )
            (location_dir / "manifest.json").write_text(
                json.dumps({"id": "stage", "kind": "location", "sheet_path": str(location_sheet)}),
                encoding="utf-8",
            )
            render_plan_path = temp / "render_plan.json"
            render_plan_path.write_text(
                json.dumps([
                    {"scene": 1, "references": {"actor_ids": ["singer"], "location_id": "stage"}}
                ]),
                encoding="utf-8",
            )

            output_path = enrich_render_plan_with_reference_sheets(render_plan_path, temp, temp / "render_plan_refs.json")

            enriched = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([str(actor_sheet)], enriched[0]["references"]["actor_sheet_paths"])
            self.assertEqual(str(location_sheet), enriched[0]["references"]["location_sheet_path"])
