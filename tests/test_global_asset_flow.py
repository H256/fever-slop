import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.global_library import GlobalLibraryAdapter
from feverslop.application.global_cast_resolver import GlobalCastResolver
from feverslop.config.project_config import GlobalAssetConfig
from feverslop.domain.global_library import AssetKind, GlobalAsset
from feverslop.domain.reference_workspace import (
    PropInteraction,
    SceneReferenceAssignment,
)


class GlobalAssetFlowTests(unittest.TestCase):
    def test_fake_port_flow_is_portable_without_comfyui(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = GlobalLibraryAdapter(root / "library")
            library.create(GlobalAsset("ava", AssetKind.CHARACTER, "Ava"))
            library.create(GlobalAsset("guitar", AssetKind.PROP, "Guitar"))
            resolved = GlobalCastResolver(library).resolve(
                cast=(GlobalAssetConfig("ava"),), locations=(), styles=(), props=(GlobalAssetConfig("guitar"),),
                project_reference_dir=root / "project" / "references",
            )
            assignment = SceneReferenceAssignment(
                scene_number=1, actor_ids=("ava",), prop_ids=("guitar",),
                prop_interactions=(PropInteraction("ava", "guitar", "holds"),),
            )
            self.assertEqual([], assignment.validate_against(
                known_actor_ids=[item["id"] for item in resolved.actors], known_location_ids=[],
                known_prop_ids=[item["id"] for item in resolved.props], max_scene_actors=4,
            ))
            self.assertTrue((root / "project" / "references" / "global_assets" / "prop" / "guitar" / "default" / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
