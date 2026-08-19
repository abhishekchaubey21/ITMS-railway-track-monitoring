"""Dataset Registry data models and metadata service for ITMS.

Provides structured access to candidate datasets, licenses, capability mappings,
and source metadata without hardcoding paths or downloading external binaries.

NOTE: This module manages metadata and registry schema only. It does not download datasets.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PrimaryVideoSource:
    """Metadata for the primary input video source."""
    name: str
    local_path: str
    modality: str
    viewpoint: str
    duration_seconds: float
    total_frames: int
    extracted_frames_path: str
    status: str
    license: str


@dataclass
class CandidateDataset:
    """Metadata and capability mapping for an external candidate dataset."""
    id: str
    name: str
    source_url: str
    paper_title: str
    license: str
    academic_hackathon_usable: bool
    modality: str
    viewpoint: str
    defect_or_object_classes: List[str]
    approximate_size: str
    annotation_format: str
    relevance_score: float
    itms_capability_mapping: List[str]
    download_status: str


class DatasetRegistry:
    """Loads and queries candidate dataset metadata for the ITMS AI pipeline."""

    def __init__(
        self,
        schema_version: str,
        last_updated: str,
        project: str,
        primary_video_source: PrimaryVideoSource,
        candidate_datasets: List[CandidateDataset],
    ) -> None:
        self.schema_version = schema_version
        self.last_updated = last_updated
        self.project = project
        self.primary_video = primary_video_source
        self.candidate_datasets = candidate_datasets
        self._dataset_map: Dict[str, CandidateDataset] = {d.id: d for d in candidate_datasets}

    @classmethod
    def from_file(cls, registry_path: str | Path = "data/dataset_registry.json") -> "DatasetRegistry":
        """Loads dataset registry from a JSON file."""
        path = Path(registry_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset registry file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pv_data = data.get("primary_video_source", {})
        pv = PrimaryVideoSource(
            name=pv_data.get("name", "Unknown"),
            local_path=pv_data.get("local_path", ""),
            modality=pv_data.get("modality", ""),
            viewpoint=pv_data.get("viewpoint", ""),
            duration_seconds=float(pv_data.get("duration_seconds", 0.0)),
            total_frames=int(pv_data.get("total_frames", 0)),
            extracted_frames_path=pv_data.get("extracted_frames_path", ""),
            status=pv_data.get("status", ""),
            license=pv_data.get("license", ""),
        )

        candidates: List[CandidateDataset] = []
        for c in data.get("candidate_datasets", []):
            candidates.append(
                CandidateDataset(
                    id=c["id"],
                    name=c["name"],
                    source_url=c["source_url"],
                    paper_title=c.get("paper_title", ""),
                    license=c.get("license", "Unknown"),
                    academic_hackathon_usable=bool(c.get("academic_hackathon_usable", False)),
                    modality=c.get("modality", ""),
                    viewpoint=c.get("viewpoint", ""),
                    defect_or_object_classes=c.get("defect_or_object_classes", []),
                    approximate_size=c.get("approximate_size", ""),
                    annotation_format=c.get("annotation_format", ""),
                    relevance_score=float(c.get("relevance_score", 0.0)),
                    itms_capability_mapping=c.get("itms_capability_mapping", []),
                    download_status=c.get("download_status", "Unregistered"),
                )
            )

        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            last_updated=data.get("last_updated", ""),
            project=data.get("project", ""),
            primary_video_source=pv,
            candidate_datasets=candidates,
        )

    def get_dataset_by_id(self, dataset_id: str) -> Optional[CandidateDataset]:
        """Retrieves a candidate dataset record by its unique ID."""
        return self._dataset_map.get(dataset_id)

    def get_datasets_for_capability(self, capability_keyword: str) -> List[CandidateDataset]:
        """Returns datasets mapped to a specific capability keyword."""
        kw = capability_keyword.lower()
        matched = []
        for d in self.candidate_datasets:
            for cap in d.itms_capability_mapping:
                if kw in cap.lower():
                    matched.append(d)
                    break
        return matched

    def list_usable_datasets(self) -> List[CandidateDataset]:
        """Returns all datasets approved for academic and prototype use."""
        return [d for d in self.candidate_datasets if d.academic_hackathon_usable]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes registry to dictionary representation."""
        return {
            "schema_version": self.schema_version,
            "last_updated": self.last_updated,
            "project": self.project,
            "primary_video_source": asdict(self.primary_video),
            "candidate_datasets": [asdict(d) for d in self.candidate_datasets],
        }
