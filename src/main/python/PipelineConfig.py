# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import json
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, ClassVar, Optional
import xarray as xr

from L2.L2_Algorithm import L2_output

# Configure module-level logger
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """
    Configuration-driven multi-stage pipeline executor.

    Attributes:
        l1_input: Input stage instance
        l2_algorithms: List of L2 algorithm instances
        l3_algorithms: List of L3 algorithm instances
        l4_algorithm: Single L4 algorithm instance
    """
    l1_input: Any
    l2_algorithms: List[Any] = field(default_factory=list)
    l3_algorithms: List[Any] = field(default_factory=list)
    l4_algorithm: List[Any] = field(default_factory=list)

    # ─── Registry: type name → import path; extend via register() ──────────
    CLASS_REGISTRY: ClassVar[Dict[str, str]] = {
        "Airborne":  "L1.Airborne.Airborne",
        "Satellite": "L1.Satellite.Satellite",
        "AMWI":      "L1.Satellite.SpectralIndices.AMWI",
        "BSI":       "L1.Satellite.SpectralIndices.BSI",
        "EVI":       "L1.Satellite.SpectralIndices.EVI",
        "NDCI":      "L1.Satellite.SpectralIndices.NDCI",
        "NDDI":      "L1.Satellite.SpectralIndices.NDDI",
        "NDTI":      "L1.Satellite.SpectralIndices.NDTI",
        "NDVI":      "L1.Satellite.SpectralIndices.NDVI",
        "NDWI":      "L1.Satellite.SpectralIndices.NDWI",
        "AtmosphericCorrection": "L2.AtmosphericCorrection.AtmosphericCorrection",
        "CloudMasking":          "L2.CloudMasking.CloudMasking",
        "PanSharpening":         "L2.PanSharpening.PanSharpening",
        "Orthorectification":    "L2.Orthorectification.Orthorectification",
        "ObjectDetectionDetrex":       "L3.ObjectDetection.ObjectDetectionDetrex",
        "ObjectDetectionGroundedSAM2": "L3.ObjectDetection.ObjectDetectionGroundedSAM2",
        "ChangeDetection":       "L3.ChangeDetection.ChangeDetection",
        "ChangeDetectionMineNetCD":             "L3.ChangeDetection.ChangeDetectionMineNetCD",
        "LulcClassification":    "L3.LulcClassification.LulcClassification",
        "SemanticCaptioning":    "L3.SemanticCaptioning.SemanticCaptioning",
        "AMWI":                  "L3.SpectralIndices.AMWI",
        "BSI":                   "L3.SpectralIndices.BSI",
        "EVI":                   "L3.SpectralIndices.EVI",
        "NDCI":                  "L3.SpectralIndices.NDCI",
        "NDDI":                  "L3.SpectralIndices.NDDI",
        "NDTI":                  "L3.SpectralIndices.NDTI",
        "NDVI":                  "L3.SpectralIndices.NDVI",
        "NDWI":                  "L3.SpectralIndices.NDWI",
        "EnvIndicator":          "L3.SpectralIndices.EnvIndicator",
        "InformationFusionGenerativeVLM": "L4.InformationFusionGenerativeVLM.InformationFusionGenerativeVLM",
        "LLaVACustom": "L4.LLaVACustom.LLaVACustom",
        "DummyLLaVACustom": "L4.LLaVACustom.DummyLLaVACustom",
        "QwenCustom": "L4.QwenCustom.QwenCustom",
    }
    _class_cache: ClassVar[Dict[str, Type[Any]]] = {}

    @classmethod
    def register(cls, type_name: str, module_path: str) -> None:
        """
        Register or override a component type mapping.
        """
        cls.CLASS_REGISTRY[type_name] = module_path
        cls._class_cache.pop(type_name, None)
        logger.debug(f"Registered type '{type_name}' -> '{module_path}'")

    @classmethod
    def _load_class(cls, type_name: str) -> Type[Any]:
        """
        Dynamically imports and caches a class by its registered name.
        Raises KeyError if type_name is unknown.
        """
        if type_name in cls._class_cache:
            return cls._class_cache[type_name]

        try:
            module_path = cls.CLASS_REGISTRY[type_name]
        except KeyError:
            raise KeyError(f"Unknown type '{type_name}'. Please register it.")

        try:
            module = importlib.import_module(module_path)
            AlgClass = getattr(module, type_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Cannot load '{type_name}' from '{module_path}': {e}")
            raise

        cls._class_cache[type_name] = AlgClass
        logger.debug(f"Loaded class '{type_name}' from '{module_path}'")
        return AlgClass

    @staticmethod
    def _instantiate(Alg: Type[Any], params: Any) -> Any:
        """
        Instantiates an algorithm class with flexible params:
          - dict -> **kwargs
          - list of dicts -> single positional argument
          - list/tuple -> *args
        """
        if isinstance(params, dict):
            return Alg(**params)
        if isinstance(params, (list, tuple)):
            if all(isinstance(el, dict) for el in params):
                return Alg(params)
            return Alg(*params)
        raise TypeError(f"Invalid params type {type(params)} for {Alg}")

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PipelineConfig":
        """
        Build PipelineConfig from a dict. Validates required sections.
        """
        # Required
        if "l1_input" not in cfg or "l4_algorithm" not in cfg:
            raise KeyError("Both 'l1_input' and 'l4_algorithm' must be defined")

        def build_section(key: str):
            block = cfg.get(key)
            if key in ("l2_algorithms", "l3_algorithms"):
                return [cls._instantiate(cls._load_class(b["type"]), b.get("params", {}))
                        for b in block or []]
            # single-instance section - extract target_time_index if present
            if block is None:
                return None
            params = block.get("params", {})
            if isinstance(params, dict):
                target_time_index = params.pop("target_time_index", None)
            else:
                target_time_index = None
            alg = cls._instantiate(cls._load_class(block["type"]), params)
            if target_time_index is not None:
                alg.target_time_index = target_time_index
            return alg

        l1 = build_section("l1_input")
        l2 = build_section("l2_algorithms")
        l3 = build_section("l3_algorithms")
        l4 = build_section("l4_algorithm")

        return cls(l1_input=l1, l2_algorithms=l2, l3_algorithms=l3, l4_algorithm=l4)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """
        Loads config from JSON file and delegates to from_dict().
        """
        with open(path, "r") as f:
            cfg = json.load(f)
        return cls.from_dict(cfg)

    def __repr__(self) -> str:
        return (
            f"<PipelineConfig l1={self.l1_input.__class__.__name__}, "
            f"l2={[a.__class__.__name__ for a in self.l2_algorithms]}, "
            f"l3={[a.__class__.__name__ for a in self.l3_algorithms]}, "
            f"l4={self.l4_algorithm.__class__.__name__}>"
        )

    # ─── Stage runners ─────────────────────────────────────────────────
    def run_l1(self) -> Any:
        return self.l1_input

    def run_l2(self, l1_data: Any) -> Optional[L2_output]:
        if not self.l2_algorithms:
            return None
        results = [alg.process_data(l1_data) for alg in self.l2_algorithms]
        return results[0] if results else None

    def run_l3(self, l1_data: Any, l2_output: Optional[L2_output] = None) -> List[Any]:
        l2_datacube = l2_output.datacube if l2_output else None
        all_results = []
        for alg in self.l3_algorithms:
            results = alg.process_data(l1_data, l2_datacube)
            all_results.extend(results)
            if hasattr(alg, 'save_results'):
                alg.save_results(results)
        return all_results

    def run_l4(self, l1_data: Any, l3_results: List[Any], target_time_index: Optional[int] = None) -> Any:
        if self.l4_algorithm is None:
            return None
        result = self.l4_algorithm.process_data(l1_data, l3_results, target_time_index)
        if hasattr(self.l4_algorithm, 'save_results'):
            self.l4_algorithm.save_results(result)
        return result

    def run(self) -> Any:
        """
        Executes the full pipeline: L1 -> L2 -> L3 -> L4.
        """
        l1 = self.run_l1()
        l2_output = self.run_l2(l1)
        l3_results = self.run_l3(l1, l2_output)

        if self.l4_algorithm is None:
            return l3_results

        target_time_index = getattr(self.l4_algorithm, 'target_time_index', None)
        return self.run_l4(l1, l3_results, target_time_index)
