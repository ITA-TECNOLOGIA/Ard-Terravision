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

from L2.L2_Algorithm import L2_output

# Configure module-level logger
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """
    Configuration-driven multi-stage pipeline executor.

    Time configuration (time_indices, debug_time_index) is defined at the L1 level
    and injected into all downstream L2/L3/L4 algorithms before process_data is called.

    Attributes:
        l1_inputs: List of L1 input stage instances
        l2_algorithms: List of L2 algorithm instances
        l3_algorithms: List of L3 algorithm instances
        l4_algorithm: Single L4 algorithm instance
    """
    l1_inputs: List[Any] = field(default_factory=list)
    l2_algorithms: List[Any] = field(default_factory=list)
    l3_algorithms: List[Any] = field(default_factory=list)
    l4_algorithm: Optional[Any] = None

    # ─── Registry: type name → import path; extend via register() ──────────
    CLASS_REGISTRY: ClassVar[Dict[str, str]] = {
        "Airborne":  "L1.Airborne.Airborne",
        "NumericalData": "L1.NumericalData.NumericalData",
        "AtmosphericCorrection": "L2.AtmosphericCorrection.AtmosphericCorrection",
        "Satellite": "L1.Satellite.Satellite",
        "NumericalData": "L1.NumericalData.NumericalData",
        "AMWI":      "L1.Satellite.SpectralIndices.AMWI",
        "BSI":       "L1.Satellite.SpectralIndices.BSI",
        "EVI":       "L1.Satellite.SpectralIndices.EVI",
        "NDCI":      "L1.Satellite.SpectralIndices.NDCI",
        "NDDI":      "L1.Satellite.SpectralIndices.NDDI",
        "NDTI":      "L1.Satellite.SpectralIndices.NDTI",
        "NDVI":      "L1.Satellite.SpectralIndices.NDVI",
        "NDWI":      "L1.Satellite.SpectralIndices.NDWI",
        "TasseledCapGreenness":      "L1.Satellite.Transformations.TasseledCapGreenness",
        "TasseledCapWetness":      "L1.Satellite.Transformations.TasseledCapWetness",
        "LST":      "L1.Satellite.LandSurfaceTemperature.LST",
        "CloudMasking":          "L2.CloudMasking.CloudMasking",
        "SpectralIndexFusion":   "L2.SpectralIndexFusion.SpectralIndexFusion",
        "ObjectDetectionDetrex":       "L3.ObjectDetection.ObjectDetectionDetrex",
        "ObjectDetectionGroundedSAM2": "L3.ObjectDetection.ObjectDetectionGroundedSAM2",
        "ChangeDetectionMineNetCD":             "L3.ChangeDetection.ChangeDetectionMineNetCD",
        "LulcClassification":    "L3.LulcClassification.LulcClassification",
        "EnvIndicator":          "L3.SpectralIndices.EnvIndicator",
        "LLaVACustom": "L4.LLaVACustom.LLaVACustom",
        "DummyLLaVACustom": "L4.LLaVACustom.DummyLLaVACustom",
        "QwenCustom": "L4.QwenCustom.QwenCustom",
        "TimeSeriesTreatment": "L2.TimeSeriesTreatment.TimeSeriesTreatment",
        "TimeSeriesAnomalyDetection": "L3.TimeSeriesAnalysis.TimeSeriesAnomalyDetection",
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

    # ─── Time config extraction ───────────────────────────────────────────
    @staticmethod
    def _extract_time_config(l1_inputs: List[Any]) -> Dict[str, Any]:
        """
        Extract unified time configuration from L1 inputs.
        Validates that all L1 inputs have consistent time_indices.
        If they differ, uses the intersection and logs a warning.
        """
        time_indices_list = []
        debug_indices = []

        for inp in l1_inputs:
            ti = getattr(inp, 'time_indices', None)
            if ti is not None and len(ti) > 0:
                time_indices_list.append(list(ti))
            else:
                time_indices_list.append([])
            di = getattr(inp, 'debug_time_index', None)
            if di is not None:
                debug_indices.append(di)

        # Determine final time_indices
        non_empty = [ti for ti in time_indices_list if ti]
        if non_empty:
            unique = {tuple(ti) for ti in non_empty}
            if len(unique) > 1:
                logger.warning(
                    f"L1 inputs have mismatched time_indices: {non_empty}. "
                    f"Using intersection."
                )
                common = set(non_empty[0])
                for ti in non_empty[1:]:
                    common &= set(ti)
                if not common:
                    raise ValueError("L1 inputs have no overlapping time_indices")
                time_indices = sorted(common)
            else:
                time_indices = non_empty[0]
        else:
            time_indices = []

        # Determine final debug_time_index
        debug_time_index = debug_indices[0] if debug_indices else 0
        if debug_indices and len(set(debug_indices)) > 1:
            logger.warning(
                f"L1 inputs have mismatched debug_time_index: {debug_indices}. "
                f"Using first: {debug_time_index}"
            )

        return {
            "time_indices": time_indices,
            "debug_time_index": debug_time_index,
        }

    @staticmethod
    def _inject_time_config(algorithm: Any, time_config: Dict[str, Any]) -> None:
        """
        Inject time indices and debug_time_index into an algorithm instance.
        """
        algorithm.time_indices = list(time_config["time_indices"])
        algorithm.debug_time_index = time_config["debug_time_index"]

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PipelineConfig":
        """
        Build PipelineConfig from a dict. Validates required sections.
        """
        def build_section(key: str):
            block = cfg.get(key)
            if key in ("l2_algorithms", "l3_algorithms"):
                return [cls._instantiate(cls._load_class(b["type"]), b.get("params", {}))
                        for b in block or []]
            if block is None:
                return None
            params = block.get("params", {})
            return cls._instantiate(cls._load_class(block["type"]), params)

        def build_l1_inputs():
            l1_list = []
            if "l1_inputs" in cfg:
                for block in cfg["l1_inputs"]:
                    params = block.get("params", {})
                    alg = cls._instantiate(cls._load_class(block["type"]), params)
                    l1_list.append(alg)
            elif "l1_input" in cfg:
                l1_list.append(build_section("l1_input"))
            else:
                raise KeyError("Either 'l1_inputs' (list) or 'l1_input' (single) must be defined")
            return l1_list

        l1_inputs = build_l1_inputs()
        l2 = build_section("l2_algorithms")
        l3 = build_section("l3_algorithms")
        l4 = build_section("l4_algorithm")

        return cls(l1_inputs=l1_inputs, l2_algorithms=l2, l3_algorithms=l3, l4_algorithm=l4)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """
        Loads config from JSON file and delegates to from_dict().
        """
        with open(path, "r") as f:
            cfg = json.load(f)
        return cls.from_dict(cfg)

    def __repr__(self) -> str:
        l1_names = [inp.__class__.__name__ for inp in self.l1_inputs]
        return (
            f"<PipelineConfig l1_inputs={l1_names}, "
            f"l2={[a.__class__.__name__ for a in self.l2_algorithms]}, "
            f"l3={[a.__class__.__name__ for a in self.l3_algorithms]}, "
            f"l4={self.l4_algorithm.__class__.__name__ if self.l4_algorithm else None}>"
        )

    # ─── Stage runners ─────────────────────────────────────────────────
    def run_l1(self) -> List[Any]:
        return self.l1_inputs

    def run_l2(self, l1_inputs: List[Any]) -> Optional[L2_output]:
        if not self.l2_algorithms:
            return None

        time_config = self._extract_time_config(l1_inputs)
        for alg in self.l2_algorithms:
            self._inject_time_config(alg, time_config)

        results = [alg.process_data(l1_inputs) for alg in self.l2_algorithms]
        return results[0] if results else None

    def run_l3(self, l1_inputs: List[Any], l2_output: Optional[L2_output] = None) -> List[Any]:
        l2_datacube = l2_output.datacube if l2_output else None

        if len(l1_inputs) > 1 and not l2_datacube:
            raise ValueError(
                f"Multiple L1 inputs ({len(l1_inputs)}) provided to L3 and no L2. "
                "Multiple inputs should be fused in L2 (e.g., SpectralIndexFusion) "
                "before passing to L3."
            )

        time_config = self._extract_time_config(l1_inputs)

        if l2_datacube is not None and "t" in l2_datacube.dims:
            l2_n_t = l2_datacube.sizes["t"]
            ti = time_config["time_indices"]
            if ti and max(ti) >= l2_n_t:
                logger.info(
                    f"L2 datacube has {l2_n_t} timesteps but L1 time_indices "
                    f"reference up to index {max(ti)} (likely due to temporal "
                    f"aggregation in L2). Resetting to use all L2 timesteps."
                )
                time_config["time_indices"] = []
            if time_config["debug_time_index"] >= l2_n_t:
                time_config["debug_time_index"] = 0

        self._validated_time_config = time_config

        for alg in self.l3_algorithms:
            self._inject_time_config(alg, time_config)

        l1_input = l1_inputs[0] if l1_inputs and not l2_datacube else None
        
        all_results = []
        for alg in self.l3_algorithms:
            results = alg.process_data(l1_input, l2_datacube)
            all_results.extend(results)
            if hasattr(alg, 'save_results'):
                alg.save_results(results)
        return all_results

    def run_l4(self, l1_inputs: List[Any], l3_results: List[Any]) -> Any:
        if len(l1_inputs) > 1:
            raise ValueError(
                f"Multiple L1 inputs ({len(l1_inputs)}) provided to L4. "
                "Multiple inputs should be fused in L2 (e.g., SpectralIndexFusion) "
                "before passing to L4."
            )

        if hasattr(self, '_validated_time_config'):
            time_config = self._validated_time_config
        else:
            time_config = self._extract_time_config(l1_inputs)
        if self.l4_algorithm:
            self._inject_time_config(self.l4_algorithm, time_config)

        l1_input = l1_inputs[0] if l1_inputs else None

        target_time_index = getattr(self.l4_algorithm, 'debug_time_index', None)
        result = self.l4_algorithm.process_data(l1_input, l3_results, target_time_index)
        if hasattr(self.l4_algorithm, 'save_results'):
            self.l4_algorithm.save_results(result)
        return result

    def run(self) -> Any:
        """
        Executes the full pipeline: L1 -> L2 -> L3 -> L4.
        """
        l1_inputs = self.run_l1()
        l2_output = self.run_l2(l1_inputs)
        l3_results = self.run_l3(l1_inputs, l2_output)

        if self.l4_algorithm is None:
            return l3_results

        return self.run_l4(l1_inputs, l3_results)
