from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Any, Type
import numpy as np

MAX_STEPS = {
    "cart_pole_obst": 400,
    "bipedal_walker": 1000,
    "lunar_lander": 600,
    "racecar2": 300,
    "racecar": 300
}

_register = {}


class EvaluationFunction:
    @abstractmethod
    def __call__(self, data: Dict[str, np.ndarray], env_name: str) -> np.ndarray:
        pass


class WeightedEvalFunction(EvaluationFunction):
    safety_prefix = "s"
    target_prefix = "t_"    # be careful: also "timesteps" starts with "t"
    comfort_prefix = "c"

    def __init__(self, name: str, safety_w: float, target_w: float, comfort_w: float):
        self.name = name
        self.weights = {cls: w for cls, w in zip(["safety", "target", "comfort"],
                                                 [safety_w, target_w, comfort_w])}

    def __call__(self, data: Dict[str, np.ndarray], env_name: str):
        # extract metrics names
        safeties = [k for k in data.keys() if k.startswith(self.safety_prefix)]
        targets = [k for k in data.keys() if k.startswith(self.target_prefix)]
        comforts = [k for k in data.keys() if k.startswith(self.comfort_prefix)]
        # evaluation
        # bool valuation of safety satisfaction as conjunction (product) of individual satisfaction (k==max_steps)
        safety_aggregated = np.prod(np.array([data[s] == data["ep_lengths"] for s in safeties]), axis=0)
        assert safety_aggregated.shape == data[safeties[0]].shape, "unexpected shape of safety aggregation"
        # bool valuation of target satisfaction as conjunction (product) of individual satisfaction (k>0)
        target_aggregated = np.prod(np.array([data[t] > 0 for t in targets]), axis=0)
        assert target_aggregated.shape == data[targets[0]].shape, "unexpected shape of target aggregation"
        # comfort valuation as average (mean) of individual comfort satisfaction (k/max_steps)
        comfort_aggregated = np.mean(np.array([data[c] / MAX_STEPS[env_name] for c in comforts]), axis=0)
        assert comfort_aggregated.shape == data[comforts[0]].shape, "unexpected shape of comfort aggregation"
        # compute weighted results
        result = self.weights["safety"] * safety_aggregated + \
                 self.weights["target"] * target_aggregated + \
                 self.weights["comfort"] * comfort_aggregated
        return result


class TargetSafetyAndComfortEvalFunction(EvaluationFunction):
    safety_prefix = "s"
    target_prefix = "t_"    # be careful: also "timesteps" starts with "t"
    comfort_prefix = "c"


    def __init__(self, name: str):
        self.name = name

    def __call__(self, data: Dict[str, np.ndarray], env_name: str):
        # extract metrics names
        safeties = [k for k in data.keys() if k.startswith(self.safety_prefix)]
        targets = [k for k in data.keys() if k.startswith(self.target_prefix)]
        comforts = [k for k in data.keys() if k.startswith(self.comfort_prefix)]
        # evaluation
        # bool valuation of safety satisfaction as conjunction (product) of individual satisfaction (k==max_steps)
        safety_aggregated = np.prod(np.array([data[s] == data["ep_lengths"] for s in safeties]), axis=0)
        assert safety_aggregated.shape == data[safeties[0]].shape, "unexpected shape of safety aggregation"
        # bool valuation of target satisfaction as conjunction (product) of individual satisfaction (k>0)
        target_aggregated = np.prod(np.array([data[t] > 0 for t in targets]), axis=0)
        assert target_aggregated.shape == data[targets[0]].shape, "unexpected shape of target aggregation"
        # comfort valuation as average (mean) of individual comfort satisfaction (k/max_steps)
        comfort_aggregated = np.mean(np.array([data[c] / MAX_STEPS[env_name] for c in comforts]), axis=0)
        assert comfort_aggregated.shape == data[comforts[0]].shape, "unexpected shape of comfort aggregation"
        # compute weighted results
        result = safety_aggregated * target_aggregated + 0.5 * comfort_aggregated
        return result

class NormalizeEvalFunction(EvaluationFunction):

    def __init__(self, name: str, metric_prefix: str):
        self.name = name
        self.metric_prefix = metric_prefix

    def __call__(self, data: Dict[str, np.ndarray], env_name: str):
        metrics = [k for k in data.keys() if k.startswith(self.metric_prefix)]
        result = np.mean(np.array([data[m] / MAX_STEPS[env_name] for m in metrics]), axis=0)
        return result

class SafetyScore(EvaluationFunction):
    def __init__(self, name: str):
        self.name = name
        self.metric_prefix = "s"

    def __call__(self, data: Dict[str, np.ndarray], env_name: str):
        metrics = [k for k in data.keys() if k.startswith(self.metric_prefix)]
        # Note: safety is when all safety requirements are satisfied for the entire episode.
        # Since episodes can be shorter than the maximum number of steps (e.g., task completion),
        # we need to check that the safety requirements are satisfied for the episode length.
        result = np.all(np.array([data[m] == data["ep_lengths"] for m in metrics]), axis=0)
        return result

class TargetScore(EvaluationFunction):
    def __init__(self, name: str):
        self.name = name
        self.metric_prefix = "t_"

    def __call__(self, data: Dict[str, np.ndarray], env_name: str):
        metrics = [k for k in data.keys() if k.startswith(self.metric_prefix)]
        result = np.prod(np.array([data[m] > 0 for m in metrics]), axis=0)
        return result

class ComfortScore(EvaluationFunction):
    def __init__(self, name: str):
        self.name = name
        self.metric_prefix = "c"

    def __call__(self, data: Dict[str, np.ndarray], env_name: str):
        metrics = [k for k in data.keys() if k.startswith(self.metric_prefix)]
        result = np.mean(np.array([data[m] / MAX_STEPS[env_name] for m in metrics]), axis=0)
        return result

class SafetyTargetScore(EvaluationFunction):
    def __init__(self, name: str):
        self.name = name
        self.safety_metric_prefix = "s"
        self.target_metric_prefix = "t_"

    def __call__(self, data: Dict[str, np.ndarray], env_name: str):
        safety_metrics = [k for k in data.keys() if k.startswith(self.safety_metric_prefix)]
        target_metrics = [k for k in data.keys() if k.startswith(self.target_metric_prefix)]
        data_target = np.prod(np.array([data[m] > 0 for m in target_metrics]), axis=0)
        data_safety = np.all(np.array([data[m] == data["ep_lengths"] for m in safety_metrics]), axis=0)
        result = data_target * data_safety
        return result

class SafetyTargetComfortScore(EvaluationFunction):
    def __init__(self, name: str):
        self.name = name
        self.safety_metric_prefix = "s"
        self.target_metric_prefix = "t_"
        self.comfort_metric_prefix = "c"

    def __call__(self, data: Dict[str, np.ndarray], env_name: str):
        safety_metrics = [k for k in data.keys() if k.startswith(self.safety_metric_prefix)]
        target_metrics = [k for k in data.keys() if k.startswith(self.target_metric_prefix)]
        comfort_metrics = [k for k in data.keys() if k.startswith(self.comfort_metric_prefix)]

        data_target = np.prod(np.array([data[m] > 0 for m in target_metrics]), axis=0)
        data_safety = np.all(np.array([data[m] == data["ep_lengths"] for m in safety_metrics]), axis=0)
        data_comfort = np.mean(np.array([data[m] / MAX_STEPS[env_name] for m in comfort_metrics]), axis=0)

        result = data_target * data_safety * data_comfort
        return result


def register_custom_evaluation(name: str, fn_factory: Type[EvaluationFunction], kwargs: Dict[str, Any] = None):
    assert name not in _register, f"already exists a custom evaluation function with name {name}"
    kwargs = kwargs or {}
    _register[name] = fn_factory(name, **kwargs)


def get_custom_evaluation(name: str):
    assert name in _register, f"{name} is not a custom evaluation function"
    return _register[name]


register_custom_evaluation(name="eval_stc", fn_factory=WeightedEvalFunction,
                           kwargs={"safety_w": 1.0, "target_w": 0.5, "comfort_w": 0.25})

register_custom_evaluation(name="eval_tsc", fn_factory=WeightedEvalFunction,
                           kwargs={"safety_w": 0.5, "target_w": 1.0, "comfort_w": 0.25})

register_custom_evaluation(name="eval_txs_c", fn_factory=TargetSafetyAndComfortEvalFunction)

register_custom_evaluation(name="norm_s", fn_factory=NormalizeEvalFunction, kwargs={"metric_prefix": "s"})
register_custom_evaluation(name="norm_t", fn_factory=NormalizeEvalFunction, kwargs={"metric_prefix": "t_"})
register_custom_evaluation(name="norm_c", fn_factory=NormalizeEvalFunction, kwargs={"metric_prefix": "c"})

register_custom_evaluation(name="safety_score", fn_factory=SafetyScore)
register_custom_evaluation(name="target_score", fn_factory=TargetScore)
register_custom_evaluation(name="comfort_score", fn_factory=ComfortScore)
register_custom_evaluation(name="safety_target_score", fn_factory=SafetyTargetScore)
register_custom_evaluation(name="safety_target_comfort_score", fn_factory=SafetyTargetComfortScore)