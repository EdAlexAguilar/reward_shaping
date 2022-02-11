from abc import abstractmethod
from typing import Dict, Callable, Any
import numpy as np

_register = {}


class EvaluationFunction:
    @abstractmethod
    def __call__(self, data: Dict[str, np.ndarray], **kwargs):
        pass


class WeightedEvalFunction(EvaluationFunction):
    safety_prefix = "s"
    target_prefix = "t_"    # be careful: also "timesteps" starts with "t"
    comfort_prefix = "c"
    max_steps = {"cart_pole_obst": 401,
                 "bipedal_walker": 1600,
                 "lunar_lander": 600}

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
        comfort_aggregated = np.mean(np.array([data[c] / self.max_steps[env_name] for c in comforts]), axis=0)
        assert comfort_aggregated.shape == data[comforts[0]].shape, "unexpected shape of comfort aggregation"
        # compute weighted results
        result = self.weights["safety"] * safety_aggregated + \
                 self.weights["target"] * target_aggregated + \
                 self.weights["comfort"] * comfort_aggregated
        return result


def register_custom_evaluation(name: str, fn_factory: EvaluationFunction, kwargs: Dict[str, Any]):
    assert name not in _register, f"already exists a custom evaluation function with name {name}"
    _register[name] = fn_factory(name, **kwargs)


def get_custom_evaluation(name: str):
    assert name in _register, f"{name} is not a custom evaluation function"
    return _register[name]


register_custom_evaluation(name="eval_stc", fn_factory=WeightedEvalFunction,
                           kwargs={"safety_w": 1.0, "target_w": 0.5, "comfort_w": 0.25})

register_custom_evaluation(name="eval_tsc", fn_factory=WeightedEvalFunction,
                           kwargs={"safety_w": 0.5, "target_w": 1.0, "comfort_w": 0.25})
