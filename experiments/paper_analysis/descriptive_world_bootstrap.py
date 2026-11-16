"""Additional retrospective failure-rate intervals, not new hypothesis tests.

Frozen behavior RESULT files already contain prospective fact-accuracy and any
one-correct coreference intervals. This routine does NOT replace those. It adds
missing listed-wrong-object and listed-wrong-coreference rate intervals using
whole worlds and a separately declared retrospective seed. No p-values.
"""

import math
import random


def intervals(behavior_results):
    worlds = sorted(next(iter(behavior_results.values()))["world_summaries"])
    if len(worlds) != 32 or any(
        sorted(result["world_summaries"]) != worlds
        for result in behavior_results.values()
    ):
        raise ValueError(
            "The same32 development worlds must be retained for each model"
        )
    rng = random.Random(20261115)
    draws = [[rng.randrange(32) for _ in range(32)] for _ in range(2000)]

    def quantile(samples, fraction):
        position = fraction * (len(samples) - 1)
        low, high = math.floor(position), math.ceil(position)
        return samples[low] + (samples[high] - samples[low]) * (position - low)

    models = {}
    for model, result in behavior_results.items():
        summaries = [result["world_summaries"][identity] for identity in worlds]
        if any(
            item["fact_accuracy"]["count"] != 72 or item["coref_pairs"] != 24
            for item in summaries
        ):
            raise ValueError(
                "Each world must retain72 factual questions and24 paired-name comparisons"
            )
        rates = {
            "wrong_object_fact_rate": [item["wrong_object"] / 72 for item in summaries],
            "eligible_wrong_object_coref_rate": [
                item["wrong_object_coref"] / 24 for item in summaries
            ],
        }
        models[model] = {}
        for metric, values in rates.items():
            samples = sorted(
                sum(values[index] for index in draw) / 32 for draw in draws
            )
            models[model][metric] = {
                "mean": sum(values) / 32,
                "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
                "worlds": 32,
            }
    return {
        "retrospective_descriptive": True,
        "seed": 20261115,
        "resamples": 2000,
        "unit": "whole world, retaining all repeated conditions/questions",
        "percentile_interpolation": "linear positions p*(B-1)",
        "same_world_draws_for_both_models": True,
        "p_values_computed": False,
        "models": models,
        "scope": "Conditional descriptive development intervals, not population-risk bounds or confirmation. A degenerate zero interval does not prove zero risk.",
    }
