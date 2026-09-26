"""Rebuild every released manuscript table from model-evaluation outputs.

The pipeline has two levels.  ``matrix`` and ``evaluations`` convert the large
external model outputs into the two released readout-level tables.  ``derived``
rebuilds every smaller table from those readout-level tables and the released
baseline response table.  ``verify`` checks that no unregistered CSV remains
in ``figures/tables``.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, rankdata, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests


AREAS = ["V1", "LM", "RL", "AL"]
CONDITIONS = ["contrast", "rotation", "translation", "temporal_reverse", "speed_up", "slow_down"]
KEYS = ["session", "scan_idx", "readout_id"]
STRESSES = {
    "contrast": 0.25,
    "rotation": 90.0,
    "translation": 48.0,
    "temporal_reverse": -1.0,
    "speed_up": 2.0,
    "slow_down": 0.5,
}
SUMMARY_METRICS = [
    "area_balanced_sparseness", "global_sparseness", "self_rank_percentile",
    "self_over_library_max", "home_minus_other_logmean", "effective_participation",
]
TABLE_NAMES = {
    "area_layer_transformation_summary.csv", "area_summary.csv",
    "baseline_paired_tests.csv", "layer_selectivity_summary.csv",
    "layer_selectivity_tests.csv",
    "nested_selectivity_tolerance_audit.csv", "neuron_family_metrics.csv.gz",
    "parameter_readouts.csv.gz",
    "quality_sensitivity_summary.csv", "response_area_summary.csv",
    "selectivity_tolerance_by_parameter.csv", "selectivity_tolerance_selected_parameters.csv",
    "selectivity_full_loo.csv", "selectivity_tolerance_loo_by_parameter.csv",
    "six_condition_parameter_omnibus_tests.csv", "six_condition_parameter_profiles.csv",
    "temporal_reverse_mechanism_screen.csv", "temporal_reverse_two_mechanisms.csv",
    "within_readout_normalized_area_blocks.csv",
    "hierarchy_consistency_selectivity.csv", "hierarchy_consistency_tolerance.csv",
    "adjacent_area_statistics.csv",
}


def _sparseness(total: np.ndarray, square: np.ndarray, count: int) -> np.ndarray:
    mean = total / count
    mean_square = square / count
    ratio = np.divide(mean * mean, mean_square, out=np.ones_like(mean), where=mean_square > 0)
    return (1 - ratio) / (1 - 1 / count)


def _leave_one_out_area_sparseness(
    area_total: dict[str, np.ndarray], area_square: dict[str, np.ndarray],
    area_count: dict[str, int], source_areas: np.ndarray, self_response: np.ndarray,
) -> np.ndarray:
    """Area-balanced sparseness after removing each target's own MEDI.

    The diagonal is removed only from the source-area pool containing the
    target readout.  Other source pools remain unchanged.  This is the exact
    leave-one-out statistic used by Figure S13 and is computed from streaming
    sums, so it never materializes another copy of the dense matrix.
    """
    scores = []
    for area in AREAS:
        total = area_total[area].copy()
        square = area_square[area].copy()
        count = area_count[area]
        target = source_areas == area
        total[target] -= self_response[target]
        square[target] -= self_response[target] ** 2
        count_for_target = np.full(len(source_areas), count, dtype=int)
        count_for_target[target] -= 1
        mean = np.divide(total, count_for_target, out=np.zeros_like(total), where=count_for_target > 0)
        mean_square = np.divide(square, count_for_target,
                                out=np.zeros_like(square), where=count_for_target > 0)
        ratio = np.divide(mean * mean, mean_square, out=np.ones_like(mean), where=mean_square > 0)
        scores.append(np.divide(1 - ratio, 1 - 1 / count_for_target,
                                out=np.zeros_like(ratio), where=count_for_target > 1))
    return np.mean(np.stack(scores), axis=0)


def _normalize_conditions(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "condition" not in data and "family" in data:
        data["condition"] = data["family"]
    speed = data.condition.eq("temporal_speed")
    data.loc[speed & data.parameter.gt(1), "condition"] = "speed_up"
    data.loc[speed & data.parameter.lt(1), "condition"] = "slow_down"
    data.loc[data.condition.eq("temporal_reverse"), "parameter"] = -1.0
    return data[data.condition.isin(CONDITIONS)].copy()


def build_matrix_tables(matrix_path: Path, manifest_path: Path, tables: Path, row_chunk: int = 256) -> None:
    """Build selection and response-block tables from the dense MEDI response matrix."""
    manifest = pd.read_csv(manifest_path).sort_values("task_index").reset_index(drop=True)
    matrix = np.load(matrix_path, mmap_mode="r")
    n = len(manifest)
    if matrix.shape != (n, n):
        raise ValueError(f"matrix shape {matrix.shape} does not match {n} manifest rows")
    source_areas = manifest.brain_area.astype(str).to_numpy()
    self_response = np.asarray(matrix[np.arange(n), np.arange(n)], dtype=np.float64)
    total = np.zeros(n); square = np.zeros(n); maximum = np.full(n, -np.inf)
    greater = np.zeros(n, dtype=np.int64)
    area_total = {area: np.zeros(n) for area in AREAS}
    area_square = {area: np.zeros(n) for area in AREAS}
    area_log = {area: np.zeros(n) for area in AREAS}
    area_count = {area: int(np.sum(source_areas == area)) for area in AREAS}
    for start in range(0, n, row_chunk):
        stop = min(start + row_chunk, n)
        values = np.asarray(matrix[start:stop], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("dense response matrix contains non-finite values")
        if np.nanmin(values) <= -1:
            raise ValueError("dense response matrix contains values <= -1, so log1p is undefined")
        total += values.sum(0); square += np.square(values).sum(0)
        maximum = np.maximum(maximum, values.max(0)); greater += (values > self_response).sum(0)
        chunk_areas = source_areas[start:stop]
        for area in AREAS:
            selected = values[chunk_areas == area]
            area_total[area] += selected.sum(0)
            area_square[area] += np.square(selected).sum(0)
            area_log[area] += np.log1p(selected).sum(0)
        print(f"matrix rows {start}:{stop}", flush=True)
    profiles = manifest.copy()
    profiles["self_response"] = self_response
    profiles["library_max"] = maximum
    profiles["self_over_library_max"] = np.divide(self_response, maximum,
                                                    out=np.full(n, np.nan), where=maximum > 0)
    profiles["self_rank_percentile"] = 1 - greater / max(n - 1, 1)
    profiles["global_sparseness"] = _sparseness(total, square, n)
    area_sparseness = {area: _sparseness(area_total[area], area_square[area], area_count[area]) for area in AREAS}
    profiles["area_balanced_sparseness"] = np.mean(np.stack(list(area_sparseness.values())), axis=0)
    profiles["area_balanced_sparseness_leave_self_out"] = _leave_one_out_area_sparseness(
        area_total, area_square, area_count, source_areas, self_response)
    profiles["effective_participation"] = np.divide(total * total, square, out=np.ones(n), where=square > 0)
    area_log_mean = {area: area_log[area] / area_count[area] for area in AREAS}
    profiles["home_minus_other_logmean"] = np.nan
    for area in AREAS:
        target = source_areas == area
        other = np.mean(np.stack([area_log_mean[x] for x in AREAS if x != area]), axis=0)
        profiles.loc[target, "home_minus_other_logmean"] = area_log_mean[area][target] - other[target]
    summary = profiles.groupby("brain_area")[SUMMARY_METRICS].agg(["count", "median", "mean", "std"])
    summary.to_csv(tables / "area_summary.csv")
    summary.to_csv(tables / "response_area_summary.csv")
    layer = profiles[profiles.layer.isin(["L23", "L2/3", "L4", "L5"])].copy()
    layer["layer"] = layer.layer.replace({"L23": "L2/3"})
    layer = layer.groupby(["brain_area", "layer"], as_index=False).area_balanced_sparseness.agg(
        n="size", sparseness="median")
    layer.rename(columns={"brain_area": "area"}).to_csv(tables / "layer_selectivity_summary.csv", index=False)
    source_log_means = np.stack([area_log_mean[area] for area in AREAS])
    z = ((source_log_means - source_log_means.mean(axis=0))
         / source_log_means.std(axis=0, ddof=0).clip(1e-12))
    rows = []
    for source_index, source in enumerate(AREAS):
        for target in AREAS:
            rows.append({"source_area": source, "target_area": target,
                         "median_within_readout_z": float(np.median(z[source_index, source_areas == target]))})
    pd.DataFrame(rows).to_csv(tables / "within_readout_normalized_area_blocks.csv", index=False)
    # Keep this intermediate beside other large inputs; it is not a published table.
    input_dir = tables.parent / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    profiles[[*KEYS, "area_balanced_sparseness", "area_balanced_sparseness_leave_self_out"]].to_csv(
        input_dir / "selectivity_profiles.csv.gz", index=False, compression="gzip"
    )
    loo_columns = [*KEYS, "brain_area", "self_response",
                   "area_balanced_sparseness", "area_balanced_sparseness_leave_self_out"]
    if "task_index" in profiles:
        loo_columns.insert(0, "task_index")
    if "unit_id" in profiles:
        loo_columns.insert(3, "unit_id")
    profiles[loo_columns].rename(
        columns={"self_response": "original_response",
                 "area_balanced_sparseness": "selectivity_full",
                 "area_balanced_sparseness_leave_self_out": "selectivity_leave_self_out"}
    ).to_csv(tables / "selectivity_full_loo.csv", index=False)


def build_evaluation_tables(evaluations_path: Path, selectivity_path: Path, tables: Path) -> None:
    """Build readout-level retention tables from raw transformed-stimulus evaluations."""
    raw = _normalize_conditions(pd.read_csv(evaluations_path))
    required = {*KEYS, "brain_area", "condition", "parameter", "response", "original_response"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"evaluation export lacks columns: {sorted(missing)}")
    if "retention" not in raw:
        raw["retention"] = raw.response / raw.original_response.clip(lower=1e-12)
    selectivity = pd.read_csv(selectivity_path, usecols=[*KEYS, "area_balanced_sparseness"])
    parameter = raw.groupby([*KEYS, "brain_area", "condition", "parameter"], as_index=False).retention.mean()
    parameter = parameter.merge(selectivity, on=KEYS, validate="many_to_one").rename(
        columns={"area_balanced_sparseness": "selectivity"})
    parameter.to_csv(tables / "parameter_readouts.csv.gz", index=False)
    metadata = [column for column in ["unit_id", "layer", "cc_max"] if column in raw]
    aggregations = {
        "n_variants": ("retention", "size"), "mean_retention": ("retention", "mean"),
        "median_retention": ("retention", "median"), "min_retention": ("retention", "min"),
        "max_retention": ("retention", "max"), "original_response": ("original_response", "first"),
    }
    if "pixel_distance" in raw:
        aggregations["mean_pixel_distance"] = ("pixel_distance", "mean")
    group_columns = [*KEYS, *metadata, "brain_area", "condition"]
    family = raw.groupby(group_columns, as_index=False).agg(**aggregations).rename(columns={"condition": "family"})
    family.to_csv(tables / "neuron_family_metrics.csv.gz", index=False)


def _bootstrap_difference(first: np.ndarray, second: np.ndarray, rng: np.random.Generator,
                          repeats: int) -> tuple[float, float, float]:
    estimate = float(np.median(first) - np.median(second))
    values = np.empty(repeats); batch = 250
    for start in range(0, repeats, batch):
        stop = min(start + batch, repeats)
        first_idx = rng.integers(0, len(first), (stop - start, len(first)))
        second_idx = rng.integers(0, len(second), (stop - start, len(second)))
        values[start:stop] = np.median(first[first_idx], axis=1) - np.median(second[second_idx], axis=1)
    low, high = np.quantile(values, [.025, .975])
    return estimate, float(low), float(high)


def build_derived_tables(repo: Path) -> None:
    """Rebuild every compact table from the two released readout-level tables."""
    tables = repo / "figures" / "tables"
    parameter = _normalize_conditions(pd.read_csv(tables / "parameter_readouts.csv.gz"))
    family = pd.read_csv(tables / "neuron_family_metrics.csv.gz")
    profiles = parameter.groupby(["brain_area", "condition", "parameter"], as_index=False).retention.agg(
        mean="mean", sd="std", median="median", n="size")
    profiles.to_csv(tables / "six_condition_parameter_profiles.csv", index=False)
    correlations = []
    omnibus = []
    for (area, condition, value), group in parameter.groupby(["brain_area", "condition", "parameter"]):
        rho, p_value = spearmanr(group.selectivity, group.retention)
        correlations.append({"area": area, "condition": condition, "parameter": value,
                             "n": len(group), "rho": rho, "p_value": p_value})
    for (condition, value), group in parameter.groupby(["condition", "parameter"]):
        samples = [part.retention.to_numpy() for _, part in group.groupby("brain_area")]
        statistic, p_value = kruskal(*samples)
        omnibus.append({"condition": condition, "parameter": value, "n_areas": len(samples),
                        "statistic": statistic, "p_value": p_value})
    pd.DataFrame(correlations).to_csv(tables / "selectivity_tolerance_by_parameter.csv", index=False)
    pd.DataFrame(omnibus).to_csv(tables / "six_condition_parameter_omnibus_tests.csv", index=False)
    chosen = select_lowest_median_retention_parameter(parameter)
    chosen.to_csv(tables / "selectivity_tolerance_selected_parameters.csv", index=False)
    metadata = family.drop_duplicates(KEYS)[[*KEYS, "layer"]]
    layered = parameter.merge(metadata, on=KEYS, validate="many_to_one")
    layered["layer"] = layered.layer.replace({"L2/3": "L23"})
    layered = layered[layered.layer.isin(["L23", "L4", "L5"])]
    layered.groupby(["brain_area", "layer", "condition", "parameter"], as_index=False).retention.agg(
        median_retention="median", n="size").to_csv(
            tables / "area_layer_transformation_summary.csv", index=False)
    _build_layer_selectivity_tests(parameter, metadata, tables)
    reverse = family[family.family.eq("temporal_reverse")].copy()
    if "mean_pixel_distance" in reverse:
        reverse["ret_q90"] = reverse.groupby("brain_area").mean_retention.transform("quantile", .9)
        reverse["pix_q75"] = reverse.groupby("brain_area").mean_pixel_distance.transform("quantile", .75)
        reverse["high_retention"] = reverse.mean_retention >= reverse.ret_q90
        reverse["high_pixel_change"] = reverse.mean_pixel_distance >= reverse.pix_q75
        reverse["mechanism"] = "other"
        reverse.loc[reverse.high_retention & ~reverse.high_pixel_change, "mechanism"] = "near-static candidate"
        reverse.loc[reverse.high_retention & reverse.high_pixel_change, "mechanism"] = "response-invariant despite large change"
        reverse.to_csv(tables / "temporal_reverse_two_mechanisms.csv", index=False)
        reverse.groupby(["brain_area", "high_retention"], as_index=False).agg(
            n=("readout_id", "size"), retention=("mean_retention", "median"),
            pixel_distance=("mean_pixel_distance", "median")).to_csv(
                tables / "temporal_reverse_mechanism_screen.csv", index=False)
    _build_nested(parameter, tables)
    _build_quality(parameter, family, tables)
    _build_hierarchy_consistency_selectivity(parameter, tables)
    _build_hierarchy_consistency_tolerance(parameter, repo, tables)
    _build_selected_loo_correlations(parameter, tables)
    _build_baseline_tests(repo / "data" / "baseline_responses.csv", tables)
    verify_registry(tables)


def _build_hierarchy_consistency_selectivity(parameter: pd.DataFrame, tables: Path) -> None:
    """Bootstrap compatibility of selectivity with V1 -> LM/RL -> AL."""
    data = parameter.drop_duplicates(KEYS)[["brain_area", "selectivity"]]
    values = {area: data.loc[data.brain_area.eq(area), "selectivity"].to_numpy() for area in AREAS}
    observed = {area: float(np.median(values[area])) for area in AREAS}
    rng = np.random.default_rng(20260926); repeats = 5000; increasing = 0; decreasing = 0
    for _ in range(repeats):
        med = {area: np.median(v[rng.integers(0, len(v), len(v))]) for area, v in values.items()}
        lo = min(med["LM"], med["RL"]); hi = max(med["LM"], med["RL"])
        increasing += int(med["V1"] <= lo and hi <= med["AL"])
        decreasing += int(med["V1"] >= hi and lo >= med["AL"])
    pd.DataFrame([
        {"pattern": "increasing", **{f"{a.lower()}_median": observed[a] for a in AREAS},
         "bootstrap_count": increasing, "bootstrap_replicates": repeats,
         "bootstrap_probability": increasing / repeats, "seed": 20260926},
        {"pattern": "decreasing", **{f"{a.lower()}_median": observed[a] for a in AREAS},
         "bootstrap_count": decreasing, "bootstrap_replicates": repeats,
         "bootstrap_probability": decreasing / repeats, "seed": 20260926},
    ]).to_csv(tables / "hierarchy_consistency_selectivity.csv", index=False)


def _build_hierarchy_consistency_tolerance(parameter: pd.DataFrame, repo: Path, tables: Path) -> None:
    """Bootstrap compatibility of tolerance with V1 -> LM/RL -> AL."""
    speed_path = repo / "figures" / "inputs" / "variable_speed_readout_summary.csv"
    ordinary = parameter[~parameter.condition.isin(["speed_up", "slow_down"])].groupby(
        [*KEYS, "brain_area", "condition"], as_index=False).retention.median()
    speed = pd.read_csv(speed_path)
    speed["condition"] = np.where(speed.speed_factor > 1, "speed_up", "slow_down")
    speed = speed.groupby([*KEYS, "brain_area", "condition"], as_index=False).retention.median()
    readout = pd.concat([ordinary, speed], ignore_index=True)
    conditions = CONDITIONS; repeats = 5000; rng = np.random.default_rng(20260927); rows = []
    for condition in conditions:
        values = {area: readout.loc[(readout.condition == condition) & (readout.brain_area == area), "retention"].dropna().to_numpy(float) for area in AREAS}
        med = {area: float(np.median(values[area])) for area in AREAS}
        increasing_count = decreasing_count = 0
        for start in range(0, repeats, 250):
            n = min(250, repeats - start)
            boot = {area: np.median(rng.choice(values[area], (n, len(values[area])), replace=True), axis=1) for area in AREAS}
            increasing_count += int(((boot["V1"] <= np.minimum(boot["LM"], boot["RL"])) & (np.maximum(boot["LM"], boot["RL"]) <= boot["AL"])).sum())
            decreasing_count += int(((boot["V1"] >= np.maximum(boot["LM"], boot["RL"])) & (np.minimum(boot["LM"], boot["RL"]) >= boot["AL"])).sum())
        for direction in ["increasing", "decreasing"]:
            rows.append({"transformation": condition, "direction": direction,
                         **{f"{a}_median": med[a] for a in AREAS},
                         "compatible_bootstrap_count": increasing_count if direction == "increasing" else decreasing_count,
                         "compatible_probability": (increasing_count if direction == "increasing" else decreasing_count) / repeats,
                         "bootstrap_replicates": repeats, "seed": 20260927})
    pd.DataFrame(rows).to_csv(tables / "hierarchy_consistency_tolerance.csv", index=False)


def select_lowest_median_retention_parameter(parameter: pd.DataFrame) -> pd.DataFrame:
    """Select one stress per condition by pooled readout median retention."""
    pooled = parameter.groupby(["condition", "parameter"], as_index=False).retention.median()
    chosen = pooled.loc[pooled.groupby("condition").retention.idxmin()].copy()
    return chosen.rename(columns={"retention": "global_median_retention"})


def _build_layer_selectivity_tests(
    parameter: pd.DataFrame, metadata: pd.DataFrame, tables: Path
) -> None:
    """Test selectivity differences among cortical layers within each area."""
    readouts = parameter.drop_duplicates(KEYS)[[*KEYS, "brain_area", "selectivity"]]
    data = readouts.merge(metadata, on=KEYS, validate="one_to_one")
    data["layer"] = data.layer.replace({"L23": "L2/3"})
    data = data[data.layer.isin(["L2/3", "L4", "L5"])]
    rows = []
    for area, group in data.groupby("brain_area", sort=False):
        samples = {
            layer: group.loc[group.layer.eq(layer), "selectivity"].to_numpy()
            for layer in ["L2/3", "L4", "L5"]
        }
        statistic, p_value = kruskal(*samples.values())
        rows.append({
            "area": area, "test": "Kruskal-Wallis", "comparison": "L2/3 vs L4 vs L5",
            "n_first": len(samples["L2/3"]), "n_second": len(samples["L4"]),
            "n_third": len(samples["L5"]), "statistic": statistic,
            "median_difference": np.nan, "rank_biserial": np.nan, "p_value": p_value,
        })
        for layer in ["L4", "L5"]:
            first, second = samples["L2/3"], samples[layer]
            statistic, p_value = mannwhitneyu(first, second, alternative="two-sided")
            rows.append({
                "area": area, "test": "Mann-Whitney U", "comparison": f"L2/3 vs {layer}",
                "n_first": len(first), "n_second": len(second), "n_third": np.nan,
                "statistic": statistic, "median_difference": np.median(first) - np.median(second),
                "rank_biserial": 2 * statistic / (len(first) * len(second)) - 1,
                "p_value": p_value,
            })
    output = pd.DataFrame(rows)
    for test in output.test.unique():
        selected = output.test.eq(test)
        output.loc[selected, "p_fdr_bh"] = multipletests(
            output.loc[selected, "p_value"], method="fdr_bh"
        )[1]
    output.to_csv(tables / "layer_selectivity_tests.csv", index=False)


def _build_nested(data: pd.DataFrame, tables: Path) -> None:
    keys = data.session.astype(str) + ":" + data.scan_idx.astype(str) + ":" + data.readout_id.astype(str)
    data = data.assign(half=keys.map(lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 2))
    rows = []
    for selection_half in [0, 1]:
        for condition in CONDITIONS:
            source = data[(data.half == selection_half) & data.condition.eq(condition)]
            selected = source.groupby("parameter").retention.median().idxmin()
            test = data[(data.half != selection_half) & data.condition.eq(condition) & data.parameter.eq(selected)]
            for area in AREAS:
                group = test[test.brain_area.eq(area)]
                rho, p_value = spearmanr(group.selectivity, group.retention)
                x, y = rankdata(group.selectivity), rankdata(group.retention)
                rng = np.random.default_rng(20260901 + len(rows))
                boot = [np.corrcoef(x[index], y[index])[0, 1] for index in
                        (rng.integers(0, len(group), len(group)) for _ in range(5000))]
                low, high = np.quantile(boot, [.025, .975])
                rows.append({"selection_half": selection_half, "test_half": 1 - selection_half,
                             "condition": condition, "selected_parameter": selected, "area": area,
                             "n_test": len(group), "rho": rho, "p_value": p_value,
                             "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                             "bootstrap_replicates": 5000})
    output = pd.DataFrame(rows)
    output["p_fdr_bh"] = multipletests(output.p_value, method="fdr_bh")[1]
    output.to_csv(tables / "nested_selectivity_tolerance_audit.csv", index=False)


def _build_quality(parameter: pd.DataFrame, family: pd.DataFrame, tables: Path) -> None:
    meta_columns = [*KEYS, "original_response", "cc_max"]
    meta = family[meta_columns].drop_duplicates(KEYS)
    data = parameter.merge(meta, on=KEYS, validate="many_to_one")
    quantiles = data.original_response.quantile([.05, .10, .25])
    masks = {"all": data.original_response.notna(),
             "original_ge_q05": data.original_response >= quantiles.loc[.05],
             "original_ge_q10": data.original_response >= quantiles.loc[.10],
             "original_ge_q25": data.original_response >= quantiles.loc[.25],
             "cc_ge_0.3": data.cc_max >= .3, "cc_ge_0.5": data.cc_max >= .5,
             "cc_ge_0.7": data.cc_max >= .7}
    rng = np.random.default_rng(20260902); rows = []
    for name, mask in masks.items():
        selected = data[mask].groupby([*KEYS, "brain_area", "condition"], as_index=False).retention.median()
        for condition in CONDITIONS:
            v1 = selected[selected.condition.eq(condition) & selected.brain_area.eq("V1")].retention.to_numpy()
            for area in ["LM", "RL", "AL"]:
                values = selected[selected.condition.eq(condition) & selected.brain_area.eq(area)].retention.to_numpy()
                estimate, low, high = _bootstrap_difference(values, v1, rng, 5000)
                rows.append({"filter": name, "family": condition, "brain_area": area,
                             "n_area": len(values), "n_v1": len(v1), "contrast_vs_v1": estimate,
                             "ci_low": low, "ci_high": high, "bootstrap_replicates": 5000})
    pd.DataFrame(rows).to_csv(tables / "quality_sensitivity_summary.csv", index=False)


def _build_selected_loo_correlations(parameter: pd.DataFrame, tables: Path) -> None:
    loo = pd.read_csv(tables / "selectivity_full_loo.csv")
    chosen = select_lowest_median_retention_parameter(parameter)
    rows = []
    for condition in CONDITIONS:
        value = chosen.loc[chosen.condition.eq(condition), "parameter"].iloc[0]
        data = parameter[parameter.condition.eq(condition) & parameter.parameter.eq(value)]
        data = data.merge(
            loo[[*KEYS, "selectivity_leave_self_out"]],
            on=KEYS, validate="many_to_one",
        )
        for area, group in data.groupby("brain_area", sort=True):
            rho, p_value = spearmanr(group.selectivity_leave_self_out, group.retention)
            rows.append({"area": area, "condition": condition, "parameter": value,
                         "n": len(group), "rho": rho, "p_value": p_value})
    pd.DataFrame(rows).to_csv(tables / "selectivity_tolerance_loo_by_parameter.csv", index=False)


def _build_baseline_tests(path: Path, tables: Path) -> None:
    data = pd.read_csv(path)
    comparisons = [("MEDI > dynamic Gabor", "medi_response", "dynamic_gabor_response"),
                   ("MEDI > static Gabor", "medi_response", "static_gabor_response"),
                   ("MEDI > MESI", "medi_response", "MESI"),
                   ("MEDI > drifting grating", "medi_response", "grating_response"),
                   ("MEDI > natural", "medi_response", "Natural"),
                   ("MESI > static Gabor", "MESI", "static_gabor_response")]
    rows = []
    for label, first, second in comparisons:
        pair = data.dropna(subset=[first, second])[[first, second]]
        result = wilcoxon(pair[first], pair[second], alternative="greater")
        rows.append({"comparison": label, "n": len(pair), "wilcoxon_statistic": result.statistic,
                     "p_value": result.pvalue})
    pd.DataFrame(rows).to_csv(tables / "baseline_paired_tests.csv", index=False)


def verify_registry(tables: Path) -> None:
    present = {path.name for path in tables.iterdir() if path.is_file()}
    unknown = present - TABLE_NAMES - {"_selectivity_profiles.csv.gz"}
    missing = TABLE_NAMES - present
    if unknown or missing:
        raise RuntimeError(f"table registry mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["matrix", "evaluations", "derived", "verify"])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--evaluations", type=Path)
    parser.add_argument("--selectivity", type=Path)
    parser.add_argument("--row-chunk", type=int, default=256)
    args = parser.parse_args()
    tables = args.repo / "figures" / "tables"; tables.mkdir(parents=True, exist_ok=True)
    if args.command == "matrix":
        if not args.matrix or not args.manifest:
            parser.error("matrix requires --matrix and --manifest")
        build_matrix_tables(args.matrix, args.manifest, tables, args.row_chunk)
    elif args.command == "evaluations":
        if not args.evaluations:
            parser.error("evaluations requires --evaluations")
        selectivity = args.selectivity or args.repo / "figures" / "inputs" / "selectivity_profiles.csv.gz"
        build_evaluation_tables(args.evaluations, selectivity, tables)
    elif args.command == "derived":
        build_derived_tables(args.repo)
    else:
        verify_registry(tables)
    print(f"completed table pipeline command: {args.command}")


if __name__ == "__main__":
    main()
