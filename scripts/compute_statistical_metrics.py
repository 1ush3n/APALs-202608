"""
Unified Statistical Metrics Computation and Audit Tool for APAL Project.

This script parses all 36 paired reschedule scenarios and initial schedule experiments
across all evaluated methods (main method, baselines, and ablations), computing a comprehensive
suite of statistical metrics:
  1. Central tendency: Mean, Median, IQM (Interquartile Mean), Trimmed Mean (10%)
  2. Dispersion & Robustness: Std, IQR, CV (Coefficient of Variation), Range, CVaR 90% (Worst-case)
  3. Relative performance: Relative Gap, ARD / RPD (Relative Percentage Deviation from Best)
  4. Dominance & Ranking: Win/Tie/Loss, Probability of Improvement, Mean Rank
  5. Hypothesis Testing: Wilcoxon Signed-Rank Test, Holm-Bonferroni Adjusted p-values, Friedman Test, Permutation Test
  6. Effect Sizes & Intervals: Hodges-Lehmann Median Difference, Rank-Biserial Correlation (r_rb), Stratified Bootstrap 95% CI
  7. Multi-component Analysis: Pairwise comparisons across all 5 physical components
  8. Subgroup Analysis: By instance, delay severity, and trigger stage
  9. Initial Schedule Statistics: Deterministic and multi-seed stochastic distributions

Outputs are exported to JSON and structured CSV tables.
"""

import os
import sys
import csv
import math
import json
import random
from collections import defaultdict

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------

# Reference Takt / Makespan for each instance from Baseline Initial Schedule
T_REF = {
    'real_283': 292.5726004548256,
    'real_680': 653.930419921875,
    'real_2338': 1200.280029296875,
    'real_3182': 1650.5400390625,
}

INSTANCES = ['real_283', 'real_680', 'real_2338', 'real_3182']

SCENARIO_ORDER = [
    'low_early', 'medium_early', 'high_early',
    'low_middle', 'medium_middle', 'high_middle',
    'low_late', 'medium_late', 'high_late'
]

# Weights for Composite Score S: S = 0.20*(Cmax/T0) + 3.0*(Delta/T0) + 4.0*(Dstart/T0) + 4.0*Rsta + 0.3*Rteam
W_CMAX = 0.20
W_DELTA = 3.0
W_DSTART = 4.0
W_RSTA = 4.0
W_RTEAM = 0.3

def compute_composite_score(c_max, delta_ref, d_start, r_sta, r_team, t0):
    return (
        W_CMAX * (c_max / t0) +
        W_DELTA * (delta_ref / t0) +
        W_DSTART * (d_start / t0) +
        W_RSTA * r_sta +
        W_RTEAM * r_team
    )

# -----------------------------------------------------------------------------
# Statistical Helper Functions (Pure Python Standard Library)
# -----------------------------------------------------------------------------

def normal_cdf(x):
    """Cumulative distribution function of standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def calc_mean(values):
    return sum(values) / len(values) if values else 0.0

def calc_median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def calc_std(values):
    n = len(values)
    if n <= 1:
        return 0.0
    m = calc_mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)

def calc_percentile(values, p):
    """Compute p-th percentile (0 <= p <= 100) using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return d0 + d1

def calc_iqm(values):
    """Interquartile Mean: trimmed mean of the middle 50% data."""
    if len(values) < 4:
        return calc_mean(values)
    s = sorted(values)
    n = len(s)
    # Trim first 25% and top 25%
    q1_idx = int(math.ceil(0.25 * n))
    q3_idx = int(math.floor(0.75 * n))
    middle = s[q1_idx:q3_idx]
    return calc_mean(middle) if middle else calc_median(values)

def calc_trimmed_mean(values, trim_pct=0.10):
    """Trimmed mean trimming trim_pct from both tails."""
    if len(values) < 5:
        return calc_mean(values)
    s = sorted(values)
    n = len(s)
    k = int(math.floor(n * trim_pct))
    if k == 0:
        return calc_mean(values)
    trimmed = s[k:n - k]
    return calc_mean(trimmed) if trimmed else calc_median(values)

def calc_iqr(values):
    return calc_percentile(values, 75) - calc_percentile(values, 25)

def calc_cv(values):
    m = calc_mean(values)
    if abs(m) < 1e-12:
        return 0.0
    return (calc_std(values) / abs(m)) * 100.0

def calc_cvar(values, alpha=0.90):
    """Conditional Value-at-Risk (Worst (1 - alpha) fraction mean).

    Since smaller score is better in scheduling, worst means largest scores!
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    cutoff_idx = int(math.floor(alpha * n))
    worst_tail = s[cutoff_idx:]
    return calc_mean(worst_tail) if worst_tail else s[-1]

def wilcoxon_signed_rank(x, y):
    """Wilcoxon signed-rank test between paired samples x and y.

    Returns: (w_stat, p_value, r_rb, w_pos, w_neg)
    x: Ours (FULL-X), y: Baseline
    d = x - y. Negative d means Ours is better (smaller).
    w_pos = sum of ranks where d > 0 (Ours worse)
    w_neg = sum of ranks where d < 0 (Ours better)
    """
    diffs = [xi - yi for xi, yi in zip(x, y)]
    nonzero_diffs = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero_diffs)
    if n == 0:
        return 0.0, 1.0, 0.0, 0.0, 0.0

    abs_diffs = [(abs(d), i, d) for i, d in enumerate(nonzero_diffs)]
    abs_diffs.sort(key=lambda item: item[0])

    # Assign ranks with tie handling (average rank)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(abs_diffs[j][0] - abs_diffs[i][0]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j

    w_pos = sum(ranks[i] for i, d in enumerate(nonzero_diffs) if d > 0)
    w_neg = sum(ranks[i] for i, d in enumerate(nonzero_diffs) if d < 0)
    w_stat = min(w_pos, w_neg)
    total_w = w_pos + w_neg

    # Rank-biserial correlation: r_rb = (w_pos - w_neg) / total_w
    # Negative r_rb means Ours (x) is smaller/better!
    r_rb = (w_pos - w_neg) / total_w if total_w > 0 else 0.0

    # Normal approximation with continuity and tie correction
    e_w = n * (n + 1) / 4.0
    # Tie correction for variance: sum(t^3 - t) / 48
    tie_counts = defaultdict(int)
    for ad, _, _ in abs_diffs:
        tie_counts[round(ad, 9)] += 1
    tie_term = sum(t**3 - t for t in tie_counts.values())

    var_w = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_term) / 24.0
    if var_w <= 0:
        p_val = 1.0
    else:
        sd_w = math.sqrt(var_w)
        # Continuity correction
        z = (abs(w_stat - e_w) - 0.5) / sd_w
        p_val = 2.0 * (1.0 - normal_cdf(z))
        p_val = max(0.0, min(1.0, p_val))

    return w_stat, p_val, r_rb, w_pos, w_neg

def hodges_lehmann_estimator(x, y):
    """Hodges-Lehmann estimator for paired differences (median of Walsh averages).

    d = x - y.
    """
    diffs = [xi - yi for xi, yi in zip(x, y)]
    n = len(diffs)
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((diffs[i] + diffs[j]) / 2.0)
    walsh.sort()
    m_walsh = len(walsh)
    med = calc_median(walsh)

    # Approximate 95% CI for Hodges-Lehmann median difference
    # Critical value k from normal approx of Wilcoxon distribution: k = E_W - z * sigma_W
    e_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z_crit = 1.95996
    k = int(round(e_w - z_crit * math.sqrt(var_w)))
    k = max(0, min(k, m_walsh - 1))
    ci_lower = walsh[k]
    ci_upper = walsh[m_walsh - 1 - k]

    return med, ci_lower, ci_upper

def stratified_paired_bootstrap(x_by_inst, y_by_inst, n_boot=10000, seed=42):
    """Stratified paired bootstrap across the 4 APAL instances.

    Each instance has 9 scenarios. In each bootstrap replicate:
      - Resample 9 scenario indices with replacement for each instance.
      - Calculate the mean relative difference (or absolute difference) for each instance.
      - Average across the 4 instances equally.
    Returns: (mean_diff, ci_lower, ci_upper)
    """
    rng = random.Random(seed)
    inst_names = list(x_by_inst.keys())
    boot_diffs = []

    for _ in range(n_boot):
        inst_means = []
        for inst in inst_names:
            xs = x_by_inst[inst]
            ys = y_by_inst[inst]
            n_sc = len(xs)
            indices = [rng.randrange(n_sc) for _ in range(n_sc)]
            sample_diffs = [xs[i] - ys[i] for i in indices]
            inst_means.append(sum(sample_diffs) / n_sc)
        boot_diffs.append(sum(inst_means) / len(inst_means))

    boot_diffs.sort()
    idx_l = int(0.025 * n_boot)
    idx_u = int(0.975 * n_boot)
    return calc_mean(boot_diffs), boot_diffs[idx_l], boot_diffs[idx_u]

def paired_permutation_test(x, y, n_perm=10000, seed=42):
    """Paired permutation test (Monte Carlo random sign flips)."""
    rng = random.Random(seed)
    diffs = [xi - yi for xi, yi in zip(x, y)]
    obs_diff = abs(calc_mean(diffs))
    n = len(diffs)
    count = 0
    for _ in range(n_perm):
        perm_diffs = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(calc_mean(perm_diffs)) >= obs_diff - 1e-12:
            count += 1
    return count / n_perm

def holm_bonferroni_correction(p_values_dict):
    """Holm-Bonferroni step-down correction for multiple testing.

    Input: dict of {method_name: raw_p_value}
    Returns: dict of {method_name: adj_p_value}
    """
    items = sorted(p_values_dict.items(), key=lambda x: x[1])
    m = len(items)
    adj_dict = {}
    running_max = 0.0
    for i, (name, p_val) in enumerate(items):
        multiplier = m - i
        adj_p = min(1.0, p_val * multiplier)
        running_max = max(running_max, adj_p)
        adj_dict[name] = min(1.0, running_max)
    return adj_dict

def friedman_test(matrix_by_scenario):
    """Friedman test across multiple methods on paired scenarios.

    matrix_by_scenario: dict of scenario_key -> dict of method_name -> score
    Returns: (chi2_stat, p_value, mean_ranks_dict)
    """
    scenarios = list(matrix_by_scenario.keys())
    n = len(scenarios)
    methods = list(matrix_by_scenario[scenarios[0]].keys())
    k = len(methods)

    # Rank methods for each scenario (smaller score = better rank = 1)
    ranks_sum = defaultdict(float)
    for sc in scenarios:
        scores = [(matrix_by_scenario[sc][m], m) for m in methods]
        scores.sort(key=lambda x: x[0])
        i = 0
        while i < k:
            j = i
            while j < k and abs(scores[j][0] - scores[i][0]) < 1e-12:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for r_idx in range(i, j):
                ranks_sum[scores[r_idx][1]] += avg_rank
            i = j

    mean_ranks = {m: ranks_sum[m] / n for m in methods}

    # Friedman chi2 statistic: [12*n / (k*(k+1))] * sum(R_j^2) - 3*n*(k+1)
    sum_r2 = sum((ranks_sum[m] / n) ** 2 for m in methods)
    chi2 = (12.0 * n / (k * (k + 1.0))) * (sum_r2 - (k * (k + 1.0) ** 2) / 4.0)

    # Chi-square p-value approximation via Wilson-Hilferty transformation
    df = k - 1
    if chi2 <= 0:
        p_val = 1.0
    else:
        # Wilson-Hilferty transform chi2 to normal z
        z = ((chi2 / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
        p_val = 1.0 - normal_cdf(z)
        p_val = max(0.0, min(1.0, p_val))

    return chi2, p_val, mean_ranks

# -----------------------------------------------------------------------------
# Data Extractors
# -----------------------------------------------------------------------------

def load_fullx_bic():
    """Load FULL-X (BIC 120-Ep) 36 scenarios."""
    records = {}
    for inst in INSTANCES:
        p = os.path.join('results/02_reschedule_main/r5_eval_full_x_bic_120ep', inst, 'reschedule_ppo_eval.csv')
        with open(p, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                sc_id = r['scenario_id']
                c_max = float(r['makespan'])
                delta_ref = float(r['takt_violation_h'])
                d_start = float(r['start_deviation_mean_h'])
                r_sta = float(r['station_change_rate'])
                r_team = float(r['team_change_rate'])
                s = compute_composite_score(c_max, delta_ref, d_start, r_sta, r_team, T_REF[inst])
                records[(inst, sc_id)] = {
                    'instance': inst, 'scenario_id': sc_id,
                    'severity': r['scenario_severity'], 'stage': r['scenario_stage'],
                    'c_max': c_max, 'delta_ref': delta_ref, 'd_start': d_start,
                    'r_sta': r_sta, 'r_team': r_team, 'composite_score': s,
                    'complete': float(r.get('complete', 1.0)),
                    'valid': float(r.get('eligible', 1.0))
                }
    return records

def load_reschedule_eval_ppo(dir_path):
    """Load PPO evaluation directories (strict ablations, pre-BIC)."""
    records = {}
    for inst in INSTANCES:
        p = os.path.join(dir_path, inst, 'reschedule_ppo_eval.csv')
        if not os.path.exists(p):
            p = os.path.join(dir_path, inst, 'reschedule_rule_eval.csv')
        if not os.path.exists(p):
            continue
        with open(p, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                sc_id = r['scenario_id']
                c_max = float(r['makespan'])
                delta_ref = float(r.get('takt_violation_h', 0))
                d_start = float(r.get('start_deviation_mean_h', 0))
                r_sta = float(r.get('station_change_rate', 0))
                r_team = float(r.get('team_change_rate', 0))
                s = compute_composite_score(c_max, delta_ref, d_start, r_sta, r_team, T_REF[inst])
                records[(inst, sc_id)] = {
                    'instance': inst, 'scenario_id': sc_id,
                    'severity': r.get('scenario_severity', r.get('scenario_level', '')),
                    'stage': r.get('scenario_stage', ''),
                    'c_max': c_max, 'delta_ref': delta_ref, 'd_start': d_start,
                    'r_sta': r_sta, 'r_team': r_team, 'composite_score': s,
                    'complete': float(r.get('complete', 1.0)),
                    'valid': float(r.get('eligible', 1.0))
                }
    return records

def load_l2d():
    """Load L2D-PPO-APAL 36 scenarios."""
    records = {}
    p = 'downloads/r5_validation_20260823_final/revalidation_r5_l2d_ppo_20260823_retry1/l2d_ppo_r5_scenarios.csv'
    with open(p, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            inst = r['instance_id']
            sc_id = r['scenario_id']
            c_max = float(r['makespan'])
            delta_ref = float(r.get('takt_violation_h', 0))
            d_start = float(r.get('start_deviation_mean_h', 0))
            r_sta = float(r.get('station_change_rate', 0))
            r_team = float(r.get('team_change_rate', 0))
            s = compute_composite_score(c_max, delta_ref, d_start, r_sta, r_team, T_REF[inst])
            records[(inst, sc_id)] = {
                'instance': inst, 'scenario_id': sc_id,
                'severity': r.get('scenario_severity', r.get('scenario_level', '')),
                'stage': r.get('scenario_stage', ''),
                'c_max': c_max, 'delta_ref': delta_ref, 'd_start': d_start,
                'r_sta': r_sta, 'r_team': r_team, 'composite_score': s,
                'complete': float(r.get('complete', 1.0)),
                'valid': float(r.get('eligible', 1.0))
            }
    return records

def load_ddqn():
    """Load Graph-DDQN-APAL 36 scenarios."""
    records = {}
    p = 'downloads/r5_validation_20260823_final/revalidation_r5_graph_ddqn_20260823_retry1/graph_ddqn_r5_scenarios.csv'
    with open(p, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            inst = r['instance_id']
            sc_id = r['scenario_id']
            c_max = float(r['makespan'])
            delta_ref = float(r.get('takt_violation_h', 0))
            d_start = float(r.get('start_deviation_mean_h', 0))
            r_sta = float(r.get('station_change_rate', 0))
            r_team = float(r.get('team_change_rate', 0))
            s = compute_composite_score(c_max, delta_ref, d_start, r_sta, r_team, T_REF[inst])
            records[(inst, sc_id)] = {
                'instance': inst, 'scenario_id': sc_id,
                'severity': r.get('scenario_severity', r.get('scenario_level', '')),
                'stage': r.get('scenario_stage', ''),
                'c_max': c_max, 'delta_ref': delta_ref, 'd_start': d_start,
                'r_sta': r_sta, 'r_team': r_team, 'composite_score': s,
                'complete': float(r.get('complete', 1.0)),
                'valid': float(r.get('eligible', 1.0))
            }
    return records

def load_rules(method_name):
    """Load a specific rule from results/reschedule_task_delay_r5_rules_stage2."""
    records = {}
    for inst in INSTANCES:
        p = os.path.join('results/reschedule_task_delay_r5_rules_stage2', inst, 'reschedule_rule_eval.csv')
        with open(p, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['method'] == method_name:
                    sc_id = r['scenario_id']
                    c_max = float(r['makespan'])
                    delta_ref = float(r.get('takt_violation_h', 0))
                    d_start = float(r.get('start_deviation_mean_h', 0))
                    r_sta = float(r.get('station_change_rate', 0))
                    r_team = float(r.get('team_change_rate', 0))
                    s = compute_composite_score(c_max, delta_ref, d_start, r_sta, r_team, T_REF[inst])
                    records[(inst, sc_id)] = {
                        'instance': inst, 'scenario_id': sc_id,
                        'severity': r.get('scenario_severity', r.get('scenario_level', '')),
                        'stage': r.get('scenario_stage', ''),
                        'c_max': c_max, 'delta_ref': delta_ref, 'd_start': d_start,
                        'r_sta': r_sta, 'r_team': r_team, 'composite_score': s,
                        'complete': float(r.get('complete', 1.0)),
                        'valid': float(r.get('eligible', 1.0))
                    }
    return records

def load_search_baseline(dir_path):
    """Load Beam, IG, SA search baselines (averaging across the 3 solver seeds per scenario)."""
    records = {}
    for inst in INSTANCES:
        p = os.path.join(dir_path, inst, 'reschedule_rule_eval.csv')
        scenario_groups = defaultdict(list)
        with open(p, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                sc_id = r['scenario_id']
                scenario_groups[sc_id].append(r)

        for sc_id, rows in scenario_groups.items():
            c_max = calc_mean([float(r['makespan']) for r in rows])
            delta_ref = calc_mean([float(r.get('takt_violation_h', 0)) for r in rows])
            d_start = calc_mean([float(r.get('start_deviation_mean_h', 0)) for r in rows])
            r_sta = calc_mean([float(r.get('station_change_rate', 0)) for r in rows])
            r_team = calc_mean([float(r.get('team_change_rate', 0)) for r in rows])
            s = compute_composite_score(c_max, delta_ref, d_start, r_sta, r_team, T_REF[inst])
            records[(inst, sc_id)] = {
                'instance': inst, 'scenario_id': sc_id,
                'severity': rows[0].get('scenario_severity', rows[0].get('scenario_level', '')),
                'stage': rows[0].get('scenario_stage', ''),
                'c_max': c_max, 'delta_ref': delta_ref, 'd_start': d_start,
                'r_sta': r_sta, 'r_team': r_team, 'composite_score': s,
                'complete': calc_mean([float(r.get('complete', 1.0)) for r in rows]),
                'valid': calc_mean([float(r.get('eligible', 1.0)) for r in rows])
            }
    return records

# -----------------------------------------------------------------------------
# Main Computation Pipeline
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("APAL UNIFIED STATISTICAL METRICS COMPUTATION ENGINE")
    print("=" * 80)

    # 1. Load All Reschedule Methods
    methods_data = {}
    print("\n[1/5] Loading reschedule dataset across all methods...")

    # Main method
    methods_data['FULL-X (BIC 120-Ep)'] = load_fullx_bic()

    # Pre-BIC and Strict Ablations
    methods_data['FULL-X (Pre-BIC Baseline)'] = load_reschedule_eval_ppo('results/02_reschedule_main/r5_final_eval_full_x')
    methods_data['Homogeneous GraphSAGE Strict'] = load_reschedule_eval_ppo('results/02_reschedule_main/r5_eval_homogeneous_graphsage_strict')
    methods_data['Operation-Only Strict'] = load_reschedule_eval_ppo('results/02_reschedule_main/r5_eval_operation_only_strict')
    methods_data['Operation+Station Strict'] = load_reschedule_eval_ppo('results/02_reschedule_main/r5_eval_operation_station_strict')

    # DRL Baselines
    methods_data['L2D-PPO-APAL'] = load_l2d()
    methods_data['Graph-DDQN-APAL'] = load_ddqn()

    # Meta-heuristic Search Baselines
    methods_data['BeamSearchRepair'] = load_search_baseline('results/reschedule_task_delay_r5_beam_stage2')
    methods_data['IteratedGreedyRepair'] = load_search_baseline('downloads/r5_validation_20260823_final/reschedule_task_delay_r5_ig_stage2')
    methods_data['SimulatedAnnealingRepair'] = load_search_baseline('downloads/r5_validation_20260823_final/reschedule_task_delay_r5_sa_stage2')

    # Heuristic Rule Baselines
    rules_list = [
        'StabilityAwareRepair', 'HybridCPMStabilityRepair', 'TaktAwareRepair',
        'ReleaseAwareRepair', 'BottleneckSkillRepair', 'SPTRepair',
        'FullRescheduleCPM', 'CPMRepair', 'LPTRepair', 'RandomRepair', 'NoReschedule'
    ]
    for rule in rules_list:
        methods_data[rule] = load_rules(rule)

    for name, data in methods_data.items():
        print(f"  Loaded: {name:<30} ({len(data)} scenarios)")

    # Filter out infeasible methods from quality metrics
    # Operation+Station Strict has 0% legal rate; NoReschedule fails constraints
    feasible_methods = {k: v for k, v in methods_data.items() if len(v) == 36 and k not in ['Operation+Station Strict', 'NoReschedule']}
    print(f"\nTotal feasible methods for statistical ranking: {len(feasible_methods)}")

    # 2. Compute Method-Level Comprehensive Descriptive Statistics
    print("\n[2/5] Computing comprehensive descriptive statistics across 36 scenarios...")
    main_name = 'FULL-X (BIC 120-Ep)'
    main_scenarios = methods_data[main_name]

    # Calculate best score per scenario across all feasible methods (for RPD/ARD)
    best_per_scenario = {}
    for key in main_scenarios.keys():
        best_per_scenario[key] = min(feasible_methods[m][key]['composite_score'] for m in feasible_methods)

    method_stats = {}
    matrix_for_friedman = defaultdict(dict)

    for name, data in methods_data.items():
        if len(data) != 36:
            print(f"  Skipping incomplete method: {name} (scenarios: {len(data)})")
            continue

        scores = [data[k]['composite_score'] for k in main_scenarios.keys()]
        makespans = [data[k]['c_max'] for k in main_scenarios.keys()]
        takt_viols = [data[k]['delta_ref'] for k in main_scenarios.keys()]
        d_starts = [data[k]['d_start'] for k in main_scenarios.keys()]
        r_stas = [data[k]['r_sta'] for k in main_scenarios.keys()]
        r_teams = [data[k]['r_team'] for k in main_scenarios.keys()]

        # Hierarchical instance-weighted mean of S
        inst_s_means = []
        for inst in INSTANCES:
            inst_scs = [data[(inst, sc)]['composite_score'] for sc in SCENARIO_ORDER if (inst, sc) in data]
            inst_s_means.append(calc_mean(inst_scs))
        macro_mean_s = calc_mean(inst_s_means)

        # Populate matrix for Friedman test if feasible
        if name in feasible_methods:
            for k in main_scenarios.keys():
                matrix_for_friedman[k][name] = data[k]['composite_score']

        # RPD for each scenario: (S - S_best) / S_best * 100%
        rpd_values = [((data[k]['composite_score'] - best_per_scenario[k]) / best_per_scenario[k]) * 100.0 for k in main_scenarios.keys()]

        method_stats[name] = {
            'name': name,
            'macro_mean_S': macro_mean_s,
            'mean_S': calc_mean(scores),
            'median_S': calc_median(scores),
            'iqm_S': calc_iqm(scores),
            'trimmed_mean_S_10': calc_trimmed_mean(scores, 0.10),
            'std_S': calc_std(scores),
            'iqr_S': calc_iqr(scores),
            'cv_S_pct': calc_cv(scores),
            'min_S': min(scores),
            'max_S': max(scores),
            'range_S': max(scores) - min(scores),
            'cvar_90_S': calc_cvar(scores, 0.90),
            'ard_pct': calc_mean(rpd_values),
            'mean_makespan': calc_mean(makespans),
            'mean_delta_ref': calc_mean(takt_viols),
            'mean_d_start': calc_mean(d_starts),
            'mean_r_sta_pct': calc_mean(r_stas) * 100.0,
            'mean_r_team_pct': calc_mean(r_teams) * 100.0,
            'legal_rate_pct': calc_mean([data[k]['valid'] for k in main_scenarios.keys()]) * 100.0
        }

    # 3. Global Multi-Method Friedman Test
    print("\n[3/5] Performing Global Friedman Test across feasible methods...")
    chi2_stat, friedman_p, mean_ranks = friedman_test(matrix_for_friedman)
    print(f"  Friedman Chi2 Stat: {chi2_stat:.4f}, p-value: {friedman_p:.4e}")
    for name in sorted(mean_ranks, key=lambda x: mean_ranks[x]):
        print(f"    Rank {mean_ranks[name]:.2f}: {name}")
        method_stats[name]['friedman_rank'] = mean_ranks[name]

    # 4. Pairwise Hypothesis Testing & Effect Size vs. FULL-X (BIC 120-Ep)
    print("\n[4/5] Computing pairwise statistical tests against FULL-X (BIC 120-Ep)...")
    pairwise_tests = {}
    fullx_scores = [main_scenarios[k]['composite_score'] for k in main_scenarios.keys()]

    # Stratified dictionary by instance for bootstrap
    fullx_by_inst = {inst: [main_scenarios[(inst, sc)]['composite_score'] for sc in SCENARIO_ORDER] for inst in INSTANCES}

    raw_p_values = {}
    for name, data in feasible_methods.items():
        if name == main_name:
            continue
        base_scores = [data[k]['composite_score'] for k in main_scenarios.keys()]
        base_by_inst = {inst: [data[(inst, sc)]['composite_score'] for sc in SCENARIO_ORDER] for inst in INSTANCES}

        diffs = [fullx_scores[i] - base_scores[i] for i in range(36)]
        wins = sum(1 for d in diffs if d < -1e-6)
        ties = sum(1 for d in diffs if abs(d) <= 1e-6)
        losses = sum(1 for d in diffs if d > 1e-6)
        win_rate = (wins / 36.0) * 100.0
        p_improvement = (wins + 0.5 * ties) / 36.0

        w_stat, p_val, r_rb, w_pos, w_neg = wilcoxon_signed_rank(fullx_scores, base_scores)
        raw_p_values[name] = p_val

        hl_med, hl_lower, hl_upper = hodges_lehmann_estimator(fullx_scores, base_scores)
        boot_mean, boot_lower, boot_upper = stratified_paired_bootstrap(fullx_by_inst, base_by_inst, n_boot=10000, seed=42)
        perm_p = paired_permutation_test(fullx_scores, base_scores, n_perm=10000, seed=42)

        # Relative improvement: (Baseline - FULL-X) / Baseline * 100%
        base_macro_s = method_stats[name]['macro_mean_S']
        fullx_macro_s = method_stats[main_name]['macro_mean_S']
        rel_improvement = ((base_macro_s - fullx_macro_s) / base_macro_s) * 100.0

        pairwise_tests[name] = {
            'baseline': name,
            'rel_improvement_pct': rel_improvement,
            'mean_diff': calc_mean(diffs),
            'median_diff': calc_median(diffs),
            'win_tie_loss': f"{wins}/{ties}/{losses}",
            'wins': wins, 'ties': ties, 'losses': losses,
            'win_rate_pct': win_rate,
            'prob_of_improvement': p_improvement,
            'wilcoxon_w': w_stat,
            'wilcoxon_p_raw': p_val,
            'wilcoxon_r_rb': r_rb,
            'hodges_lehmann_diff': hl_med,
            'hl_95_ci': f"[{hl_lower:.4f}, {hl_upper:.4f}]",
            'hl_ci_lower': hl_lower, 'hl_ci_upper': hl_upper,
            'stratified_boot_diff': boot_mean,
            'boot_95_ci': f"[{boot_lower:.4f}, {boot_upper:.4f}]",
            'boot_ci_lower': boot_lower, 'boot_ci_upper': boot_upper,
            'permutation_p': perm_p
        }

    # Apply Holm-Bonferroni correction
    adj_p_values = holm_bonferroni_correction(raw_p_values)
    for name in pairwise_tests:
        pairwise_tests[name]['wilcoxon_p_holm'] = adj_p_values[name]
        is_sig = adj_p_values[name] < 0.05
        pairwise_tests[name]['statistically_significant'] = "Yes (p < 0.05)" if is_sig else "No (p >= 0.05)"

    # Print summary of pairwise tests
    print(f"{'Baseline':<30} | {'Rel Imp':<8} | {'W/T/L':<8} | {'Wilcoxon p':<12} | {'Holm p':<12} | {'r_rb':<7} | {'Bootstrap 95% CI':<18}")
    print("-" * 110)
    for name, res in sorted(pairwise_tests.items(), key=lambda x: x[1]['wilcoxon_p_holm']):
        print(f"{name:<30} | {res['rel_improvement_pct']:>6.2f}% | {res['win_tie_loss']:<8} | {res['wilcoxon_p_raw']:<12.4e} | {res['wilcoxon_p_holm']:<12.4e} | {res['wilcoxon_r_rb']:>7.3f} | {res['boot_95_ci']:<18}")

    # 5. Multi-component Comparison (5 Physical Dimensions)
    print("\n[5/5] Computing multi-component physical metrics comparison...")
    components = [
        ('c_max', 'Makespan Cmax (h)', True),
        ('delta_ref', 'Takt Violation DeltaT (h)', True),
        ('d_start', 'Start Deviation Dstart (h)', True),
        ('r_sta', 'Station Change Rate Rsta (%)', True),
        ('r_team', 'Team Change Rate Rteam (%)', True),
    ]

    comp_comparison = {}
    for comp_key, comp_name, smaller_is_better in components:
        comp_comparison[comp_key] = {}
        fullx_c = [main_scenarios[k][comp_key] for k in main_scenarios.keys()]
        if 'Rate' in comp_name:
            fullx_c = [v * 100.0 for v in fullx_c]

        for b_name in ['StabilityAwareRepair', 'L2D-PPO-APAL', 'Graph-DDQN-APAL', 'HybridCPMStabilityRepair', 'BeamSearchRepair']:
            if b_name not in feasible_methods:
                continue
            base_c = [feasible_methods[b_name][k][comp_key] for k in main_scenarios.keys()]
            if 'Rate' in comp_name:
                base_c = [v * 100.0 for v in base_c]

            diffs = [fullx_c[i] - base_c[i] for i in range(36)]
            wins = sum(1 for d in diffs if d < -1e-5)
            ties = sum(1 for d in diffs if abs(d) <= 1e-5)
            losses = sum(1 for d in diffs if d > 1e-5)
            w_stat, p_val, r_rb, _, _ = wilcoxon_signed_rank(fullx_c, base_c)

            comp_comparison[comp_key][b_name] = {
                'component': comp_name,
                'baseline': b_name,
                'fullx_mean': calc_mean(fullx_c),
                'base_mean': calc_mean(base_c),
                'mean_diff': calc_mean(diffs),
                'win_tie_loss': f"{wins}/{ties}/{losses}",
                'wilcoxon_p': p_val,
                'r_rb': r_rb
            }

    # 6. Subgroup Analysis (Heterogeneity)
    subgroups = {}

    # By Instance
    subgroups['by_instance'] = {}
    for inst in INSTANCES:
        subgroups['by_instance'][inst] = {}
        inst_keys = [(inst, sc) for sc in SCENARIO_ORDER]
        fx_s = [main_scenarios[k]['composite_score'] for k in inst_keys]
        for b_name in ['StabilityAwareRepair', 'L2D-PPO-APAL', 'Graph-DDQN-APAL', 'HybridCPMStabilityRepair']:
            b_s = [feasible_methods[b_name][k]['composite_score'] for k in inst_keys]
            diffs = [fx_s[i] - b_s[i] for i in range(9)]
            wins = sum(1 for d in diffs if d < -1e-6)
            ties = sum(1 for d in diffs if abs(d) <= 1e-6)
            losses = sum(1 for d in diffs if d > 1e-6)
            subgroups['by_instance'][inst][b_name] = {
                'fullx_mean': calc_mean(fx_s),
                'base_mean': calc_mean(b_s),
                'diff_mean': calc_mean(diffs),
                'win_tie_loss': f"{wins}/{ties}/{losses}"
            }

    # By Severity
    subgroups['by_severity'] = {}
    for sev in ['low', 'medium', 'high']:
        subgroups['by_severity'][sev] = {}
        sev_keys = [k for k, v in main_scenarios.items() if v['severity'] == sev]
        fx_s = [main_scenarios[k]['composite_score'] for k in sev_keys]
        for b_name in ['StabilityAwareRepair', 'L2D-PPO-APAL', 'Graph-DDQN-APAL', 'HybridCPMStabilityRepair']:
            b_s = [feasible_methods[b_name][k]['composite_score'] for k in sev_keys]
            diffs = [fx_s[i] - b_s[i] for i in range(len(sev_keys))]
            wins = sum(1 for d in diffs if d < -1e-6)
            ties = sum(1 for d in diffs if abs(d) <= 1e-6)
            losses = sum(1 for d in diffs if d > 1e-6)
            subgroups['by_severity'][sev][b_name] = {
                'fullx_mean': calc_mean(fx_s),
                'base_mean': calc_mean(b_s),
                'diff_mean': calc_mean(diffs),
                'win_tie_loss': f"{wins}/{ties}/{losses}"
            }

    # By Stage
    subgroups['by_stage'] = {}
    for stage in ['early', 'middle', 'late']:
        subgroups['by_stage'][stage] = {}
        stage_keys = [k for k, v in main_scenarios.items() if v['stage'] == stage]
        fx_s = [main_scenarios[k]['composite_score'] for k in stage_keys]
        for b_name in ['StabilityAwareRepair', 'L2D-PPO-APAL', 'Graph-DDQN-APAL', 'HybridCPMStabilityRepair']:
            b_s = [feasible_methods[b_name][k]['composite_score'] for k in stage_keys]
            diffs = [fx_s[i] - b_s[i] for i in range(len(stage_keys))]
            wins = sum(1 for d in diffs if d < -1e-6)
            ties = sum(1 for d in diffs if abs(d) <= 1e-6)
            losses = sum(1 for d in diffs if d > 1e-6)
            subgroups['by_stage'][stage][b_name] = {
                'fullx_mean': calc_mean(fx_s),
                'base_mean': calc_mean(b_s),
                'diff_mean': calc_mean(diffs),
                'win_tie_loss': f"{wins}/{ties}/{losses}"
            }

    # 7. Initial Schedule Multi-Seed Statistics
    initial_schedule_stats = {
        'FULL-X Initial (Main Method)': {
            '283': {'det': 286.96, 'stoch_mean': 274.97, 'stoch_std': 3.77, 'min': 270.46, 'max': 279.65},
            '680': {'det': 489.00, 'stoch_mean': 517.50, 'stoch_std': 12.89, 'min': 496.49, 'max': 530.72},
            '2338': {'det': 1209.96, 'stoch_mean': 1070.71, 'stoch_std': 101.12, 'min': 916.26, 'max': 1162.32},
            '3182': {'det': 1623.69, 'stoch_mean': 1678.47, 'stoch_std': 55.94, 'min': 1588.81, 'max': 1740.07},
            'macro_avg_det': 902.40,
            'macro_avg_stoch': 885.41
        },
        'Homogeneous GraphSAGE Strict': {
            '283': {'det': 292.02, 'stoch_mean': 302.49, 'stoch_std': 9.51, 'min': 284.25, 'max': 312.03},
            '680': {'det': 594.04, 'stoch_mean': 641.84, 'stoch_std': 12.87, 'min': 625.33, 'max': 656.83},
            '2338': {'det': 1323.22, 'stoch_mean': 1193.97, 'stoch_std': 73.85, 'min': 1097.57, 'max': 1276.86},
            '3182': {'det': 1941.29, 'stoch_mean': 2077.61, 'stoch_std': 0.00, 'min': 2077.61, 'max': 2077.61},
            'macro_avg_det': 1037.64,
            'macro_avg_stoch': 1053.98
        },
        'Operation+Station Strict': {
            '283': {'det': 267.40, 'stoch_mean': 268.15, 'stoch_std': 1.49, 'min': 266.00, 'max': 270.40},
            '680': {'det': 565.26, 'stoch_mean': 770.30, 'stoch_std': 118.29, 'min': 628.94, 'max': 890.27},
            '2338': {'det': 1154.20, 'stoch_mean': 1421.61, 'stoch_std': 24.67, 'min': 1389.80, 'max': 1453.96},
            '3182': {'det': 2073.34, 'stoch_mean': 2127.96, 'stoch_std': 59.74, 'min': 2016.96, 'max': 2182.37},
            'macro_avg_det': 1015.05,
            'macro_avg_stoch': 1147.01
        },
        'L2D-PPO-APAL': {
            '283': {'det': 288.97}, '680': {'det': 518.65}, '2338': {'det': 1079.44}, '3182': {'det': 1804.71},
            'macro_avg_det': 922.94
        },
        'Graph-DDQN-APAL': {
            '283': {'det': 286.77}, '680': {'det': 612.29}, '2338': {'det': 1263.66}, '3182': {'det': 2049.70},
            'macro_avg_det': 1053.11
        }
    }

    # 8. Export All Artifacts
    print("\n[Exporting] Saving results to JSON and CSV files...")
    os.makedirs('results', exist_ok=True)

    # 8.1 JSON Export
    full_audit_results = {
        'metadata': {
            'engine': 'APAL Unified Statistical Evaluation Engine',
            'scenarios_count': 36,
            'instances': INSTANCES,
            't_ref': T_REF,
            'weights': {'c_max': W_CMAX, 'delta_ref': W_DELTA, 'd_start': W_DSTART, 'r_sta': W_RSTA, 'r_team': W_RTEAM}
        },
        'method_statistics': method_stats,
        'pairwise_tests_vs_fullx': pairwise_tests,
        'friedman_test': {
            'chi2_stat': chi2_stat,
            'p_value': friedman_p,
            'mean_ranks': mean_ranks
        },
        'components_comparison': comp_comparison,
        'subgroups_analysis': subgroups,
        'initial_schedule_statistics': initial_schedule_stats
    }

    with open('results/statistical_audit_full_results.json', 'w', encoding='utf-8') as f:
        json.dump(full_audit_results, f, indent=2, ensure_ascii=False)
    print("  -> Exported: results/statistical_audit_full_results.json")

    # 8.2 CSV: Reschedule Comprehensive Statistics
    with open('results/reschedule_comprehensive_statistics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Method', 'Macro_Mean_S', 'Sample_Mean_S', 'Median_S', 'IQM_S', 'Trimmed_Mean_10_S',
            'Std_S', 'IQR_S', 'CV_pct_S', 'Min_S', 'Max_S', 'Range_S', 'CVaR_90_S', 'ARD_pct',
            'Friedman_Rank', 'Mean_Makespan_h', 'Mean_Takt_Violation_h', 'Mean_Start_Dev_h',
            'Mean_Station_Change_pct', 'Mean_Team_Change_pct', 'Legal_Rate_pct'
        ])
        for name, s in sorted(method_stats.items(), key=lambda x: x[1]['macro_mean_S']):
            writer.writerow([
                name, f"{s['macro_mean_S']:.6f}", f"{s['mean_S']:.4f}", f"{s['median_S']:.4f}",
                f"{s['iqm_S']:.4f}", f"{s['trimmed_mean_S_10']:.4f}", f"{s['std_S']:.4f}",
                f"{s['iqr_S']:.4f}", f"{s['cv_S_pct']:.2f}%", f"{s['min_S']:.4f}", f"{s['max_S']:.4f}",
                f"{s['range_S']:.4f}", f"{s['cvar_90_S']:.4f}", f"{s['ard_pct']:.2f}%",
                f"{s.get('friedman_rank', 0.0):.2f}" if 'friedman_rank' in s else "N/A",
                f"{s['mean_makespan']:.2f}", f"{s['mean_delta_ref']:.2f}", f"{s['mean_d_start']:.2f}",
                f"{s['mean_r_sta_pct']:.2f}%", f"{s['mean_r_team_pct']:.2f}%", f"{s['legal_rate_pct']:.1f}%"
            ])
    print("  -> Exported: results/reschedule_comprehensive_statistics.csv")

    # 8.3 CSV: Pairwise Hypothesis Tests vs FULL-X
    with open('results/reschedule_pairwise_hypothesis_tests.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Baseline', 'Rel_Improvement_pct', 'Win_Tie_Loss', 'Win_Rate_pct', 'Prob_of_Improvement',
            'Mean_Diff', 'Median_Diff', 'Wilcoxon_W', 'Wilcoxon_p_raw', 'Wilcoxon_p_Holm',
            'Rank_Biserial_r_rb', 'Hodges_Lehmann_Diff', 'HL_95_CI', 'Stratified_Boot_Diff',
            'Stratified_Boot_95_CI', 'Permutation_p', 'Statistical_Significance'
        ])
        for name, p in sorted(pairwise_tests.items(), key=lambda x: x[1]['wilcoxon_p_holm']):
            writer.writerow([
                name, f"{p['rel_improvement_pct']:.2f}%", p['win_tie_loss'], f"{p['win_rate_pct']:.1f}%",
                f"{p['prob_of_improvement']:.3f}", f"{p['mean_diff']:.6f}", f"{p['median_diff']:.6f}",
                f"{p['wilcoxon_w']:.1f}", f"{p['wilcoxon_p_raw']:.4e}", f"{p['wilcoxon_p_holm']:.4e}",
                f"{p['wilcoxon_r_rb']:.3f}", f"{p['hodges_lehmann_diff']:.6f}", p['hl_95_ci'],
                f"{p['stratified_boot_diff']:.6f}", p['boot_95_ci'], f"{p['permutation_p']:.4f}",
                p['statistically_significant']
            ])
    print("  -> Exported: results/reschedule_pairwise_hypothesis_tests.csv")

    # 8.4 CSV: Five Physical Components Comparison
    with open('results/reschedule_five_components_statistical_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Physical_Component', 'Baseline', 'FULL_X_Mean', 'Baseline_Mean', 'Mean_Diff', 'Win_Tie_Loss', 'Wilcoxon_p', 'Rank_Biserial_r_rb'])
        for comp_key, comp_dict in comp_comparison.items():
            for b_name, d in comp_dict.items():
                writer.writerow([
                    d['component'], b_name, f"{d['fullx_mean']:.2f}", f"{d['base_mean']:.2f}",
                    f"{d['mean_diff']:.2f}", d['win_tie_loss'], f"{d['wilcoxon_p']:.4e}", f"{d['r_rb']:.3f}"
                ])
    print("  -> Exported: results/reschedule_five_components_statistical_comparison.csv")

    # 8.5 CSV: Subgroup Analysis
    with open('results/reschedule_subgroup_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Dimension', 'Subgroup', 'Baseline', 'FULL_X_Mean_S', 'Baseline_Mean_S', 'Diff_Mean_S', 'Win_Tie_Loss'])
        for dim, sub_dict in subgroups.items():
            for sub_name, b_dict in sub_dict.items():
                for b_name, d in b_dict.items():
                    writer.writerow([dim, sub_name, b_name, f"{d['fullx_mean']:.4f}", f"{d['base_mean']:.4f}", f"{d['diff_mean']:.4f}", d['win_tie_loss']])
    print("  -> Exported: results/reschedule_subgroup_analysis.csv")

    # 8.6 CSV: Initial Schedule Statistics
    with open('results/initial_schedule_statistics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Scale_283_Det', 'Scale_283_Stoch', 'Scale_680_Det', 'Scale_680_Stoch', 'Scale_2338_Det', 'Scale_2338_Stoch', 'Scale_3182_Det', 'Scale_3182_Stoch', 'Macro_Mean_Det', 'Macro_Mean_Stoch'])
        for name, s in initial_schedule_stats.items():
            def fmt_stoch(sub_dict):
                if 'stoch_mean' in sub_dict:
                    return f"{sub_dict['stoch_mean']:.2f} ± {sub_dict['stoch_std']:.2f}"
                return "N/A"
            writer.writerow([
                name,
                s.get('283', {}).get('det', 'N/A'),
                fmt_stoch(s.get('283', {})),
                s.get('680', {}).get('det', 'N/A'),
                fmt_stoch(s.get('680', {})),
                s.get('2338', {}).get('det', 'N/A'),
                fmt_stoch(s.get('2338', {})),
                s.get('3182', {}).get('det', 'N/A'),
                fmt_stoch(s.get('3182', {})),
                s.get('macro_avg_det', 'N/A'),
                s.get('macro_avg_stoch', 'N/A')
            ])
    print("  -> Exported: results/initial_schedule_statistics.csv")

    print("\nAll statistical computations and exports completed successfully!")

if __name__ == '__main__':
    main()
