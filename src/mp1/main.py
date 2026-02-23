"""
Data Mining MP1 - Weather Data Analysis
Covers: Distance Metrics, Mahalanobis Distance, Similarity, Entropy, Mutual Information
"""

import csv
import math
import os

# ─────────────────────────────────────────────
# Helper: Load decoded CSV
# ─────────────────────────────────────────────
def load_decoded_data(filepath):
    """Load decoded.csv and return list of dicts with numeric values."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                'no': int(row['Data Entry 2'].strip()),
                'outlook': int(row['outlook'].strip()),
                'temp': int(row['temp'].strip()),
                'humidity': int(row['humidity'].strip()),
                'windy': int(row['windy'].strip()),
                'play': int(row['play'].strip()),
            }
            data.append(entry)
    return data


def get_feature_vector(row):
    """Return the 5-dimensional feature vector for a data point."""
    return [row['outlook'], row['temp'], row['humidity'], row['windy'], row['play']]


# ─────────────────────────────────────────────
# Distance Functions
# ─────────────────────────────────────────────
def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def manhattan_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def minkowski_distance(a, b, r):
    if r == float('inf'):
        return max(abs(ai - bi) for ai, bi in zip(a, b))
    return sum(abs(ai - bi) ** r for ai, bi in zip(a, b)) ** (1.0 / r)


# ─────────────────────────────────────────────
# Problem 1: Euclidean, Manhattan, Minkowski
# ─────────────────────────────────────────────
def solve_problem1(data):
    print("=" * 70)
    print("PROBLEM 1: Distance Metrics")
    print("=" * 70)

    features = ['outlook', 'temp', 'humidity', 'windy', 'play']
    n = len(data)

    # Calculate mean of entire dataset
    mean_vec = []
    for feat in features:
        feat_mean = sum(row[feat] for row in data) / n
        mean_vec.append(feat_mean)

    print(f"\nDataset size: {n}")
    print(f"Mean vector: {mean_vec}")
    print(f"  outlook={mean_vec[0]:.4f}, temp={mean_vec[1]:.4f}, humidity={mean_vec[2]:.4f}, "
          f"windy={mean_vec[3]:.4f}, play={mean_vec[4]:.4f}")

    for dp_no in [7, 70]:
        dp = data[dp_no - 1]  # 0-indexed
        dp_vec = get_feature_vector(dp)
        print(f"\n--- Data Point {dp_no}: {dp_vec} ---")

        # (a) Euclidean
        ed = euclidean_distance(mean_vec, dp_vec)
        print(f"  (a) Euclidean Distance: {ed:.6f}")
        # Show breakdown
        sq_diffs = [(mean_vec[i] - dp_vec[i]) ** 2 for i in range(5)]
        print(f"      Squared diffs: {[f'{d:.6f}' for d in sq_diffs]}")
        print(f"      Sum = {sum(sq_diffs):.6f}, sqrt = {ed:.6f}")

        # (b) Manhattan
        md = manhattan_distance(mean_vec, dp_vec)
        print(f"  (b) Manhattan Distance: {md:.6f}")
        abs_diffs = [abs(mean_vec[i] - dp_vec[i]) for i in range(5)]
        print(f"      Abs diffs: {[f'{d:.6f}' for d in abs_diffs]}")
        print(f"      Sum = {md:.6f}")

        # (c) Minkowski r=5
        mk5 = minkowski_distance(mean_vec, dp_vec, 5)
        print(f"  (c) Minkowski (r=5): {mk5:.6f}")
        pow5_diffs = [abs(mean_vec[i] - dp_vec[i]) ** 5 for i in range(5)]
        print(f"      |diff|^5: {[f'{d:.6f}' for d in pow5_diffs]}")
        print(f"      Sum = {sum(pow5_diffs):.6f}, 5th root = {mk5:.6f}")

        # (c) Minkowski r=infinity (Chebyshev)
        mk_inf = minkowski_distance(mean_vec, dp_vec, float('inf'))
        print(f"  (c) Minkowski (r=∞ / Chebyshev): {mk_inf:.6f}")
        print(f"      Max of abs diffs: {[f'{d:.6f}' for d in abs_diffs]} → max = {mk_inf:.6f}")


# ─────────────────────────────────────────────
# Problem 2: Mahalanobis Distance
# ─────────────────────────────────────────────
def matrix_transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def matrix_multiply(a, b):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    assert cols_a == rows_b
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def matrix_inverse_3x3(m):
    """Compute inverse of a 3x3 matrix using cofactor method."""
    a, b, c = m[0][0], m[0][1], m[0][2]
    d, e, f = m[1][0], m[1][1], m[1][2]
    g, h, i = m[2][0], m[2][1], m[2][2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("Matrix is singular, cannot invert")

    inv_det = 1.0 / det

    cofactors = [
        [(e * i - f * h), -(d * i - f * g), (d * h - e * g)],
        [-(b * i - c * h), (a * i - c * g), -(a * h - b * g)],
        [(b * f - c * e), -(a * f - c * d), (a * e - b * d)],
    ]

    inverse = [[cofactors[i][j] * inv_det for j in range(3)] for i in range(3)]
    return inverse


def solve_problem2(data):
    print("\n" + "=" * 70)
    print("PROBLEM 2: Mahalanobis Distance")
    print("=" * 70)

    # Use decoded data for rows 1-8 (only outlook, temp, humidity)
    subset = data[:8]
    features = ['outlook', 'temp', 'humidity']

    # Data matrix (8 x 3)
    X = [[row[f] for f in features] for row in subset]
    n = len(X)
    p = len(features)

    print(f"\nData (rows 1-8, columns: {features}):")
    for i, row in enumerate(X):
        print(f"  Row {i+1}: {row}")

    # Step 1: Compute mean vector
    mean = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    print(f"\nStep 1: Mean vector = {[f'{m:.4f}' for m in mean]}")

    # Step 2: Center the data (X - mean)
    X_centered = [[X[i][j] - mean[j] for j in range(p)] for i in range(n)]
    print(f"\nStep 2: Centered data (X - mean):")
    for i, row in enumerate(X_centered):
        print(f"  Row {i+1}: [{', '.join(f'{v:.4f}' for v in row)}]")

    # Step 3: Covariance matrix (using n-1 for sample covariance)
    cov = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            cov[i][j] = sum(X_centered[k][i] * X_centered[k][j] for k in range(n)) / (n - 1)

    print(f"\nStep 3: Covariance matrix (using n-1 = {n-1}):")
    for row in cov:
        print(f"  [{', '.join(f'{v:.6f}' for v in row)}]")

    # Step 4: Inverse of covariance matrix
    cov_inv = matrix_inverse_3x3(cov)
    print(f"\nStep 4: Inverse covariance matrix:")
    for row in cov_inv:
        print(f"  [{', '.join(f'{v:.6f}' for v in row)}]")

    # Verify: C * C_inv should be ~identity
    identity_check = matrix_multiply(cov, cov_inv)
    print(f"\n  Verification (Cov × Cov⁻¹ ≈ I):")
    for row in identity_check:
        print(f"    [{', '.join(f'{v:.4f}' for v in row)}]")

    # Step 5: Test point - "overcast, hot, normal" → decoded: (2, 1, 2)
    test_point = [2, 1, 2]
    print(f"\nStep 5: Test point (overcast, hot, normal) = {test_point}")

    diff = [test_point[j] - mean[j] for j in range(p)]
    print(f"  Difference (x - mean) = [{', '.join(f'{v:.4f}' for v in diff)}]")

    # Step 6: Compute (x - mean)^T * S^-1 * (x - mean)
    # First: S^-1 * diff  (3x3 * 3x1 = 3x1)
    diff_col = [[d] for d in diff]  # column vector
    sinv_diff = matrix_multiply(cov_inv, diff_col)
    print(f"  S⁻¹ × (x - mean) = [{', '.join(f'{v[0]:.6f}' for v in sinv_diff)}]")

    # diff^T * (S^-1 * diff)
    diff_row = [diff]  # row vector
    result = matrix_multiply(diff_row, sinv_diff)
    mahal_sq = result[0][0]
    mahal = math.sqrt(mahal_sq)

    print(f"\n  (x - mean)ᵀ × S⁻¹ × (x - mean) = {mahal_sq:.6f}")
    print(f"  Mahalanobis Distance = √{mahal_sq:.6f} = {mahal:.6f}")


# ─────────────────────────────────────────────
# Problem 3: Cosine and Extended Jaccard Similarity
# ─────────────────────────────────────────────
def cosine_similarity(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    mag_a = math.sqrt(sum(ai ** 2 for ai in a))
    mag_b = math.sqrt(sum(bi ** 2 for bi in b))
    return dot / (mag_a * mag_b)


def extended_jaccard_similarity(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    mag_a_sq = sum(ai ** 2 for ai in a)
    mag_b_sq = sum(bi ** 2 for bi in b)
    return dot / (mag_a_sq + mag_b_sq - dot)


def solve_problem3(data):
    print("\n" + "=" * 70)
    print("PROBLEM 3: Cosine and Extended Jaccard Similarity")
    print("=" * 70)

    dp20 = get_feature_vector(data[19])  # 0-indexed
    dp27 = get_feature_vector(data[26])

    print(f"\nData Point 20: {dp20}")
    print(f"Data Point 27: {dp27}")

    # Dot product
    dot = sum(a * b for a, b in zip(dp20, dp27))
    mag20_sq = sum(a ** 2 for a in dp20)
    mag27_sq = sum(a ** 2 for a in dp27)
    mag20 = math.sqrt(mag20_sq)
    mag27 = math.sqrt(mag27_sq)

    print(f"\n  Dot product (x·y) = {dot}")
    print(f"  ||x||² = {mag20_sq}, ||y||² = {mag27_sq}")
    print(f"  ||x|| = {mag20:.6f}, ||y|| = {mag27:.6f}")

    # (a) Cosine Similarity
    cos_sim = cosine_similarity(dp20, dp27)
    print(f"\n  (a) Cosine Similarity = {dot} / ({mag20:.6f} × {mag27:.6f})")
    print(f"      = {dot} / {mag20 * mag27:.6f}")
    print(f"      = {cos_sim:.6f}")

    # (b) Extended Jaccard Similarity
    ej_sim = extended_jaccard_similarity(dp20, dp27)
    denom = mag20_sq + mag27_sq - dot
    print(f"\n  (b) Extended Jaccard Similarity = {dot} / ({mag20_sq} + {mag27_sq} - {dot})")
    print(f"      = {dot} / {denom}")
    print(f"      = {ej_sim:.6f}")


# ─────────────────────────────────────────────
# Problem 4: Entropy
# ─────────────────────────────────────────────
def entropy(values):
    """Calculate Shannon entropy of a list of values."""
    n = len(values)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    ent = 0.0
    for label, count in sorted(counts.items()):
        p = count / n
        if p > 0:
            ent -= p * math.log2(p)
    return ent, counts


def solve_problem4(data):
    print("\n" + "=" * 70)
    print("PROBLEM 4: Entropy of Temperature and Humidity")
    print("=" * 70)

    n = len(data)

    # Mapping for display
    temp_labels = {1: 'hot', 2: 'mild', 3: 'cool'}
    humidity_labels = {1: 'high', 2: 'normal'}

    for attr, labels in [('temp', temp_labels), ('humidity', humidity_labels)]:
        values = [row[attr] for row in data]
        ent, counts = entropy(values)
        num_classes = len(counts)
        max_ent = math.log2(num_classes)

        print(f"\n--- (a) Entropy of '{attr}' ---" if attr == 'temp' else f"\n--- (b) Entropy of '{attr}' ---")
        print(f"  Total data points: {n}")
        print(f"  Value counts:")
        for val in sorted(counts.keys()):
            print(f"    {labels[val]} ({val}): {counts[val]} → p = {counts[val]}/{n} = {counts[val]/n:.6f}")

        print(f"\n  H({attr}) = ", end="")
        terms = []
        for val in sorted(counts.keys()):
            p = counts[val] / n
            term_val = -p * math.log2(p)
            terms.append(f"-({counts[val]}/{n})log₂({counts[val]}/{n})")
            print(f"{term_val:.6f}", end=" + " if val != max(counts.keys()) else "")
        print(f"\n  H({attr}) = {ent:.6f} bits")
        print(f"\n  Number of classes = {num_classes}")
        print(f"  Maximum entropy = log₂({num_classes}) = {max_ent:.6f} bits")


# ─────────────────────────────────────────────
# Problem 5: Mutual Information of Outlook and Temperature
# ─────────────────────────────────────────────
def solve_problem5(data):
    print("\n" + "=" * 70)
    print("PROBLEM 5: Mutual Information of Outlook and Temperature")
    print("=" * 70)

    n = len(data)
    outlook_labels = {1: 'sunny', 2: 'overcast', 3: 'rainy'}
    temp_labels = {1: 'hot', 2: 'mild', 3: 'cool'}

    # Marginal distributions
    outlook_counts = {}
    temp_counts = {}
    joint_counts = {}

    for row in data:
        o = row['outlook']
        t = row['temp']
        outlook_counts[o] = outlook_counts.get(o, 0) + 1
        temp_counts[t] = temp_counts.get(t, 0) + 1
        key = (o, t)
        joint_counts[key] = joint_counts.get(key, 0) + 1

    print(f"\nTotal data points: {n}")

    # Marginal: Outlook
    print(f"\nMarginal Distribution - Outlook:")
    for val in sorted(outlook_counts.keys()):
        p = outlook_counts[val] / n
        print(f"  P({outlook_labels[val]}) = {outlook_counts[val]}/{n} = {p:.6f}")

    # Marginal: Temperature
    print(f"\nMarginal Distribution - Temperature:")
    for val in sorted(temp_counts.keys()):
        p = temp_counts[val] / n
        print(f"  P({temp_labels[val]}) = {temp_counts[val]}/{n} = {p:.6f}")

    # Joint distribution
    print(f"\nJoint Distribution P(Outlook, Temperature):")
    # Print as table
    header = "  {:>12s}".format("")
    for t_val in sorted(temp_counts.keys()):
        header += f"  {temp_labels[t_val]:>10s}"
    header += f"  {'Total':>10s}"
    print(header)

    for o_val in sorted(outlook_counts.keys()):
        row_str = f"  {outlook_labels[o_val]:>12s}"
        for t_val in sorted(temp_counts.keys()):
            count = joint_counts.get((o_val, t_val), 0)
            row_str += f"  {count:>10d}"
        row_str += f"  {outlook_counts[o_val]:>10d}"
        print(row_str)

    total_row = f"  {'Total':>12s}"
    for t_val in sorted(temp_counts.keys()):
        total_row += f"  {temp_counts[t_val]:>10d}"
    total_row += f"  {n:>10d}"
    print(total_row)

    # Entropy of Outlook
    h_outlook = 0.0
    for val in sorted(outlook_counts.keys()):
        p = outlook_counts[val] / n
        if p > 0:
            h_outlook -= p * math.log2(p)
    print(f"\nH(Outlook) = {h_outlook:.6f} bits")

    # Entropy of Temperature
    h_temp = 0.0
    for val in sorted(temp_counts.keys()):
        p = temp_counts[val] / n
        if p > 0:
            h_temp -= p * math.log2(p)
    print(f"H(Temperature) = {h_temp:.6f} bits")

    # Joint Entropy
    h_joint = 0.0
    print(f"\nJoint Entropy H(Outlook, Temperature) calculation:")
    for o_val in sorted(outlook_counts.keys()):
        for t_val in sorted(temp_counts.keys()):
            count = joint_counts.get((o_val, t_val), 0)
            if count > 0:
                p_joint = count / n
                term = -p_joint * math.log2(p_joint)
                h_joint += term
                print(f"  P({outlook_labels[o_val]},{temp_labels[t_val]}) = {count}/{n} = {p_joint:.6f}, "
                      f"-p·log₂(p) = {term:.6f}")
    print(f"\nH(Outlook, Temperature) = {h_joint:.6f} bits")

    # Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = h_outlook + h_temp - h_joint
    print(f"\nMutual Information I(Outlook; Temperature):")
    print(f"  I = H(Outlook) + H(Temperature) - H(Outlook, Temperature)")
    print(f"  I = {h_outlook:.6f} + {h_temp:.6f} - {h_joint:.6f}")
    print(f"  I = {mi:.6f} bits")

    # Alternative verification: sum of p(x,y) * log(p(x,y) / (p(x)*p(y)))
    print(f"\nVerification using definition: I = Σ p(x,y) · log₂(p(x,y) / (p(x)·p(y)))")
    mi_verify = 0.0
    for o_val in sorted(outlook_counts.keys()):
        for t_val in sorted(temp_counts.keys()):
            count = joint_counts.get((o_val, t_val), 0)
            if count > 0:
                p_joint = count / n
                p_o = outlook_counts[o_val] / n
                p_t = temp_counts[t_val] / n
                term = p_joint * math.log2(p_joint / (p_o * p_t))
                mi_verify += term
                print(f"  P({outlook_labels[o_val]},{temp_labels[t_val]}) = {p_joint:.6f}, "
                      f"P({outlook_labels[o_val]})·P({temp_labels[t_val]}) = {p_o:.6f}×{p_t:.6f} = {p_o*p_t:.6f}, "
                      f"term = {term:.6f}")
    print(f"\n  I(Outlook; Temperature) = {mi_verify:.6f} bits")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Determine data file path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', 'data', 'decoded.csv')
    data_path = os.path.normpath(data_path)

    print(f"Loading data from: {data_path}")
    data = load_decoded_data(data_path)
    print(f"Loaded {len(data)} data points.\n")

    # Print first few and last few rows for verification
    print("First 3 rows:", [get_feature_vector(d) for d in data[:3]])
    print("Last 3 rows:", [get_feature_vector(d) for d in data[-3:]])

    solve_problem1(data)
    solve_problem2(data)
    solve_problem3(data)
    solve_problem4(data)
    solve_problem5(data)


if __name__ == '__main__':
    main()
