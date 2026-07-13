"""
Geometry helpers used for reading-order sorting, label/value pairing,
line grouping, and table clustering.
"""
from typing import List, Tuple


def vertical_overlap_ratio(box_a, box_b) -> float:
    """How much two boxes overlap vertically, relative to the shorter box."""
    top = max(box_a.y1, box_b.y1)
    bottom = min(box_a.y2, box_b.y2)
    overlap = max(0.0, bottom - top)
    shorter = min(box_a.height, box_b.height)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def same_line(box_a, box_b, overlap_threshold: float = 0.5) -> bool:
    """Two boxes are considered on the same line if they overlap vertically enough."""
    return vertical_overlap_ratio(box_a, box_b) >= overlap_threshold


def is_right_of(label_box, value_box, max_gap: float = None) -> bool:
    """Value box starts to the right of the label box (same line)."""
    if value_box.x1 < label_box.x2 - 2:
        return False
    if max_gap is not None and (value_box.x1 - label_box.x2) > max_gap:
        return False
    return True


def is_below(label_box, value_box, max_gap: float = None) -> bool:
    """Value box sits below the label box, horizontally aligned (roughly)."""
    if value_box.y1 < label_box.y2 - 2:
        return False
    if max_gap is not None and (value_box.y1 - label_box.y2) > max_gap:
        return False
    # require some horizontal overlap so we don't grab a value from a totally
    # different column
    h_overlap = min(label_box.x2, value_box.x2) - max(label_box.x1, value_box.x1)
    return h_overlap > -label_box.width  # lenient horizontal check


def euclidean_distance(box_a, box_b) -> float:
    return ((box_a.cx - box_b.cx) ** 2 + (box_a.cy - box_b.cy) ** 2) ** 0.5


def group_into_lines(tokens: List, y_overlap_threshold: float = 0.5) -> List[List[int]]:
    """
    Groups token indices into visual lines based on vertical (y) overlap.
    Returns a list of lines, each a list of token indices, sorted top-to-bottom
    then left-to-right within the line.
    """
    if not tokens:
        return []

    indices = sorted(range(len(tokens)), key=lambda i: tokens[i].bbox.y1)
    lines: List[List[int]] = []

    for idx in indices:
        box = tokens[idx].bbox
        placed = False
        for line in lines:
            # compare against the first token's box in the line as reference
            ref_box = tokens[line[0]].bbox
            if same_line(ref_box, box, y_overlap_threshold):
                line.append(idx)
                placed = True
                break
        if not placed:
            lines.append([idx])

    # sort tokens inside each line left-to-right, and sort lines top-to-bottom
    for line in lines:
        line.sort(key=lambda i: tokens[i].bbox.x1)
    lines.sort(key=lambda line: min(tokens[i].bbox.y1 for i in line))

    return lines


def cluster_columns(x_centers: List[float], gap_threshold: float = 20.0) -> List[Tuple[float, float]]:
    """
    Clusters a list of x-center coordinates into column ranges. Used by the
    table parser to figure out how many columns exist.
    """
    if not x_centers:
        return []
    xs = sorted(x_centers)
    clusters = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] <= gap_threshold:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [(min(c), max(c)) for c in clusters]
